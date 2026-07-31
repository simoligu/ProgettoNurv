# -*- coding: utf-8 -*-
"""
Diagnostica CHANGE DETECTOR — capisce perche' ANOMALIA_STRUTTURALE spara su quasi
ogni frame invece di scattare solo su cambiamenti reali.

Controlla tre possibili cause:
  1. L'omografia fallisce spesso -> fallback a resize senza allineamento -> diff enorme ovunque
  2. Differenza di luminosita' tra reference e query -> diff alta anche senza vere anomalie
  3. Soglie (diff_thresh, min_area) non calibrate per questo video

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python diagnostica_change_detector.py --ref data/videos/reference.mp4 --query data/videos/query.mp4
    python diagnostica_change_detector.py --ref data/videos/reference.mp4 --query data/videos/query.mp4 --n_test 30
"""

import argparse
from pathlib import Path
import numpy as np
import cv2

from video_io import VideoIO
from alignment import FrameAligner
from background import BackgroundModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Video di reference")
    ap.add_argument("--query", required=True, help="Video di query")
    ap.add_argument("--n_test", type=int, default=30, help="Numero di frame query da testare")
    ap.add_argument("--diff_thresh", type=int, default=90, help="Soglia diff attuale della pipeline")
    args = ap.parse_args()

    out_dir = Path("diag_change_detector")
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(" DIAGNOSTICA CHANGE DETECTOR")
    print("=" * 70)

    # --- costruisci il background come fa la pipeline ---
    print("\n[1/3] Costruzione background di riferimento...")
    ref_frames_tuple = VideoIO.sample_frames(args.ref, max_frames=200, step=8)
    ref_anchor = ref_frames_tuple[len(ref_frames_tuple) // 2][1]
    h, w = ref_anchor.shape[:2]

    only_frames = [f[1] for f in ref_frames_tuple]
    ref_median = BackgroundModel.build_median_background(
        only_frames, sample_n=50, resize_to=(w, h), use_luminance=True
    )

    # luminanza del reference
    ref_lum = BackgroundModel.estimate_luminance(ref_median)
    print(f"       Background costruito ({w}x{h}). Luminanza media: {ref_lum:.1f}")

    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # FIX: omografia sul frame nitido (ref_anchor), non sul background sfocato
    ref_anchor_gray = clahe_obj.apply(cv2.cvtColor(ref_anchor, cv2.COLOR_BGR2GRAY))
    # FIX: background equalizzato per il confronto, per normalizzare l'illuminazione
    ref_median_gray_eq = clahe_obj.apply(cv2.cvtColor(ref_median, cv2.COLOR_BGR2GRAY))

    # --- testa n_test frame del query video ---
    print(f"\n[2/3] Test su {args.n_test} frame del query video...")
    cap = cv2.VideoCapture(args.query)
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire {args.query}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // args.n_test)

    aligner = FrameAligner()

    homography_ok = 0
    homography_fail = 0
    diff_means = []
    diff_max_list = []
    pct_above_thresh_list = []
    query_lums = []

    idx = 0
    tested = 0
    while tested < args.n_test:
        ok, qf = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue

        qf_gray = cv2.cvtColor(qf, cv2.COLOR_BGR2GRAY)
        qf_gray_eq = clahe_obj.apply(qf_gray)
        q_lum = BackgroundModel.estimate_luminance(qf)
        query_lums.append(q_lum)

        # FIX: prova l'omografia sul frame nitido (ref_anchor_gray), non sul background sfocato
        H = aligner.compute_homography(ref_anchor_gray, qf_gray_eq)
        homography_valid = H is not None and np.abs(np.linalg.det(H)) > 0.1

        if homography_valid:
            homography_ok += 1
            aligned = cv2.warpPerspective(qf, H, (w, h), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        else:
            homography_fail += 1
            aligned = cv2.resize(qf, (w, h))

        # FIX: calcola la diff su versioni equalizzate CLAHE (normalizza l'illuminazione)
        al_gray_cmp = clahe_obj.apply(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
        diff = cv2.absdiff(ref_median_gray_eq, al_gray_cmp)

        diff_means.append(float(diff.mean()))
        diff_max_list.append(int(diff.max()))
        pct_above = 100.0 * np.sum(diff > args.diff_thresh) / diff.size
        pct_above_thresh_list.append(pct_above)

        # salva le prime 5 diff per ispezione visiva
        if tested < 5:
            vis = np.hstack([
                cv2.resize(cv2.cvtColor(ref_median_gray_eq, cv2.COLOR_GRAY2BGR), (w // 2, h // 2)),
                cv2.resize(cv2.cvtColor(al_gray_cmp, cv2.COLOR_GRAY2BGR), (w // 2, h // 2)),
                cv2.resize(cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR), (w // 2, h // 2)),
            ])
            cv2.putText(vis, f"frame {idx} | homography={'OK' if homography_valid else 'FALLBACK'} | "
                             f"diff_mean={diff.mean():.1f} | %>thresh={pct_above:.1f}%",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imwrite(str(out_dir / f"diag_frame{idx:05d}.jpg"), vis)

        tested += 1
        idx += 1

    cap.release()

    # --- report ---
    print(f"\n[3/3] Risultati su {tested} frame testati:")
    print("-" * 70)
    print(f"  Omografia riuscita:  {homography_ok}/{tested}  ({100*homography_ok/tested:.0f}%)")
    print(f"  Omografia fallita:   {homography_fail}/{tested}  ({100*homography_fail/tested:.0f}%)  "
          f"(fallback a resize semplice)")
    print("-" * 70)
    print(f"  Luminanza reference:      {ref_lum:.1f}")
    print(f"  Luminanza query (media):  {np.mean(query_lums):.1f}")
    print(f"  Luminanza query (range):  {np.min(query_lums):.1f} - {np.max(query_lums):.1f}")
    diff_lum = abs(ref_lum - np.mean(query_lums))
    print(f"  Differenza luminanza:     {diff_lum:.1f}  "
          f"{'<-- SIGNIFICATIVA, possibile causa' if diff_lum > 15 else '(ok, non sembra il problema)'}")
    print("-" * 70)
    print(f"  Diff media sui pixel:     {np.mean(diff_means):.1f}  (soglia attuale: {args.diff_thresh})")
    print(f"  Diff massima osservata:   {np.max(diff_max_list)}")
    print(f"  %% pixel sopra soglia:    {np.mean(pct_above_thresh_list):.1f}%  (media sui frame testati)")
    print("=" * 70)

    print("\nINTERPRETAZIONE:")
    if homography_fail / tested > 0.3:
        print(f"  -> L'omografia fallisce nel {100*homography_fail/tested:.0f}% dei frame testati.")
        print(f"     Questo e' probabilmente causa primaria: senza allineamento corretto,")
        print(f"     ogni piccolo movimento di camera genera diff enormi ovunque.")
    if diff_lum > 15:
        print(f"  -> La differenza di luminanza ({diff_lum:.1f}) e' significativa.")
        print(f"     Reference e query sembrano girati in condizioni di luce diverse.")
        print(f"     Questo da solo puo' spiegare diff sistematicamente alte.")
    if homography_fail / tested <= 0.3 and diff_lum <= 15:
        print(f"  -> Omografia e luminanza sembrano a posto. Il problema e' probabilmente")
        print(f"     che diff_thresh={args.diff_thresh} e' troppo basso per il rumore/texture")
        print(f"     di questo video specifico. Guarda le immagini in {out_dir}/ per")
        print(f"     vedere quanto e' diffusa la diff anche in aree senza vere anomalie.")

    print(f"\nApri le immagini in '{out_dir}/' per ispezionare visivamente:")
    print("  pannello sinistro = background reference")
    print("  pannello centro   = frame query allineato")
    print("  pannello destro   = mappa della differenza (piu' chiaro = piu' diverso)")


if __name__ == "__main__":
    main()