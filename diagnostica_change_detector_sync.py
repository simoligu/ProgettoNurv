# -*- coding: utf-8 -*-
"""
Variante di diagnostica_change_detector.py che usa la MAPPA DI SINCRONIZZAZIONE
(un frame reference specifico per ogni frame query, quello scelto da
sync_videos_dtw.py) invece dell'unico background mediato fisso.

Serve a isolare una domanda precisa: l'omografia fallisce cosi' spesso (90%)
anche quando il confronto e' fatto contro il frame reference GIUSTO (stessa
posizione fisica sulla tratta), o solo perche' lo script diagnostico originale
confrontava contro un background mediato che rappresenta un tratto diverso
della linea? Se il problema persiste anche qui, la causa e' nell'omografia/
luminanza stessa, non nella sincronizzazione (che sappiamo gia' funzionare).

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python diagnostica_change_detector_sync.py --ref data/videos/reference.mp4 \
        --query data/videos/query.mp4 --sync_map mappa_sync.csv --n_test 30
"""

import argparse
from pathlib import Path
import numpy as np
import cv2

from alignment import FrameAligner
from background import BackgroundModel
from sync_map import MappaSincronizzazione


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Video di reference")
    ap.add_argument("--query", required=True, help="Video di query")
    ap.add_argument("--sync_map", required=True, help="CSV prodotto da sync_videos_dtw.py")
    ap.add_argument("--n_test", type=int, default=30, help="Numero di frame query da testare")
    ap.add_argument("--diff_thresh", type=int, default=90, help="Soglia diff attuale della pipeline")
    args = ap.parse_args()

    out_dir = Path("diag_change_detector_sync")
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(" DIAGNOSTICA CHANGE DETECTOR — CON MAPPA DI SINCRONIZZAZIONE")
    print("=" * 70)

    mappa = MappaSincronizzazione(args.sync_map)

    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    aligner = FrameAligner()

    ref_cap = cv2.VideoCapture(args.ref)
    if not ref_cap.isOpened():
        print(f"[ERRORE] Impossibile aprire {args.ref}")
        return

    query_cap = cv2.VideoCapture(args.query)
    if not query_cap.isOpened():
        print(f"[ERRORE] Impossibile aprire {args.query}")
        return

    total_frames = int(query_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // args.n_test)

    homography_ok = 0
    homography_fail = 0
    saltati_no_corrispondenza = 0
    diff_means = []
    diff_max_list = []
    pct_above_thresh_list = []
    ref_lums = []
    query_lums = []

    idx = 0
    tested = 0
    salvate_immagini = 0

    while tested < args.n_test:
        ok, qf = query_cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue

        h, w = qf.shape[:2]

        indice_ref = mappa.frame_reference_per(idx)
        if indice_ref is None:
            saltati_no_corrispondenza += 1
            tested += 1
            idx += 1
            continue

        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, indice_ref)
        ok_ref, rf = ref_cap.read()
        if not ok_ref:
            saltati_no_corrispondenza += 1
            tested += 1
            idx += 1
            continue
        rf = cv2.resize(rf, (w, h))

        qf_gray_eq = clahe_obj.apply(cv2.cvtColor(qf, cv2.COLOR_BGR2GRAY))
        rf_gray_eq = clahe_obj.apply(cv2.cvtColor(rf, cv2.COLOR_BGR2GRAY))

        q_lum = BackgroundModel.estimate_luminance(qf)
        r_lum = BackgroundModel.estimate_luminance(rf)
        query_lums.append(q_lum)
        ref_lums.append(r_lum)

        H = aligner.compute_homography(rf_gray_eq, qf_gray_eq)
        homography_valid = H is not None and np.abs(np.linalg.det(H)) > 0.1

        if homography_valid:
            homography_ok += 1
            aligned = cv2.warpPerspective(qf, H, (w, h), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        else:
            homography_fail += 1
            aligned = cv2.resize(qf, (w, h))

        al_gray_cmp = clahe_obj.apply(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
        diff = cv2.absdiff(rf_gray_eq, al_gray_cmp)

        diff_means.append(float(diff.mean()))
        diff_max_list.append(int(diff.max()))
        pct_above = 100.0 * np.sum(diff > args.diff_thresh) / diff.size
        pct_above_thresh_list.append(pct_above)

        if salvate_immagini < 5:
            vis = np.hstack([
                cv2.resize(cv2.cvtColor(rf_gray_eq, cv2.COLOR_GRAY2BGR), (w // 2, h // 2)),
                cv2.resize(cv2.cvtColor(al_gray_cmp, cv2.COLOR_GRAY2BGR), (w // 2, h // 2)),
                cv2.resize(cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR), (w // 2, h // 2)),
            ])
            cv2.putText(vis, f"query#{idx} vs ref#{indice_ref} | homography={'OK' if homography_valid else 'FALLBACK'} | "
                             f"diff_mean={diff.mean():.1f} | %>thresh={pct_above:.1f}%",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imwrite(str(out_dir / f"diag_sync_q{idx:05d}_r{indice_ref:05d}.jpg"), vis)
            salvate_immagini += 1

        tested += 1
        idx += 1

    ref_cap.release()
    query_cap.release()

    n_confrontati = tested - saltati_no_corrispondenza

    print(f"\nRisultati su {tested} frame testati "
          f"({saltati_no_corrispondenza} senza corrispondenza valida, "
          f"{n_confrontati} confrontati):")
    print("-" * 70)
    if n_confrontati == 0:
        print("[ATTENZIONE] Nessun frame con corrispondenza valida nel range testato — "
              "prova un --n_test piu' alto o controlla il range coperto dalla mappa.")
        return

    print(f"  Omografia riuscita:  {homography_ok}/{n_confrontati}  ({100*homography_ok/n_confrontati:.0f}%)")
    print(f"  Omografia fallita:   {homography_fail}/{n_confrontati}  ({100*homography_fail/n_confrontati:.0f}%)")
    print("-" * 70)
    print(f"  Luminanza reference (media sui frame confrontati):  {np.mean(ref_lums):.1f}")
    print(f"  Luminanza query (media):                            {np.mean(query_lums):.1f}")
    diff_lum = abs(np.mean(ref_lums) - np.mean(query_lums))
    print(f"  Differenza luminanza media:                         {diff_lum:.1f}  "
          f"{'<-- SIGNIFICATIVA' if diff_lum > 15 else '(ok)'}")
    print("-" * 70)
    print(f"  Diff media sui pixel:     {np.mean(diff_means):.1f}  (soglia attuale: {args.diff_thresh})")
    print(f"  Diff massima osservata:   {np.max(diff_max_list)}")
    print(f"  %% pixel sopra soglia:    {np.mean(pct_above_thresh_list):.1f}%")
    print("=" * 70)

    print("\nCONFRONTO CON LO SCRIPT ORIGINALE (background mediato fisso):")
    print("  Se qui l'omografia fallisce MENO spesso e la diff e' piu' bassa,")
    print("  il problema principale era il confronto contro il background mediato")
    print("  (posizione fisica sbagliata) — la mappa di sincronizzazione risolve gran")
    print("  parte del problema, e conviene concentrarsi sull'ottimizzare la soglia.")
    print("  Se invece i numeri restano simili (omografia ~90% fallita, diff alta),")
    print("  il problema e' nell'omografia/luminanza stessa, indipendente dal fatto")
    print("  che il frame reference sia quello giusto — serve intervenire su")
    print("  alignment.py (es. SIFT invece di ORB, soglia RANSAC piu' permissiva) o")
    print("  su una normalizzazione di luminanza piu' aggressiva prima del confronto.")

    print(f"\nApri le immagini in '{out_dir}/' per ispezionare visivamente.")


if __name__ == "__main__":
    main()
