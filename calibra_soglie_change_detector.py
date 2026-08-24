# -*- coding: utf-8 -*-
"""
Calibrazione empirica di diff_thresh / min_area per ChangeDetector.

I valori attuali (diff_thresh=90, min_area=8000) sembrano tarati per un
contenuto diverso da questo video: anche su frame con contenuto ben
corrispondente (stessa stazione, stesso punto della tratta), la diff media
resta 55-65 con min_area cosi' basso da bloccare quasi ogni frame, anche in
assenza di vere anomalie.

Questo script campiona N frame query CON corrispondenza valida nella mappa
di sincronizzazione (quindi presumibilmente "puliti", nessuna vera anomalia
strutturale nota in questo tratto), applica l'INTERA pipeline di confronto
(omografia validata + normalizzazione luminanza, stessa logica di
pipeline.py) e prova una griglia di (diff_thresh, min_area), riportando per
ognuna quanti box vengono generati sui frame "puliti" — un valore vicino a
zero indica una soglia che silenzia il rumore di fondo senza inseguire ogni
variazione naturale tra le due riprese.

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python calibra_soglie_change_detector.py --ref data/videos/reference.mp4 \
        --query data/videos/query.mp4 --sync_map mappa_sync.csv --n_test 30
"""

import argparse
from pathlib import Path
import numpy as np
import cv2

from alignment import FrameAligner
from background import mask_to_boxes
from sync_map import MappaSincronizzazione


def omografia_plausibile(H, w, h, margine_frazione=0.5):
    """Stessa logica di AnomalyDetectionPipeline._omografia_plausibile."""
    angoli = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    try:
        angoli_trasformati = cv2.perspectiveTransform(angoli, H).reshape(-1, 2)
    except Exception:
        return False
    margine_x, margine_y = w * margine_frazione, h * margine_frazione
    if np.any(angoli_trasformati[:, 0] < -margine_x) or np.any(angoli_trasformati[:, 0] > w + margine_x):
        return False
    if np.any(angoli_trasformati[:, 1] < -margine_y) or np.any(angoli_trasformati[:, 1] > h + margine_y):
        return False
    area_trasformata = cv2.contourArea(angoli_trasformati.astype(np.float32))
    area_originale = w * h
    if area_originale <= 0:
        return False
    rapporto = area_trasformata / area_originale
    return 0.3 <= rapporto <= 3.0


def normalizza_luminanza(sorgente_gray, target_gray):
    """Stessa logica di AnomalyDetectionPipeline._normalizza_luminanza."""
    scarto = float(target_gray.mean()) - float(sorgente_gray.mean())
    return np.clip(sorgente_gray.astype(np.float32) + scarto, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--sync_map", required=True)
    ap.add_argument("--n_test", type=int, default=30)
    ap.add_argument("--diff_thresh_values", type=int, nargs="+", default=[90, 110, 130, 150, 170])
    ap.add_argument("--min_area_values", type=int, nargs="+", default=[8000, 20000, 40000, 60000, 80000])
    args = ap.parse_args()

    print("=" * 70)
    print(" CALIBRAZIONE diff_thresh / min_area")
    print("=" * 70)

    mappa = MappaSincronizzazione(args.sync_map)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    aligner = FrameAligner()

    ref_cap = cv2.VideoCapture(args.ref)
    query_cap = cv2.VideoCapture(args.query)
    if not ref_cap.isOpened() or not query_cap.isOpened():
        print("[ERRORE] Impossibile aprire uno dei due video.")
        return

    total_frames = int(query_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // args.n_test)

    # --- raccogli le mappe diff (target_confronto, aligned_gray_norm) per i
    #     frame "puliti" testati, una volta sola, poi si prova la griglia di
    #     soglie su questi stessi dati (niente bisogno di rileggere i video
    #     per ogni combinazione) ---
    coppie_diff = []  # lista di (target_confronto, aligned_gray_norm)

    idx = 0
    tested = 0
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
            tested += 1
            idx += 1
            continue

        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, indice_ref)
        ok_ref, rf = ref_cap.read()
        if not ok_ref:
            tested += 1
            idx += 1
            continue
        rf = cv2.resize(rf, (w, h))

        qf_gray_eq = clahe_obj.apply(cv2.cvtColor(qf, cv2.COLOR_BGR2GRAY))
        rf_gray_eq = clahe_obj.apply(cv2.cvtColor(rf, cv2.COLOR_BGR2GRAY))

        H = aligner.compute_homography(rf_gray_eq, qf_gray_eq)
        if H is not None and np.abs(np.linalg.det(H)) > 0.1 and omografia_plausibile(H, w, h):
            aligned = cv2.warpPerspective(qf, H, (w, h), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        else:
            aligned = cv2.resize(qf, (w, h))

        aligned_gray_eq = clahe_obj.apply(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
        aligned_gray_norm = normalizza_luminanza(aligned_gray_eq, rf_gray_eq)

        coppie_diff.append((rf_gray_eq, aligned_gray_norm))

        tested += 1
        idx += 1

    ref_cap.release()
    query_cap.release()

    print(f"\nRaccolti {len(coppie_diff)} frame 'puliti' (con corrispondenza valida) per la calibrazione.\n")
    if not coppie_diff:
        print("[ATTENZIONE] Nessun frame valido raccolto — prova un --n_test piu' alto.")
        return

    # --- griglia di calibrazione ---
    print(f"{'diff_thresh':>12}{'min_area':>12}", end="")
    for _ in args.min_area_values:
        pass
    header = f"{'diff_thresh':>12}" + "".join(f"{ma:>12}" for ma in args.min_area_values)
    print(header)
    print("-" * len(header))

    for dt in args.diff_thresh_values:
        riga = f"{dt:>12}"
        for ma in args.min_area_values:
            n_box_tot = 0
            n_frame_con_box = 0
            for target_confronto, aligned_gray_norm in coppie_diff:
                diff = cv2.absdiff(target_confronto, aligned_gray_norm)
                boxes, _ = mask_to_boxes(diff, diff_thresh=dt, min_area=ma)
                n_box_tot += len(boxes)
                if boxes:
                    n_frame_con_box += 1
            riga += f"{n_box_tot:>12}"
        print(riga)

    print("\nOgni cella: numero TOTALE di box generati su tutti i frame 'puliti' testati,")
    print("per quella combinazione (diff_thresh, min_area). Vicino a 0 = buona scelta")
    print("(silenzia il rumore di fondo). Scegli la combinazione con il valore piu'")
    print("basso possibile SENZA esagerare (una soglia troppo alta rischierebbe di non")
    print("rilevare piu' anomalie vere — se hai un frame con un'anomalia nota, verificala")
    print("separatamente con la soglia scelta prima di adottarla in produzione).")


if __name__ == "__main__":
    main()
