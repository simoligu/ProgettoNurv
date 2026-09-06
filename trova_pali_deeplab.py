# -*- coding: utf-8 -*-
"""
Esegue la VERA segmentazione DeepLab su un frame e mostra tutti i candidati
"palo" individuati dalla stessa logica di DeepLabAnalyzer._analyze_poles()
(componenti connesse sulla maschera classe-pali, filtro area/verticalita',
minAreaRect per l'inclinazione geometrica) — inclusi quelli SOTTO soglia,
che la pipeline normale scarterebbe silenziosamente senza mostrarli.

Serve per ottenere un bbox REALE (non stimato a mano) da usare nei test di
validazione di PoleTiltAnalyzer — elimina il dubbio "il bbox disegnato a
mano è abbastanza fedele a quello che produrrebbe DeepLab davvero?".

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python trova_pali_deeplab.py --immagine frame_verifica/frame_000510.jpg \
        --deeplab runs_seg/deeplab_hires/best.pt
"""

import argparse

import cv2
import numpy as np

from deeplab_analyzer import DeepLabAnalyzer, CL_PALI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--immagine", required=True)
    ap.add_argument("--deeplab", default="runs_seg/deeplab_hires/best.pt")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--out", default=None,
                     help="Percorso immagine annotata di output (default: <immagine>_pali_trovati.jpg)")
    args = ap.parse_args()

    img = cv2.imread(args.immagine)
    if img is None:
        print(f"[ERRORE] Impossibile aprire {args.immagine}")
        return
    h, w = img.shape[:2]

    analyzer = DeepLabAnalyzer(weights_path=args.deeplab, imgsz=args.imgsz)
    class_map = analyzer.segment(img)

    # replica ESATTA della logica di individuazione candidati in _analyze_poles,
    # ma SENZA il filtro finale "tilt_finale > max_tilt_deg" — vogliamo vedere
    # tutti i candidati, non solo quelli che genererebbero un alert
    pole_mask = (class_map == CL_PALI).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pole_mask = cv2.morphologyEx(pole_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pole_mask, connectivity=8)

    candidati = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 500 or bh < h * 0.15 or bh < bw * 2:
            continue

        points = np.column_stack(np.where(labels == i))
        points_xy = points[:, ::-1].astype(np.float32)
        if len(points_xy) < 5:
            continue

        rect = cv2.minAreaRect(points_xy)
        rect_w, rect_h = rect[1]
        angle = rect[2]
        if rect_w > rect_h:
            rect_w, rect_h = rect_h, rect_w
            angle = angle + 90
        tilt_geometrico = abs(angle) if abs(angle) <= 45 else abs(90 - abs(angle))

        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        candidati.append((bx, by, bw, bh, area, tilt_geometrico))

    print(f"[INFO] Trovati {len(candidati)} candidati 'palo' (area>=500, verticali) nel frame\n")

    annotato = img.copy()
    for idx, (bx, by, bw, bh, area, tilt) in enumerate(candidati):
        print(f"  Candidato {idx}: bbox=({bx},{by},{bw},{bh})  area={area}px  "
              f"inclinazione geometrica stimata={tilt:.1f}°")
        cv2.rectangle(annotato, (bx, by), (bx + bw, by + bh), (0, 165, 255), 2)
        cv2.putText(annotato, f"#{idx} tilt={tilt:.1f}", (bx, max(by - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    out_path = args.out or args.immagine.rsplit(".", 1)[0] + "_pali_trovati.jpg"
    cv2.imwrite(out_path, annotato)
    print(f"\n[FATTO] Immagine annotata salvata in: {out_path}")
    if candidati:
        print("\nPer testare un candidato specifico con costruisci_crop_stile_training.py:")
        bx, by, bw, bh, _, _ = candidati[0]
        print(f"  python costruisci_crop_stile_training.py --immagine {args.immagine} "
              f"--bbox {bx},{by},{bw},{bh}")


if __name__ == "__main__":
    main()
