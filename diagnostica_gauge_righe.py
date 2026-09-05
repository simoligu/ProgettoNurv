# -*- coding: utf-8 -*-
"""
Diagnostica riga per riga: per UN frame specifico, mostra esattamente cosa
succede dentro _seleziona_coppia_rotaie() per ciascuna delle 20 righe
campionate — quali run sono stati trovati, quale e' stato scelto come
"ancora" (rotaia piu' vicina al centro), quale vicino e' stato abbinato, e
il gap risultante.

Utile quando un alert SCARTAMENTO_ANOMALO sembra visivamente sbagliato (il
riquadro appare sul binario giusto) ma la confidenza e' comunque altissima:
la causa e' spesso in poche righe "cattive" che spostano la mediana, non in
un errore sistematico di selezione del binario.

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python diagnostica_gauge_righe.py --video data/videos/query.mp4 \
        --frame_idx 1110 --deeplab runs_seg/deeplab_hires/best.pt
"""

import argparse
import cv2
import numpy as np

from deeplab_analyzer import DeepLabAnalyzer, CL_ROTAIE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame_idx", type=int, required=True)
    ap.add_argument("--deeplab", default="runs_seg/deeplab_hires/best.pt")
    ap.add_argument("--imgsz", type=int, default=896)
    args = ap.parse_args()

    analyzer = DeepLabAnalyzer(weights_path=args.deeplab, imgsz=args.imgsz)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"[ERRORE] Impossibile leggere il frame {args.frame_idx} da {args.video}")
        return

    h, w = frame.shape[:2]
    class_map = analyzer.segment(frame)
    rail_mask = (class_map == CL_ROTAIE).astype(np.uint8)

    y_start = int(h * analyzer.gauge_roi_top_frac)
    y_end = int(h * analyzer.gauge_roi_bottom_frac)
    rows_to_check = np.linspace(y_start, y_end, num=20, dtype=int)

    center_x = w / 2.0
    half_band = (w * analyzer.gauge_center_band_frac) / 2.0
    print(f"Frame {args.frame_idx} | dimensioni {w}x{h} | centro immagine x={center_x:.0f} | "
          f"fascia centrale accettata: [{center_x-half_band:.0f}, {center_x+half_band:.0f}]")
    print(f"Fascia verticale di scansione: y=[{y_start}, {y_end}]\n")

    distances = []
    for y in rows_to_check:
        row = rail_mask[y, :]
        tutti_i_run = analyzer._find_runs(row)
        coppia = analyzer._seleziona_coppia_rotaie(row, w)
        if coppia is not None:
            r1, r2 = coppia
            gap = r2[0] - r1[1]
            distances.append(gap)
            print(f"  y={y:4d} | run trovati: {tutti_i_run} -> coppia scelta: {r1},{r2} -> gap={gap}px")
        else:
            print(f"  y={y:4d} | run trovati: {tutti_i_run} -> NESSUNA coppia valida (riga scartata)")

    print(f"\n{len(distances)}/20 righe valide.")
    if distances:
        gauge = float(np.median(distances))
        std_dev = float(np.std(distances))
        cv = std_dev / gauge if gauge > 0 else 0
        print(f"Mediana: {gauge:.1f}px | Std: {std_dev:.1f}px | Coeff. variazione: {cv:.2f} "
              f"(soglia scarto: {analyzer.gauge_cv_threshold})")
        print(f"Valori grezzi ordinati: {sorted(distances)}")
        print(f"Confronto con expected_gauge_px corrente: {analyzer.expected_gauge_px}")

    overlay = analyzer.get_colored_overlay(frame, class_map)
    # disegna anche la fascia di scansione e la fascia centrale, per contesto visivo
    cv2.rectangle(overlay, (0, y_start), (w, y_end), (0, 255, 255), 2)
    cv2.line(overlay, (int(center_x - half_band), 0), (int(center_x - half_band), h), (255, 0, 255), 1)
    cv2.line(overlay, (int(center_x + half_band), 0), (int(center_x + half_band), h), (255, 0, 255), 1)
    out_path = f"diagnostica_overlay_frame_{args.frame_idx}.jpg"
    cv2.imwrite(out_path, overlay)
    print(f"\nOverlay salvato in: {out_path} (giallo = fascia di scansione, magenta = fascia centrale)")


if __name__ == "__main__":
    main()