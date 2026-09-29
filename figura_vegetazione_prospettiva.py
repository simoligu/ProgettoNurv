# -*- coding: utf-8 -*-
"""
Figura per la tesi (cap. 6, par. 6.8-6.9): l'effetto prospettico sulla
segnalazione di vegetazione invasiva.

Sullo STESSO fotogramma esegue l'analisi della vegetazione due volte:
  - "prima":  veg_roi_top_frac = 0.0  -> analizzata l'intera immagine
  - "dopo":   veg_roi_top_frac = 0.55 -> analizzato solo il 45% inferiore
              (valore predefinito attuale di DeepLabAnalyzer)
e salva due immagini con i rettangoli delle segnalazioni, la maschera della
vegetazione semitrasparente e, nella versione "dopo", la linea che delimita la
zona analizzata. La segmentazione viene calcolata una sola volta.

Nota: la versione "prima" isola l'effetto della sola restrizione verticale;
gli altri tre filtri (cluster, dimensione minima, bordo inferiore) restano
attivi in entrambe, perche' sono codice e non parametri.

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python figura_vegetazione_prospettiva.py --video data/videos/query.mp4 --frame_idx 1110
"""

import argparse

import cv2
import numpy as np

from deeplab_analyzer import DeepLabAnalyzer, CL_VEGETAZIONE

VERDE = (0, 200, 0)


def disegna(frame, class_map, anomalie, roi_top_frac):
    out = frame.copy()
    h, w = out.shape[:2]
    # maschera della vegetazione semitrasparente
    veg = class_map == CL_VEGETAZIONE
    strato = out.copy()
    strato[veg] = VERDE
    out = cv2.addWeighted(strato, 0.30, out, 0.70, 0)
    # zona esclusa dall'analisi (solo se esiste)
    y_roi = int(h * roi_top_frac)
    if y_roi > 0:
        velo = out.copy()
        velo[:y_roi] = (60, 60, 60)
        out = cv2.addWeighted(velo, 0.35, out, 0.65, 0)
        cv2.line(out, (0, y_roi), (w - 1, y_roi), (255, 255, 255), 3)
    # rettangoli delle segnalazioni di vegetazione
    for a in anomalie:
        if a["label"] != "VEGETAZIONE_INVASIVA":
            continue
        x, y, bw, bh = a["bbox"]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 0, 255), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame_idx", type=int, required=True)
    ap.add_argument("--deeplab", default="runs_seg/deeplab_hires/best.pt")
    ap.add_argument("--imgsz", type=int, default=896)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"[ERRORE] Impossibile leggere il frame {args.frame_idx} da {args.video}")
        return

    analyzer = DeepLabAnalyzer(weights_path=args.deeplab, imgsz=args.imgsz)
    class_map = analyzer.segment(frame)
    h, w = class_map.shape
    roi_predefinita = analyzer.veg_roi_top_frac

    for nome, roi in (("prima", 0.0), ("dopo", roi_predefinita)):
        analyzer.veg_roi_top_frac = roi
        anomalie = analyzer._analyze_vegetation(class_map, h, w)
        immagine = disegna(frame, class_map, anomalie, roi)
        percorso = f"figura_vegetazione_{nome}_frame_{args.frame_idx}.jpg"
        cv2.imwrite(percorso, immagine, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[{nome}] zona analizzata dal {roi*100:.0f}% dell'altezza in giu': "
              f"{len(anomalie)} segnalazioni -> {percorso}")
        for a in anomalie:
            print(f"    {a['severity']:8s} bbox={a['bbox']}  {a['details']}")

    analyzer.veg_roi_top_frac = roi_predefinita


if __name__ == "__main__":
    main()
