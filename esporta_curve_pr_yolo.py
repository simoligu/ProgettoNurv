# -*- coding: utf-8 -*-
"""
Esporta i DATI NUMERICI delle curve precision-recall sulle maschere del modello
YOLOv8-seg small (train_s_v2), che Ultralytics salva soltanto come immagine.

Serve alla Figura 5.2 della tesi: oggi e' ridisegnata ricavando le curve
dall'immagine MaskPR_curve.png; con questi dati la si ridisegna dai valori veri.

Rifa' la validazione sulle 1700 immagini di validazione con i pesi migliori e
scrive un CSV con una riga per valore di recall (1000 punti) e una colonna di
precision per classe, piu' la media delle classi.

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python esporta_curve_pr_yolo.py
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/segment/runs_nurv/train_s_v2/weights/best.pt")
    ap.add_argument("--data", default="data/rs19_val/dataset.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="risultati_tesi/curve_pr_yolo_small.csv")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"[ERRORE] Pesi non trovati: {args.weights}")
        return

    modello = YOLO(args.weights)
    r = modello.val(data=args.data, imgsz=args.imgsz, batch=args.batch,
                    split="val", plots=False, verbose=False)

    seg = r.seg                                  # metriche sulle maschere (M)
    recall = np.asarray(seg.px)                  # 1000 valori di recall in [0, 1]
    prec = np.asarray(seg.prec_values)           # [classi presenti, 1000]
    indici = list(r.ap_class_index)              # indice di classe di ogni riga
    nomi = [modello.names[int(i)] for i in indici]
    ap50 = np.asarray(seg.ap50)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["recall"] + nomi + ["media"])
        media = prec.mean(axis=0)
        for k in range(len(recall)):
            w.writerow([f"{recall[k]:.5f}"] + [f"{prec[c, k]:.5f}" for c in range(len(nomi))]
                       + [f"{media[k]:.5f}"])

    print(f"\nCurve scritte in {out}")
    print("AP@50 sulle maschere per classe (confrontare con la legenda della Figura 5.2):")
    for n, v in zip(nomi, ap50):
        print(f"  {n:12s} {v:.3f}")
    print(f"  {'media':12s} {seg.map50:.3f}   (mAP@50 sulle maschere)")


if __name__ == "__main__":
    main()
