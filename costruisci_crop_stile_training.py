# -*- coding: utf-8 -*-
"""
Costruisce un crop nello STESSO stile usato in training (vedi crop_dataset.py
e il docstring di PoleTiltAnalyzer): margine di sicurezza di 0.55x l'altezza
del bbox, applicato solo in ORIZZONTALE (non il margine simmetrico del 25%
usato di default da predict_angle/_espandi_bbox, mai validato su foto
reali) — poi letterbox a quadrato, esattamente come crop_dataset.py.

Il crop risultante va poi passato a valida_pole_tilt_foto_reali.py con
--bbox 0,0,LARGHEZZA,ALTEZZA --margine_bbox_frazione 0 (nessuna ulteriore
manipolazione, il crop e' gia' pronto).

Lancio (dalla radice di ProgettoNurv):
    python costruisci_crop_stile_training.py --immagine foto_palo.jpg \
        --bbox 0,305,225,355 --dimensione_output 224
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--immagine", required=True)
    ap.add_argument("--bbox", required=True, help="x,y,w,h del bbox STRETTO attorno al palo")
    ap.add_argument("--dimensione_output", type=int, default=224)
    ap.add_argument("--margine_orizzontale_frazione_altezza", type=float, default=0.55,
                     help="Margine orizzontale come frazione dell'ALTEZZA del bbox (non della "
                          "sua larghezza) — replica la convenzione di crop_dataset.py")
    ap.add_argument("--out", default=None,
                     help="Percorso di output (default: <immagine>_crop_training.png)")
    args = ap.parse_args()

    x, y, w, h = (int(v) for v in args.bbox.split(","))

    img = cv2.imread(args.immagine)
    if img is None:
        print(f"[ERRORE] Impossibile aprire {args.immagine}")
        return
    img_h, img_w = img.shape[:2]

    # margine ORIZZONTALE basato sull'ALTEZZA del bbox (non sulla sua larghezza),
    # nessun margine verticale extra — replica la logica dichiarata per
    # crop_dataset.py, diversa da _espandi_bbox() di PoleTiltAnalyzer
    margine_x = int(h * args.margine_orizzontale_frazione_altezza)

    x_min = max(0, x - margine_x)
    x_max = min(img_w, x + w + margine_x)
    y_min = max(0, y)
    y_max = min(img_h, y + h)

    print(f"[INFO] Bbox originale: x={x},y={y},w={w},h={h}")
    print(f"[INFO] Margine orizzontale applicato: {margine_x}px per lato "
          f"({args.margine_orizzontale_frazione_altezza}x l'altezza {h})")
    print(f"[INFO] Area ritagliata: [{x_min}:{x_max}, {y_min}:{y_max}] "
          f"-> {x_max-x_min}x{y_max-y_min}px")

    crop = img[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        print("[ERRORE] Crop vuoto — controlla le coordinate del bbox.")
        return

    # letterbox a quadrato, stessa tecnica di PoleTiltAnalyzer._letterbox
    ch, cw = crop.shape[:2]
    dim = args.dimensione_output
    scala = dim / max(cw, ch)
    nuova_w, nuova_h = max(1, int(round(cw * scala))), max(1, int(round(ch * scala)))
    ridim = cv2.resize(crop, (nuova_w, nuova_h), interpolation=cv2.INTER_LANCZOS4)

    risultato = np.zeros((dim, dim, 3), dtype=np.uint8)
    offset_x = (dim - nuova_w) // 2
    offset_y = (dim - nuova_h) // 2
    risultato[offset_y:offset_y + nuova_h, offset_x:offset_x + nuova_w] = ridim

    out_path = args.out or str(Path(args.immagine).with_stem(Path(args.immagine).stem + "_crop_training"))
    cv2.imwrite(out_path, risultato)
    print(f"\n[FATTO] Crop in stile training salvato in: {out_path} ({dim}x{dim}px)")
    print(f"\nOra testalo cosi':")
    print(f"  python valida_pole_tilt_foto_reali.py --immagine {out_path} "
          f"--bbox 0,0,{dim},{dim} --angolo_vero TUO_ANGOLO --margine_bbox_frazione 0")


if __name__ == "__main__":
    main()
