# -*- coding: utf-8 -*-
"""
Figura di confronto visivo YOLO contro DeepLab sulle stesse immagini (cap. 5).

Per ogni immagine di validazione scelta produce quattro pannelli separati e un
provino affiancato:
    originale | ground truth | YOLOv8-seg small (640 px) | DeepLabV3+ (896 px)
con gli stessi colori per tutti: rotaie rosso, pali blu, vegetazione verde.

Le predizioni YOLO vengono fuse in una mappa semantica con la stessa funzione
usata da valuta_semantica.py; quelle DeepLab con DeepLabAnalyzer.segment, cioe'
lo stesso codice della pipeline.

ATTENZIONE, licenza RailSem19: prima di usare un'immagine in tesi controllare
che non contenga persone o targhe riconoscibili.

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python confronto_yolo_deeplab.py                  # sceglie da solo 12 immagini
    python confronto_yolo_deeplab.py --stem rs01234 rs05678
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from deeplab_analyzer import DeepLabAnalyzer
from predici_deeplab import colora, etichetta

BASE = Path("data/rs19_val")
IMG_DIR = BASE / "jpgs" / "rs19_val"
MASK_DIR = BASE / "uint8" / "rs19_val"
VAL_DIR = BASE / "images" / "val"
REMAP = {17: 1, 18: 1, 5: 2, 8: 3}          # RailSem19 -> {0,1,2,3} come DeepLab


def gt_4classi(densa):
    out = np.zeros(densa.shape, np.uint8)
    for v, c in REMAP.items():
        out[densa == v] = c
    return out


def scegli_automaticamente(n):
    """Immagini di validazione con rotaie e pali ben rappresentati nel ground truth.
    Le scene con una copertura di palo superiore al 5% sono escluse: sono in gran
    parte i casi di rumore di etichettatura discussi nel capitolo 4 (pareti di
    galleria, pilastri), non esempi rappresentativi."""
    candidati = []
    for p in sorted(VAL_DIR.glob("*.jpg"))[:400]:
        m = cv2.imread(str(MASK_DIR / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        g = gt_4classi(m)
        pali = int((g == 2).sum())
        if pali > 0.05 * g.size:
            continue
        candidati.append((pali * 2 + int((g == 1).sum()), p.stem))
    candidati.sort(reverse=True)
    return [s for _, s in candidati[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="runs/segment/runs_nurv/train_s_v2/weights/best.pt")
    ap.add_argument("--deeplab", default="runs_seg/deeplab_hires/best.pt")
    ap.add_argument("--stem", nargs="*", default=None, help="Nomi delle immagini (es. rs01234)")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="risultati_tesi/confronto_yolo_deeplab")
    ap.add_argument("--senza_yolo", action="store_true", help="Solo per prova, senza pesi YOLO")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    yolo = None
    if not args.senza_yolo:
        if not Path(args.yolo).exists():
            print(f"[ERRORE] Pesi YOLO non trovati: {args.yolo}")
            return
        from ultralytics import YOLO
        from valuta_semantica import pred_semantica
        yolo = YOLO(args.yolo)

    deeplab = DeepLabAnalyzer(weights_path=args.deeplab, imgsz=896)
    stems = args.stem or scegli_automaticamente(args.n)

    for stem in stems:
        img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        densa = cv2.imread(str(MASK_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if densa is not None and densa.ndim == 3:
            densa = densa[...,0]
        if img is None or densa is None:
            print(f"  [SALTATA] {stem}: immagine o maschera non trovata")
            continue
        h, w = img.shape[:2]

        pannelli = {
            "originale": etichetta(img.copy(), "originale"),
            "groundtruth": etichetta(colora(gt_4classi(densa), img), "ground truth"),
        }
        if yolo is not None:
            r = yolo.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
            p3 = pred_semantica(r, h, w)                 # {0,1,2} + 255 = sfondo
            p4 = np.where(p3 == 255, 0, p3 + 1).astype(np.uint8)
            pannelli["yolo"] = etichetta(colora(p4, img), "YOLOv8-seg small")
        pannelli["deeplab"] = etichetta(colora(deeplab.segment(img), img), "DeepLabV3+")

        for nome, im in pannelli.items():
            cv2.imwrite(str(out / f"{stem}_{nome}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        provino = np.hstack(list(pannelli.values()))
        provino = cv2.resize(provino, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{stem}_provino.jpg"), provino)
        print(f"  {stem}: {len(pannelli)} pannelli")

    print(f"\n[FATTO] Output in {out.resolve()}")
    print("Colori: rotaie rosso, pali blu, vegetazione verde.")


if __name__ == "__main__":
    main()
