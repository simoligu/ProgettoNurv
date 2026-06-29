# -*- coding: utf-8 -*-
"""
Valutazione SEMANTICA (IoU per pixel, per classe) del modello YOLOv8-seg di NURV.

A differenza della mAP "instance-level" prodotta da `yolo val`, questo script misura
la COPERTURA SEMANTICA: dei pixel che appartengono davvero a una classe (rotaie/pali/
vegetazione), quanti il modello li marca correttamente? E' la metrica appropriata per
il rilevamento anomalie nel NURV, dove conta DOVE sono gli oggetti (palo storto verso
i binari, vegetazione che invade la sede), non QUANTE istanze distinte ci sono.

Per ogni immagine di validation:
  1) il modello predice le maschere delle 3 classi;
  2) si carica la maschera densa "vera" di RailSem19 (cartella uint8/);
  3) si rimappa il ground truth con la stessa CLASS_MAP della conversione;
  4) si accumulano intersezione e unione per classe su tutte le immagini;
  5) si calcola l'IoU per classe e il mIoU (media), piu Precision/Recall a livello di pixel.

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python scripts\\valuta_semantica.py
Eventuale percorso pesi diverso:
    python scripts\\valuta_semantica.py --weights runs\\segment\\runs_nurv\\train_s_v2\\weights\\best.pt
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

# ------------------ CONFIG (coerente con convert_railsem_to_yoloseg.py) ------------------
BASE = Path("data/rs19_val")
MASKS_DIR = BASE / "uint8" / "rs19_val"        # maschere dense "vere"
IMAGES_DIR = BASE / "jpgs" / "rs19_val"        # immagini originali
VAL_IMAGES_DIR = BASE / "images" / "val"       # split di validation (per sapere QUALI immagini valutare)

# stessa mappatura della conversione: pixel RailSem19 -> classe YOLO
CLASS_MAP = {17: 0, 18: 0, 5: 1, 8: 2}
NUM_CLASSI = 3
NOMI = {0: "rotaie", 1: "pali", 2: "vegetazione"}

# soglia di confidenza per le predizioni (allineabile a quella d'uso nella pipeline)
CONF = 0.25
IMGSZ = 640  # stessa risoluzione del training; alzala se valuti un modello allenato a imgsz maggiore


def gt_semantica(mask_densa: np.ndarray) -> np.ndarray:
    """Da maschera densa RailSem19 a mappa di classi {0,1,2} + 255=background/ignora."""
    out = np.full(mask_densa.shape, 255, dtype=np.uint8)
    for pixel_value, class_idx in CLASS_MAP.items():
        out[mask_densa == pixel_value] = class_idx
    return out


def pred_semantica(result, h: int, w: int) -> np.ndarray:
    """Fonde tutte le istanze predette in un'unica mappa semantica {0,1,2} + 255=sfondo.
    Se due classi si contendono un pixel, vince quella dell'istanza con confidenza piu alta
    (le istanze arrivano gia ordinate per confidenza decrescente da Ultralytics)."""
    out = np.full((h, w), 255, dtype=np.uint8)
    if result.masks is None:
        return out
    # masks.data: tensore [N, h_mask, w_mask]; classi in boxes.cls
    masks = result.masks.data.cpu().numpy()          # [N, Hm, Wm], valori 0..1
    classi = result.boxes.cls.cpu().numpy().astype(int)  # [N]
    confidenze = result.boxes.conf.cpu().numpy()         # [N]
    # ordina per confidenza CRESCENTE: cosi le piu alte, scritte per ultime, sovrascrivono
    ordine = np.argsort(confidenze)
    for i in ordine:
        m = masks[i]
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out[m > 0.5] = classi[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/segment/runs_nurv/train_s_v2/weights/best.pt")
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"[ERRORE] Pesi non trovati: {args.weights}")
        print("        Indica il percorso giusto con --weights")
        return

    # elenco immagini di validation (solo quelle, non tutte le 8500)
    val_imgs = sorted(VAL_IMAGES_DIR.glob("*.jpg"))
    if not val_imgs:
        # fallback: se la cartella images/val non esiste, prova a leggerla altrove
        print(f"[ERRORE] Nessuna immagine in {VAL_IMAGES_DIR}")
        print("        Verifica che lo split sia stato fatto e le cartelle rinominate.")
        return
    print(f"[INFO] Valuto su {len(val_imgs)} immagini di validation.")
    print(f"[INFO] Modello: {args.weights} | conf={args.conf} | imgsz={args.imgsz}\n")

    model = YOLO(args.weights)

    # accumulatori per classe
    inter = np.zeros(NUM_CLASSI, dtype=np.int64)   # intersezione (pixel veri E predetti)
    union = np.zeros(NUM_CLASSI, dtype=np.int64)   # unione (pixel veri O predetti)
    pred_tot = np.zeros(NUM_CLASSI, dtype=np.int64)  # pixel predetti (per precision)
    gt_tot = np.zeros(NUM_CLASSI, dtype=np.int64)    # pixel veri (per recall)

    saltate = 0
    for idx, img_path in enumerate(val_imgs, 1):
        stem = img_path.stem
        mask_path = MASKS_DIR / f"{stem}.png"
        if not mask_path.exists():
            saltate += 1
            continue

        mask_densa = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_densa is None:
            saltate += 1
            continue
        h, w = mask_densa.shape

        gt = gt_semantica(mask_densa)

        # predizione (verbose=False per non intasare l'output)
        res = model.predict(str(img_path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        pred = pred_semantica(res, h, w)

        for c in range(NUM_CLASSI):
            gt_c = (gt == c)
            pred_c = (pred == c)
            i_c = np.logical_and(gt_c, pred_c).sum()
            u_c = np.logical_or(gt_c, pred_c).sum()
            inter[c] += i_c
            union[c] += u_c
            pred_tot[c] += pred_c.sum()
            gt_tot[c] += gt_c.sum()

        if idx % 100 == 0:
            print(f"  ...{idx}/{len(val_imgs)} immagini")

    print("\n" + "=" * 64)
    print(" RISULTATI — VALUTAZIONE SEMANTICA (IoU per pixel, per classe)")
    print("=" * 64)
    print(f"{'Classe':<14}{'IoU':>9}{'Precision':>12}{'Recall':>10}")
    print("-" * 64)
    iou_per_classe = []
    for c in range(NUM_CLASSI):
        iou = inter[c] / union[c] if union[c] > 0 else float('nan')
        prec = inter[c] / pred_tot[c] if pred_tot[c] > 0 else float('nan')
        rec = inter[c] / gt_tot[c] if gt_tot[c] > 0 else float('nan')
        iou_per_classe.append(iou)
        print(f"{NOMI[c]:<14}{iou:>9.3f}{prec:>12.3f}{rec:>10.3f}")
    print("-" * 64)
    miou = np.nanmean(iou_per_classe)
    print(f"{'mIoU (media)':<14}{miou:>9.3f}")
    print("=" * 64)
    if saltate:
        print(f"[nota] {saltate} immagini saltate (maschera vera non trovata).")
    print("\nInterpretazione: 'Recall' = dei pixel VERI di quella classe, quanti il")
    print("modello li copre. 'Precision' = dei pixel che il modello marca come quella")
    print("classe, quanti sono giusti. 'IoU' = sovrapposizione complessiva (la metrica")
    print("piu importante per la copertura semantica).")


if __name__ == "__main__":
    main()