# -*- coding: utf-8 -*-
"""
DIAGNOSTICA skeleton ROTAIE: visualizza, su poche immagini, perche' lo skeleton
predetto e quello vero non coincidono. Salva immagini di confronto da guardare a occhio.

Su ogni immagine sovrappone 4 livelli, con colori distinti:
  - VERDE (semitrasparente): rotaia VERA (ground truth sottile, pixel 17/18)
  - GIALLO (linea piena):     skeleton della rotaia VERA
  - ROSSO (semitrasparente):  rotaia PREDETTA dal modello (banda, eventualmente ingrossata)
  - CIANO (linea piena):      skeleton della rotaia PREDETTA

Cosi si vede a colpo d'occhio:
  * se GIALLO e CIANO sono due linee PARALLELE vicine  -> disallineamento da ingrossamento
  * se il CIANO ha tante BARBE/biforcazioni             -> skeleton rumoroso da banda spessa
  * se sono TRASLATI di una distanza costante           -> problema di scala/allineamento
  * se la banda ROSSA fonde due rotaie in una           -> ingrossamento eccessivo

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python scripts\\diag_skeleton_rotaie.py
    python scripts\\diag_skeleton_rotaie.py --n 6 --weights runs\\segment\\runs_nurv\\train_s_v2\\weights\\best.pt
Output: cartella  data/rs19_val/diag_skeleton/  con le immagini di confronto.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

try:
    from skimage.morphology import skeletonize
except ImportError:
    print("[ERRORE] Manca scikit-image. Installa con:  pip install scikit-image")
    raise SystemExit(1)

BASE = Path("data/rs19_val")
MASKS_DIR = BASE / "uint8" / "rs19_val"
IMAGES_DIR = BASE / "jpgs" / "rs19_val"
VAL_IMAGES_DIR = BASE / "images" / "val"
OUT_DIR = BASE / "diag_skeleton"

ROTAIE_PIXEL = (17, 18)
CLASSE_ROTAIE_YOLO = 0
CONF = 0.25
IMGSZ = 640

# colori BGR
VERDE = (0, 200, 0)
GIALLO = (0, 255, 255)
ROSSO = (0, 0, 255)
CIANO = (255, 255, 0)


def pred_rotaie_binaria(result, h, w):
    out = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None:
        return out
    masks = result.masks.data.cpu().numpy()
    classi = result.boxes.cls.cpu().numpy().astype(int)
    for i in range(len(classi)):
        if classi[i] != CLASSE_ROTAIE_YOLO:
            continue
        m = masks[i]
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out[m > 0.5] = 255
    return out


def overlay_mask(img, mask_bool, color, alpha=0.4):
    """Sovrappone una maschera semitrasparente colorata sull'immagine (in place)."""
    color_layer = np.zeros_like(img)
    color_layer[mask_bool] = color
    cv2.addWeighted(color_layer, alpha, img, 1.0, 0, dst=img)


def draw_skeleton(img, skel_bool, color, thickness=2):
    """Disegna lo skeleton ispessito per renderlo visibile (dilata la linea 1px)."""
    sk = (skel_bool.astype(np.uint8)) * 255
    if thickness > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        sk = cv2.dilate(sk, k, iterations=1)
    img[sk > 0] = color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/segment/runs_nurv/train_s_v2/weights/best.pt")
    ap.add_argument("--n", type=int, default=5, help="quante immagini diagnosticare")
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"[ERRORE] Pesi non trovati: {args.weights}  (usa --weights)")
        return

    val_imgs = sorted(VAL_IMAGES_DIR.glob("*.jpg"))
    if not val_imgs:
        print(f"[ERRORE] Nessuna immagine in {VAL_IMAGES_DIR}")
        return

    # scelgo immagini "ricche di rotaie" per una diagnosi utile:
    # provo le prime ~40 e tengo quelle con piu pixel-rotaia nel GT, fino a n.
    candidate = val_imgs[:40]
    punteggio = []
    for p in candidate:
        mp = MASKS_DIR / f"{p.stem}.png"
        if not mp.exists():
            continue
        md = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if md is None:
            continue
        if md.ndim == 3:
            md = md[:, :, 0]
        n_rotaie = int(np.isin(md, ROTAIE_PIXEL).sum())
        punteggio.append((n_rotaie, p))
    punteggio.sort(reverse=True)
    scelte = [p for _, p in punteggio[:args.n]]
    if not scelte:
        scelte = val_imgs[:args.n]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    print(f"[INFO] Diagnostica su {len(scelte)} immagini. Output in: {OUT_DIR}\n")

    for p in scelte:
        stem = p.stem
        # immagine di base: preferisco l'originale a piena risoluzione
        img_path = IMAGES_DIR / f"{stem}.jpg"
        if not img_path.exists():
            img_path = p
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Immagine illeggibile: {img_path}")
            continue
        h, w = img.shape[:2]

        mp = MASKS_DIR / f"{stem}.png"
        md = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if md is None:
            continue
        if md.ndim == 3:
            md = md[:, :, 0]
        if md.shape[:2] != (h, w):
            md = cv2.resize(md, (w, h), interpolation=cv2.INTER_NEAREST)

        gt_bin = np.isin(md, ROTAIE_PIXEL)
        skel_gt = skeletonize(gt_bin)

        res = model.predict(str(img_path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        pred_bin = pred_rotaie_binaria(res, h, w) > 0
        skel_pred = skeletonize(pred_bin)

        vis = img.copy()
        # ordine: prima le bande semitrasparenti, poi gli skeleton sopra (ben visibili)
        overlay_mask(vis, gt_bin, VERDE, alpha=0.35)        # rotaia vera
        overlay_mask(vis, pred_bin, ROSSO, alpha=0.30)      # rotaia predetta
        draw_skeleton(vis, skel_gt, GIALLO, thickness=3)    # skeleton vero
        draw_skeleton(vis, skel_pred, CIANO, thickness=3)   # skeleton predetto

        # legenda in alto a sinistra
        y0 = 26
        for testo, col in [("rotaia VERA", VERDE), ("skeleton VERO", GIALLO),
                           ("rotaia PRED", ROSSO), ("skeleton PRED", CIANO)]:
            cv2.putText(vis, testo, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, testo, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
            y0 += 28

        out_path = OUT_DIR / f"{stem}_diag.jpg"
        cv2.imwrite(str(out_path), vis)
        n_gt = int(skel_gt.sum()); n_pr = int(skel_pred.sum())
        print(f"  salvata {out_path.name}  (skeleton: vero={n_gt}px, pred={n_pr}px)")

    print(f"\n[FATTO] Apri le immagini in {OUT_DIR} e osserva il rapporto GIALLO vs CIANO.")


if __name__ == "__main__":
    main()
