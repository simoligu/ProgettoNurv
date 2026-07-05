# -*- coding: utf-8 -*-
"""
Predizioni VISIVE del modello DeepLabV3+ (segmentazione semantica NURV).

Prende il best.pt addestrato e genera, su alcune immagini di validation, un confronto
affiancato: [immagine originale | ground truth | predizione del modello], con le classi
colorate (rotaie/pali/vegetazione). Serve per il controllo qualitativo e per le figure
della tesi.

Colori (come da convenzione usata finora nelle verifiche):
    rotaie      = rosso
    pali        = blu
    vegetazione = verde
    sfondo      = trasparente (si vede l'immagine)

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python scripts\\predici_deeplab.py
    python scripts\\predici_deeplab.py --weights runs_seg\\deeplab_v1\\best.pt --n 8 --imgsz 640
Output: cartella  runs_seg/deeplab_v1/predizioni/  con le immagini di confronto.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp

BASE = Path("data/rs19_val")
IMG_DIR = BASE / "jpgs" / "rs19_val"
MASK_SEM_DIR = BASE / "masks_sem" / "val"
VAL_IMAGES_DIR = BASE / "images" / "val"

NUM_CLASSI = 4
# colori BGR per classe: 0=sfondo(nessuno), 1=rotaie(rosso), 2=pali(blu), 3=vegetazione(verde)
COLORI = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 200, 0)}
NOMI = {0: "sfondo", 1: "rotaie", 2: "pali", 3: "vegetazione"}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def colora(mask_classi, base_img, alpha=0.5):
    """Sovrappone la maschera di classi colorata sull'immagine base."""
    out = base_img.copy()
    layer = np.zeros_like(base_img)
    for c, col in COLORI.items():
        layer[mask_classi == c] = col
    mask_any = (mask_classi > 0)
    out[mask_any] = cv2.addWeighted(base_img, 1 - alpha, layer, alpha, 0)[mask_any]
    return out


def etichetta(img, testo):
    cv2.putText(img, testo, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, testo, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs_seg/deeplab_v1/best.pt")
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"[ERRORE] Pesi non trovati: {args.weights}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.DeepLabV3Plus(encoder_name=args.backbone, encoder_weights=None,
                              in_channels=3, classes=NUM_CLASSI).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ep = ckpt.get("epoca", "?")
    print(f"[INFO] Modello caricato: {args.weights} (epoca {ep}) | device {device}")

    out_dir = Path(args.weights).parent / "predizioni"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_imgs = sorted(VAL_IMAGES_DIR.glob("*.jpg"))
    # scelgo immagini ricche di pali/rotaie per una visualizzazione utile
    candidate = val_imgs[:60]
    punteggio = []
    for p in candidate:
        mp = MASK_SEM_DIR / f"{p.stem}.png"
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        # privilegia immagini con tanti pali (classe 2) e rotaie (classe 1)
        score = int((m == 2).sum()) * 2 + int((m == 1).sum())
        punteggio.append((score, p))
    punteggio.sort(reverse=True)
    scelte = [p for _, p in punteggio[:args.n]] or val_imgs[:args.n]

    for p in scelte:
        stem = p.stem
        img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        H, W = img.shape[:2]

        # ground truth
        gt = cv2.imread(str(MASK_SEM_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if gt is not None and gt.shape[:2] != (H, W):
            gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)

        # predizione
        inp = cv2.resize(img, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = (inp - MEAN) / STD
        inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

        vis_orig = etichetta(img.copy(), "originale")
        vis_gt = etichetta(colora(gt, img) if gt is not None else img.copy(), "ground truth")
        vis_pred = etichetta(colora(pred, img), "predizione DeepLab")

        # affianca i tre pannelli in orizzontale
        combo = np.hstack([vis_orig, vis_gt, vis_pred])
        # ridimensiona per non avere file enormi (larghezza max ~2400)
        if combo.shape[1] > 2400:
            scale = 2400 / combo.shape[1]
            combo = cv2.resize(combo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        out_path = out_dir / f"{stem}_pred.jpg"
        cv2.imwrite(str(out_path), combo)
        print(f"  salvata {out_path.name}")

    print(f"\n[FATTO] {len(scelte)} confronti in: {out_dir}")
    print("Legenda colori: rotaie=rosso, pali=blu, vegetazione=verde.")


if __name__ == "__main__":
    main()
