# -*- coding: utf-8 -*-
"""
Valutazione SEMANTICA con ROI — DeepLabV3+ (segmentation-models-pytorch)

Calcola IoU / Precision / Recall per classe sulla fascia inferiore dell'immagine
(la zona operativamente rilevante per NURV: scartamento, pali storti, vegetazione
invasiva avvengono tutti vicino alla sede ferroviaria, cioe in basso nell'inquadratura).

Con --roi 100 valuta l'immagine intera (default).
Con --roi 50  valuta solo la meta inferiore.
Con --roi 33  valuta solo il terzo inferiore.
Si possono passare piu valori: --roi 100 50 33  per confrontarli in un'unica esecuzione
(l'inferenza viene fatta una sola volta, poi le metriche vengono calcolate per ogni ROI).

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python valuta_semantica_roi.py --weights runs_seg/deeplab_hires/best.pt --imgsz 896
    python valuta_semantica_roi.py --weights runs_seg/deeplab_hires/best.pt --imgsz 896 --roi 100 50 33

Requisiti: segmentation-models-pytorch, albumentations, torch, opencv, numpy
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp

# ------------------ CONFIG ------------------
BASE = Path("data/rs19_val")
IMAGES_DIR = BASE / "jpgs" / "rs19_val"
MASKS_SEM_VAL = BASE / "masks_sem" / "val"

NUM_CLASSI = 4
NOMI = {0: "sfondo", 1: "rotaie", 2: "pali", 3: "vegetazione"}

# normalizzazione ImageNet (stessa usata in training)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def carica_modello(weights_path: str, device: torch.device) -> torch.nn.Module:
    """Carica il modello DeepLabV3+ dal checkpoint salvato da train_deeplab.py."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # il checkpoint puo contenere il dict con chiave "model" (formato di train_deeplab.py)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # assume che il file sia direttamente lo state_dict
        state_dict = ckpt

    backbone = "resnet34"

    model = smp.DeepLabV3Plus(
        encoder_name=backbone,
        encoder_weights=None,  # carichiamo i pesi dal checkpoint
        in_channels=3,
        classes=NUM_CLASSI,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess(img_bgr: np.ndarray, imgsz: int) -> torch.Tensor:
    """BGR -> RGB, resize, normalizza, tensor [1, 3, H, W]."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    # HWC -> CHW -> NCHW
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def predici(model, img_bgr: np.ndarray, imgsz: int, device: torch.device,
            h_orig: int, w_orig: int) -> np.ndarray:
    """Ritorna la mappa di classi predetta [h_orig, w_orig] con valori 0..NUM_CLASSI-1."""
    tensor = preprocess(img_bgr, imgsz).to(device)
    with torch.no_grad():
        logits = model(tensor)  # [1, C, imgsz, imgsz]
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    # riporta alla dimensione originale
    if pred.shape[0] != h_orig or pred.shape[1] != w_orig:
        pred = cv2.resize(pred, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    return pred


def calcola_metriche(gt: np.ndarray, pred: np.ndarray, roi_pct: int):
    """Calcola IoU/Precision/Recall sulla fascia inferiore (roi_pct % dell'altezza).
    Ritorna dict con i risultati per classe."""
    h = gt.shape[0]
    # taglio ROI: prendi solo la fascia inferiore
    h_start = h - int(h * roi_pct / 100)
    gt_roi = gt[h_start:, :]
    pred_roi = pred[h_start:, :]

    risultati = {}
    for c in range(NUM_CLASSI):
        gt_c = (gt_roi == c)
        pred_c = (pred_roi == c)
        inter = np.logical_and(gt_c, pred_c).sum()
        union = np.logical_or(gt_c, pred_c).sum()
        risultati[c] = {
            "inter": int(inter),
            "union": int(union),
            "pred_tot": int(pred_c.sum()),
            "gt_tot": int(gt_c.sum()),
        }
    return risultati


def stampa_risultati(accumulatori: dict, roi_pct: int):
    """Stampa la tabella dei risultati per una data ROI."""
    print(f"\n{'=' * 68}")
    if roi_pct == 100:
        print(f" RISULTATI — IMMAGINE INTERA (100%)")
    else:
        print(f" RISULTATI — ROI: fascia inferiore {roi_pct}% dell'immagine")
    print(f"{'=' * 68}")
    print(f"{'Classe':<14}{'IoU':>9}{'Precision':>12}{'Recall':>10}")
    print(f"{'-' * 68}")

    iou_list = []
    for c in range(NUM_CLASSI):
        acc = accumulatori[c]
        iou = acc["inter"] / acc["union"] if acc["union"] > 0 else float("nan")
        prec = acc["inter"] / acc["pred_tot"] if acc["pred_tot"] > 0 else float("nan")
        rec = acc["inter"] / acc["gt_tot"] if acc["gt_tot"] > 0 else float("nan")
        iou_list.append(iou)
        print(f"{NOMI[c]:<14}{iou:>9.4f}{prec:>12.4f}{rec:>10.4f}")

    print(f"{'-' * 68}")
    miou = np.nanmean(iou_list)
    # mIoU senza sfondo (solo le 3 classi di interesse)
    miou_no_bg = np.nanmean(iou_list[1:])
    print(f"{'mIoU (4 cl.)':<14}{miou:>9.4f}")
    print(f"{'mIoU (no bg)':<14}{miou_no_bg:>9.4f}")
    print(f"{'=' * 68}")


def main():
    ap = argparse.ArgumentParser(description="Valutazione semantica DeepLab con ROI")
    ap.add_argument("--weights", required=True, help="Percorso al best.pt DeepLab")
    ap.add_argument("--imgsz", type=int, default=896, help="Risoluzione di inferenza")
    ap.add_argument("--roi", type=int, nargs="+", default=[100],
                    help="Percentuale(i) inferiore dell'immagine da valutare. "
                         "Es: --roi 100 50 33  (default: 100 = intera)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Valuta solo le prime N immagini (test rapido)")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"[ERRORE] Pesi non trovati: {args.weights}")
        return

    # determina il device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # carica modello
    print(f"[INFO] Carico modello da {args.weights}")
    model = carica_modello(args.weights, device)

    # elenco maschere di validation (fanno da indice: per ogni maschera, trovo l'immagine)
    val_masks = sorted(MASKS_SEM_VAL.glob("*.png"))
    if not val_masks:
        print(f"[ERRORE] Nessuna maschera in {MASKS_SEM_VAL}")
        print("        Verifica che prepara_maschere_semantiche.py e lo split siano stati eseguiti.")
        return

    if args.limit:
        val_masks = val_masks[:args.limit]

    print(f"[INFO] Valuto su {len(val_masks)} immagini | imgsz={args.imgsz} | ROI={args.roi}%\n")

    # accumulatori: uno per ogni valore di ROI
    accumulatori = {}
    for roi in args.roi:
        accumulatori[roi] = {c: {"inter": 0, "union": 0, "pred_tot": 0, "gt_tot": 0}
                             for c in range(NUM_CLASSI)}

    for idx, mask_path in enumerate(val_masks, 1):
        stem = mask_path.stem
        img_path = IMAGES_DIR / f"{stem}.jpg"
        if not img_path.exists():
            # prova anche .png
            img_path = IMAGES_DIR / f"{stem}.png"
            if not img_path.exists():
                continue

        img_bgr = cv2.imread(str(img_path))
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or gt is None:
            continue

        h, w = gt.shape[:2]

        # inferenza (una sola volta per immagine)
        pred = predici(model, img_bgr, args.imgsz, device, h, w)

        # calcola metriche per ogni ROI richiesta
        for roi in args.roi:
            ris = calcola_metriche(gt, pred, roi)
            for c in range(NUM_CLASSI):
                for k in ("inter", "union", "pred_tot", "gt_tot"):
                    accumulatori[roi][c][k] += ris[c][k]

        if idx % 100 == 0:
            print(f"  ...{idx}/{len(val_masks)} immagini")

    # stampa risultati per ogni ROI
    for roi in args.roi:
        stampa_risultati(accumulatori[roi], roi)

    print(f"\nInterpretazione:")
    print(f"  'IoU'       = sovrapposizione pixel predetti vs veri (metrica principale)")
    print(f"  'Precision' = dei pixel predetti come classe X, quanti sono corretti")
    print(f"  'Recall'    = dei pixel veri di classe X, quanti il modello li trova")
    if any(r < 100 for r in args.roi):
        print(f"\n  Le metriche ROI misurano la qualita SOLO nella fascia inferiore,")
        print(f"  dove avvengono le anomalie rilevanti per NURV (scartamento, invasione")
        print(f"  vegetazione, pali storti). Gli errori ai bordi/orizzonte non pesano.")


if __name__ == "__main__":
    main()