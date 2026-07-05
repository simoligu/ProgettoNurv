# -*- coding: utf-8 -*-
"""
TAPPA 2 — Training DeepLabV3+ (segmentazione SEMANTICA) per NURV.

Addestra un DeepLabV3+ (via segmentation-models-pytorch) a segmentare 4 classi:
    0 = sfondo, 1 = rotaie, 2 = pali, 3 = vegetazione
usando le maschere semantiche generate in Tappa 1 (data/rs19_val/masks_sem/).

Perche' DeepLabV3+: il modulo ASPP analizza l'immagine a piu' scale contemporaneamente,
utile per oggetti della stessa classe a dimensioni diverse (pali grandi in primo piano,
piccoli all'orizzonte) -> mira a migliorare proprio la copertura dei pali.

Perche' semantico invece di YOLO-seg: il compito reale e' di COPERTURA a pixel (dove
sono rotaie/pali/vegetazione), non di conteggio istanze. Qui l'IoU e' la metrica NATIVA,
senza artefatti (niente istanze da fondere, niente skeleton, niente ingrossamento).

Gestione dello sbilanciamento (rotaie ~2.6%, pali ~2.7%, sfondo ~72%):
    loss = Dice + CrossEntropy PESATA per classe (pesi ~ inverso frequenza).
La Dice ottimizza direttamente l'overlap (ottima per classi rare); la CE pesata da'
segnale pixel-per-pixel dando piu' importanza alle classi rare.

------------------------------------------------------------------------------------
USO:
  Test rapido (per validare che tutto giri, ~min):
      python scripts\\train_deeplab.py --epochs 3 --imgsz 384 --limit 300
  Training vero sul fisso (una notte):
      python scripts\\train_deeplab.py --epochs 50 --imgsz 640 --batch 8
  Riprendere da un checkpoint:
      python scripts\\train_deeplab.py --resume runs_seg/deeplab_v1/last.pt --epochs 50
------------------------------------------------------------------------------------
Dipendenze:  pip install segmentation-models-pytorch albumentations
"""

import argparse, time, json
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# ------------------ PERCORSI ------------------
BASE = Path("data/rs19_val")
IMG_DIR = BASE / "jpgs" / "rs19_val"          # immagini originali
MASK_DIR = BASE / "masks_sem"                  # maschere semantiche (train/ e val/)
OUT_ROOT = Path("runs_seg")                    # dove salvare i risultati

NUM_CLASSI = 4
NOMI = ["sfondo", "rotaie", "pali", "vegetazione"]

# pesi per classe per la CrossEntropy (inverso frequenza, dai dati di Tappa 1:
# sfondo 71.7%, rotaie 2.58%, pali 2.74%, vegetazione 22.98%). Normalizzati a media 1.
# Calcolati come (1/freq) poi riscalati; lo sfondo pesa poco, rotaie/pali molto.
PESI_CLASSE = [0.35, 9.7, 9.1, 1.1]

# normalizzazione ImageNet (il backbone e' pre-addestrato su ImageNet)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------ DATASET ------------------
class RailSemDataset(Dataset):
    """Coppie (immagine, maschera_semantica). Ridimensiona a imgsz, normalizza,
    e opzionalmente applica un flip orizzontale casuale come data augmentation."""
    def __init__(self, split: str, imgsz: int, augment: bool = False, limit: int = 0):
        self.mask_dir = MASK_DIR / split
        stems = sorted(p.stem for p in self.mask_dir.glob("*.png"))
        if limit and limit > 0:
            stems = stems[:limit]
        self.stems = stems
        self.imgsz = imgsz
        self.augment = augment
        if not self.stems:
            raise RuntimeError(f"Nessuna maschera in {self.mask_dir}. Hai lanciato la Tappa 1?")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # ridimensiona: immagine con interpolazione lineare, maschera con NEAREST
        # (la maschera contiene ID di classe: NON interpolare, altrimenti crea valori spuri)
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        # augmentation leggera: flip orizzontale casuale
        if self.augment and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])

        # normalizza immagine e porta a tensore CHW
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask.astype(np.int64))
        return img, mask


# ------------------ METRICHE ------------------
@torch.no_grad()
def calcola_iou(pred, target, num_classi=NUM_CLASSI):
    """IoU per classe accumulabile: restituisce intersezione e unione per ogni classe."""
    inter = torch.zeros(num_classi)
    union = torch.zeros(num_classi)
    for c in range(num_classi):
        p = (pred == c)
        t = (target == c)
        inter[c] = (p & t).sum().item()
        union[c] = (p | t).sum().item()
    return inter, union


# ------------------ TRAINING ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone", default="resnet34",
                    help="encoder: resnet34 (leggero), resnet50, ecc.")
    ap.add_argument("--name", default="deeplab_v1")
    ap.add_argument("--limit", type=int, default=0, help="usa solo N immagini (per test rapidi)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", default="", help="checkpoint da cui riprendere")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}", end="")
    if device == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print("  [ATTENZIONE] niente GPU: il training sara' MOLTO lento.")

    out_dir = OUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataset e loader
    train_ds = RailSemDataset("train", args.imgsz, augment=True, limit=args.limit)
    val_ds = RailSemDataset("val", args.imgsz, augment=False, limit=args.limit // 4 if args.limit else 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=(device == "cuda"), drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=(device == "cuda"))
    print(f"[INFO] Train: {len(train_ds)} img | Val: {len(val_ds)} img | "
          f"imgsz={args.imgsz} batch={args.batch} backbone={args.backbone}")

    # modello: DeepLabV3+ con backbone pre-addestrato su ImageNet
    model = smp.DeepLabV3Plus(
        encoder_name=args.backbone,
        encoder_weights="imagenet",   # transfer learning: parte "sapendo vedere"
        in_channels=3,
        classes=NUM_CLASSI,
    ).to(device)

    # loss combinata: Dice (overlap) + CrossEntropy pesata (segnale per-pixel, classi rare)
    dice_loss = smp.losses.DiceLoss(mode="multiclass")
    pesi = torch.tensor(PESI_CLASSE, dtype=torch.float32, device=device)
    ce_loss = nn.CrossEntropyLoss(weight=pesi)

    def criterio(logits, target):
        return dice_loss(logits, target) + ce_loss(logits, target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))  # mixed precision: piu' veloce, meno VRAM

    epoca_start = 0
    best_miou = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        epoca_start = ckpt.get("epoca", 0)
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"[INFO] Ripreso da {args.resume} (epoca {epoca_start}, best_miou {best_miou:.3f})")

    log_righe = []
    for epoca in range(epoca_start, args.epochs):
        # ---- TRAIN ----
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        for img, mask in train_dl:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(img)
                loss = criterio(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
        scheduler.step()
        loss_media = loss_sum / max(1, len(train_dl))

        # ---- VALIDATION (IoU per classe) ----
        model.eval()
        inter_tot = torch.zeros(NUM_CLASSI)
        union_tot = torch.zeros(NUM_CLASSI)
        with torch.no_grad():
            for img, mask in val_dl:
                img = img.to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = model(img)
                pred = logits.argmax(1).cpu()
                i_, u_ = calcola_iou(pred, mask)
                inter_tot += i_
                union_tot += u_
        iou = (inter_tot / union_tot.clamp(min=1)).numpy()
        miou = float(np.mean(iou))
        dt = time.time() - t0

        print(f"Epoca {epoca+1:3d}/{args.epochs} | loss {loss_media:.4f} | "
              f"mIoU {miou:.3f} | " +
              " ".join(f"{NOMI[c][:4]}={iou[c]:.3f}" for c in range(NUM_CLASSI)) +
              f" | {dt:.0f}s")

        log_righe.append({"epoca": epoca + 1, "loss": loss_media, "miou": miou,
                          "iou": {NOMI[c]: float(iou[c]) for c in range(NUM_CLASSI)}})
        (out_dir / "log.json").write_text(json.dumps(log_righe, indent=2))

        # ---- salvataggio checkpoint (robusto: last SEMPRE, best se migliora) ----
        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoca": epoca + 1, "best_miou": best_miou, "args": vars(args)}
        torch.save(ckpt, out_dir / "last.pt")
        if miou > best_miou:
            best_miou = miou
            ckpt["best_miou"] = best_miou
            torch.save(ckpt, out_dir / "best.pt")
            print(f"         -> nuovo BEST (mIoU {miou:.3f}), salvato best.pt")

    print(f"\n[FATTO] Training completato. Miglior mIoU: {best_miou:.3f}")
    print(f"        Pesi in: {out_dir}/best.pt  e  {out_dir}/last.pt")
    print(f"        Log epoche: {out_dir}/log.json")


if __name__ == "__main__":
    main()
