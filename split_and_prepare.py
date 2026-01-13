#!/usr/bin/env python3
import random
import json
from pathlib import Path
import shutil

# === CONFIGURAZIONE ===
BASE = Path("data/gen")                 # qui si trova il dataset convertito (imgs/ masks/ labels/)
YAML_OUT = Path("data/synrailobs.yaml")# file YAML per YOLO (sotto data/)
CLASSES = ["animals", "motos", "persons", "rocks", "vehicles"]
VAL_RATIO = 0.2
SEED = 42
random.seed(SEED)

# === CONTROLLO BASE ===
if not BASE.exists():
    raise SystemExit(f"[ERROR] Base path non trovato: {BASE} (estrai archive.zip dentro 'data' o copia la struttura in questo percorso)")

# === CREA CARTELLE DI OUTPUT (sotto BASE) ===
for split in ["train", "val"]:
    (BASE / "images" / split).mkdir(parents=True, exist_ok=True)
    (BASE / "labels" / split).mkdir(parents=True, exist_ok=True)

# === SPLIT PER CLASSE ===
summary = {}
for cls in CLASSES:
    img_dir = BASE / cls / "imgs"
    label_dir = BASE / cls / "labels"

    if not img_dir.exists():
        print(f"[WARN] Immagini mancanti per la classe '{cls}': {img_dir} (skip)")
        summary[cls] = {"total": 0, "train": 0, "val": 0}
        continue

    # raccolta immagini con estensioni comuni
    imgs = []
    for pat in ("*.jpg", "*.jpeg", "*.png"):
        imgs.extend(sorted(img_dir.glob(pat)))
    if len(imgs) == 0:
        print(f"[WARN] Nessuna immagine trovata in {img_dir} per classe {cls}")
        summary[cls] = {"total": 0, "train": 0, "val": 0}
        continue

    random.shuffle(imgs)
    n_val = int(len(imgs) * VAL_RATIO)
    val_set = set(imgs[:n_val])
    train_set = set(imgs[n_val:])

    cnt_train = 0
    cnt_val = 0

    for img_path in imgs:
        split = "val" if img_path in val_set else "train"
        dst_img = BASE / "images" / split / img_path.name
        shutil.copy2(img_path, dst_img)

        # label corrispondente
        label_path = label_dir / f"{img_path.stem}.txt"
        dst_label = BASE / "labels" / split / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy2(label_path, dst_label)
        else:
            # creazione file label vuoto (YOLO richiede .txt per ogni img)
            dst_label.write_text("")

        if split == "train":
            cnt_train += 1
        else:
            cnt_val += 1

    summary[cls] = {"total": len(imgs), "train": cnt_train, "val": cnt_val}
    print(f"[INFO] {cls}: totale={len(imgs)} train={cnt_train} val={cnt_val}")

# === RIEPILOGO ===
total_imgs = sum(s["total"] for s in summary.values())
print("\n=== Riepilogo generale ===")
for k, v in summary.items():
    print(f"{k}: {v}")
print(f"Totale immagini trovate: {total_imgs}")

# === CREA YAML compatibile YOLO (path relativo a questo file YAML) ===
yaml_content = {
    "path": str(BASE),           # es. "data/gen"
    "train": "images/train",
    "val": "images/val",
    "nc": len(CLASSES),
    "names": CLASSES
}

YAML_OUT.parent.mkdir(parents=True, exist_ok=True)
YAML_OUT.write_text(json.dumps(yaml_content, indent=2))
print(f"\n[INFO] File YAML creato in: {YAML_OUT}")
print(YAML_OUT.read_text())
