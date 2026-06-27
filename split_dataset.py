# scripts/split_dataset.py
import shutil
import random
from pathlib import Path

BASE = Path("data/rs19_val")
IMAGES_SRC = BASE / "jpgs" / "rs19_val"   # immagini originali
LABELS_SRC = BASE / "labels"              # label generate

# cartelle di destinazione nel formato YOLO
IMAGES_TRAIN = BASE / "images" / "train"
IMAGES_VAL   = BASE / "images" / "val"
LABELS_TRAIN = BASE / "labels_split" / "train"
LABELS_VAL   = BASE / "labels_split" / "val"

VAL_RATIO = 0.20   # 20% validation
SEED = 42          # per riproducibilità

def main():
    # crea le cartelle
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)

    # prendi tutte le label generate
    label_files = sorted(LABELS_SRC.glob("*.txt"))
    print(f"[INFO] Trovate {len(label_files)} label totali.")

    # mescola in modo riproducibile
    random.seed(SEED)
    random.shuffle(label_files)

    # calcola il punto di split
    n_val = int(len(label_files) * VAL_RATIO)
    val_set = label_files[:n_val]
    train_set = label_files[n_val:]

    print(f"[INFO] Train: {len(train_set)} | Val: {len(val_set)}")

    def copia(label_list, img_dst, lbl_dst):
        copiate = 0
        mancanti = 0
        for label_path in label_list:
            nome = label_path.stem
            img_src = IMAGES_SRC / f"{nome}.jpg"
            if not img_src.exists():
                mancanti += 1
                continue
            shutil.copy(img_src, img_dst / f"{nome}.jpg")
            shutil.copy(label_path, lbl_dst / f"{nome}.txt")
            copiate += 1
        return copiate, mancanti

    print("[INFO] Copia training set...")
    c_tr, m_tr = copia(train_set, IMAGES_TRAIN, LABELS_TRAIN)
    print("[INFO] Copia validation set...")
    c_val, m_val = copia(val_set, IMAGES_VAL, LABELS_VAL)

    print(f"\n[INFO] Completato.")
    print(f"  Train: {c_tr} coppie copiate ({m_tr} immagini mancanti)")
    print(f"  Val:   {c_val} coppie copiate ({m_val} immagini mancanti)")

if __name__ == "__main__":
    main()