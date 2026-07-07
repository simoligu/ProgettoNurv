# -*- coding: utf-8 -*-
"""
Split train/val per il flusso SEMANTICO (DeepLab).

A differenza dello split originale (che partiva dalle label YOLO .txt, qui inesistenti
perche' per DeepLab non serve la conversione a poligoni), questo parte direttamente
dalle IMMAGINI e le divide 80/20. Crea solo le cartelle images/train e images/val,
che sono tutto cio' che serve: prepara_maschere_semantiche.py usa proprio quelle due
cartelle per sapere quali immagini sono train e quali val.

Split riproducibile (stesso SEED dell'originale, cosi' la partizione e' identica a quella
usata sul fisso -> risultati confrontabili).

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python split_dataset_semantico.py
"""

import shutil
import random
from pathlib import Path

BASE = Path("data/rs19_val")
IMAGES_SRC = BASE / "jpgs" / "rs19_val"    # immagini originali (doppio annidamento RailSem19)

IMAGES_TRAIN = BASE / "images" / "train"
IMAGES_VAL   = BASE / "images" / "val"

VAL_RATIO = 0.20
SEED = 42   # stesso seed dell'originale -> stessa partizione train/val


def main():
    for d in [IMAGES_TRAIN, IMAGES_VAL]:
        d.mkdir(parents=True, exist_ok=True)

    # parte dalle IMMAGINI, non dalle label
    image_files = sorted(IMAGES_SRC.glob("*.jpg"))
    print(f"[INFO] Trovate {len(image_files)} immagini in {IMAGES_SRC}")
    if not image_files:
        print("[ERRORE] Nessuna immagine trovata. Verifica il percorso:")
        print(f"        {IMAGES_SRC}")
        print("        (le immagini RailSem19 stanno in jpgs\\rs19_val\\ per il doppio annidamento)")
        return

    random.seed(SEED)
    random.shuffle(image_files)

    n_val = int(len(image_files) * VAL_RATIO)
    val_set = image_files[:n_val]
    train_set = image_files[n_val:]

    print(f"[INFO] Train: {len(train_set)} | Val: {len(val_set)}")

    def copia(img_list, img_dst):
        copiate = 0
        for img_path in img_list:
            shutil.copy(img_path, img_dst / img_path.name)
            copiate += 1
        return copiate

    print("[INFO] Copia training set...")
    c_tr = copia(train_set, IMAGES_TRAIN)
    print("[INFO] Copia validation set...")
    c_val = copia(val_set, IMAGES_VAL)

    print(f"\n[INFO] Completato.")
    print(f"  Train: {c_tr} immagini in {IMAGES_TRAIN}")
    print(f"  Val:   {c_val} immagini in {IMAGES_VAL}")
    print("\nProssimo passo: python prepara_maschere_semantiche.py")


if __name__ == "__main__":
    main()
