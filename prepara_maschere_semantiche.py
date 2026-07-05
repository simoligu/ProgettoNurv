# -*- coding: utf-8 -*-
"""
TAPPA 1 — Preparazione dati per segmentazione SEMANTICA (DeepLabV3+ / SMP).

Converte le maschere dense di RailSem19 in maschere semantiche a 4 classi:
    0 = sfondo (tutto il resto)
    1 = rotaie      (valori RailSem19: 17 rail-raised, 18 rail-embedded)
    2 = pali        (valore  RailSem19: 5  pole)
    3 = vegetazione (valore  RailSem19: 8  vegetation)

A differenza della conversione YOLO (poligoni + morfologia + ingrossamento), qui si fa
una PURA RIMAPPATURA di valori-pixel: la maschera semantica risultante e' fedele al
ground truth originale (le rotaie restano sottili -> nessun artefatto di ingrossamento).

Riusa lo split train/val gia' esistente (cartelle images/train e images/val) per restare
coerente e confrontabile coi risultati YOLO: le maschere vengono organizzate nelle
stesse due partizioni.

Output:
    data/rs19_val/masks_sem/train/<stem>.png
    data/rs19_val/masks_sem/val/<stem>.png
Ogni PNG e' a canale singolo, valori {0,1,2,3}. (Sono "scure" a occhio: 0-3 su 255;
e' normale, servono al modello, non alla visione umana. Per ispezionarle usa lo script
di verifica che generiamo dopo, o moltiplica per ~60 per vederle.)

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python scripts\\prepara_maschere_semantiche.py
"""

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# ------------------ CONFIG ------------------
BASE = Path("data/rs19_val")
MASKS_DIR = BASE / "uint8" / "rs19_val"          # maschere dense originali
IMG_TRAIN = BASE / "images" / "train"            # per sapere quali stem sono train
IMG_VAL   = BASE / "images" / "val"              # e quali val
OUT_BASE  = BASE / "masks_sem"                    # output maschere semantiche

# rimappatura: valore-pixel RailSem19 -> classe semantica
# (tutti i valori non elencati restano 0 = sfondo)
REMAP = {
    17: 1,  # rail-raised   -> rotaie
    18: 1,  # rail-embedded -> rotaie
    5:  2,  # pole          -> pali
    8:  3,  # vegetation    -> vegetazione
}
NOMI = {0: "sfondo", 1: "rotaie", 2: "pali", 3: "vegetazione"}


def rimappa(mask_densa: np.ndarray) -> np.ndarray:
    """Da maschera densa RailSem19 a maschera semantica {0,1,2,3}."""
    out = np.zeros(mask_densa.shape, dtype=np.uint8)   # default: sfondo
    for valore, classe in REMAP.items():
        out[mask_densa == valore] = classe
    return out


def stem_set(cartella: Path):
    """Restituisce l'insieme degli 'stem' (nomi senza estensione) delle immagini."""
    if not cartella.exists():
        return set()
    return {p.stem for p in cartella.glob("*.jpg")}


def processa(split_nome: str, stems: set, statistiche: dict):
    out_dir = OUT_BASE / split_nome
    out_dir.mkdir(parents=True, exist_ok=True)
    fatte, mancanti = 0, 0
    for stem in tqdm(sorted(stems), desc=f"[{split_nome}]"):
        mask_path = MASKS_DIR / f"{stem}.png"
        if not mask_path.exists():
            mancanti += 1
            continue
        md = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if md is None:
            mancanti += 1
            continue
        if md.ndim == 3:
            md = md[:, :, 0]
        sem = rimappa(md)

        # accumula statistiche di copertura per classe (per report finale)
        for c in range(4):
            statistiche[c] += int((sem == c).sum())

        cv2.imwrite(str(out_dir / f"{stem}.png"), sem)
        fatte += 1
    return fatte, mancanti


def main():
    train_stems = stem_set(IMG_TRAIN)
    val_stems = stem_set(IMG_VAL)

    if not train_stems and not val_stems:
        print("[ERRORE] Non trovo immagini in images/train o images/val.")
        print("        Verifica che lo split YOLO sia stato fatto (serve per riusare la")
        print("        stessa partizione train/val).")
        return

    print(f"[INFO] Immagini train: {len(train_stems)} | val: {len(val_stems)}")
    print(f"[INFO] Rimappatura classi: {REMAP}  (resto -> 0 sfondo)\n")

    statistiche = {0: 0, 1: 0, 2: 0, 3: 0}
    ft, mt = processa("train", train_stems, statistiche)
    fv, mv = processa("val", val_stems, statistiche)

    tot_pixel = sum(statistiche.values())
    print("\n" + "=" * 56)
    print(" FATTO — maschere semantiche generate")
    print("=" * 56)
    print(f"  train: {ft} maschere   (mancanti: {mt})")
    print(f"  val:   {fv} maschere   (mancanti: {mv})")
    print(f"  output in: {OUT_BASE}")
    print("-" * 56)
    print("  Distribuzione pixel per classe (utile per pesare la loss):")
    for c in range(4):
        perc = 100.0 * statistiche[c] / tot_pixel if tot_pixel else 0.0
        print(f"    {c} {NOMI[c]:<12} {perc:6.2f}%")
    print("=" * 56)
    print("\n[nota] Lo sbilanciamento e' atteso: lo sfondo domina, i pali sono pochi.")
    print("      Ne terremo conto nella loss del training (Tappa 2).")


if __name__ == "__main__":
    main()
