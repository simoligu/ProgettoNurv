import cv2
import numpy as np
from pathlib import Path

# percorso di una maschera di esempio
mask_path = Path("data/rs19_val/uint8/rs19_val/rs00000.png")

mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if mask is None:
    print(f"ERRORE: maschera non trovata o non leggibile: {mask_path}")
else:
    print(f"Maschera caricata: {mask_path.name}")
    print(f"Dimensioni (HxW): {mask.shape}")
    print(f"Tipo di dato: {mask.dtype}")
    valori = np.unique(mask)
    print(f"Valori di classe presenti in questa maschera: {valori}")

    # mappa i valori che ci interessano
    nomi = {5: "pole (pali)", 8: "vegetation (vegetazione)", 17: "rail-raised (rotaie)"}
    print("\nClassi che ci interessano presenti in questa immagine:")
    for v in valori:
        if v in nomi:
            n_pixel = int(np.sum(mask == v))
            print(f"  valore {v} = {nomi[v]} -> {n_pixel} pixel")