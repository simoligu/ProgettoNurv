import cv2, numpy as np
from pathlib import Path

BASE = Path("data/rs19_val")
NOME = "rs00006"   # cambia per altre immagini

mask = cv2.imread(str(BASE/"uint8"/"rs19_val"/f"{NOME}.png"), cv2.IMREAD_GRAYSCALE)
h, w = mask.shape

# classi "verdi"/naturali da confrontare in RailSem19:
# 8 = vegetation (cespugli, alberi)   9 = terrain (erba, terreno basso)
classi = [
    (8,  "vegetation (alberi/cespugli)", (0, 255, 0)),     # verde acceso
    (9,  "terrain (erba/terreno)",       (0, 255, 255)),   # giallo
]

# stampa i conteggi
print(f"Immagine: {NOME}")
for val, nome, _ in classi:
    n = int(np.sum(mask == val))
    perc = n / (h * w) * 100
    print(f"  valore {val} = {nome}: {n} pixel ({perc:.1f}%)")

# crea immagine a colori
canvas = np.zeros((h, w, 3), dtype=np.uint8)
for val, _, colore in classi:
    canvas[mask == val] = colore

# legenda
y0 = 30
for val, nome, colore in classi:
    cv2.putText(canvas, nome, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colore, 2)
    y0 += 35

out = BASE / f"confronto_veg_{NOME}.jpg"
cv2.imwrite(str(out), canvas)
print(f"\nSalvata: {out}")
print("VERDE = vegetation (quello che catturiamo) | GIALLO = terrain (che NON catturiamo)")