# scripts/visualizza_solo_maschere.py
import cv2
import numpy as np
from pathlib import Path

BASE = Path("data/rs19_val")
IMAGES_DIR = BASE / "jpgs" / "rs19_val"
LABELS_DIR = BASE / "labels"

NOME = "rs00002"

COLORI = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0)}
NOMI = {0: "rotaie", 1: "pali", 2: "vegetazione"}

img_path = IMAGES_DIR / f"{NOME}.jpg"
label_path = LABELS_DIR / f"{NOME}.txt"

img = cv2.imread(str(img_path))
h, w = img.shape[:2]

# sfondo NERO, non l'immagine
canvas = np.zeros((h, w, 3), dtype=np.uint8)

righe = label_path.read_text().strip().split("\n")
for riga in righe:
    if not riga.strip():
        continue
    parti = riga.split()
    classe = int(parti[0])
    coords = list(map(float, parti[1:]))
    punti = []
    for i in range(0, len(coords), 2):
        x = int(coords[i] * w)
        y = int(coords[i + 1] * h)
        punti.append([x, y])
    punti = np.array(punti, dtype=np.int32)
    colore = COLORI.get(classe, (255, 255, 255))
    cv2.fillPoly(canvas, [punti], colore)

# legenda
y0 = 30
for classe, nome in NOMI.items():
    cv2.putText(canvas, nome, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORI[classe], 2)
    y0 += 30

out_path = BASE / f"solo_maschere_{NOME}.jpg"
cv2.imwrite(str(out_path), canvas)
print(f"[INFO] Salvata: {out_path}")
print("Sfondo nero: vedi SOLO le aree tracciate. Le rotaie devono apparire come due linee rosse.")