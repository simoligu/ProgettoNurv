# scripts/visualizza_label.py
import cv2
import numpy as np
from pathlib import Path

BASE = Path("data/rs19_val")
IMAGES_DIR = BASE / "jpgs" / "rs19_val"
LABELS_DIR = BASE / "labels"

# quale immagine verificare (cambia il nome per vederne altre)
NOME = "rs00000"

# colori per classe (BGR): 0=rotaie rosso, 1=pali blu, 2=vegetazione verde
COLORI = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0)}
NOMI = {0: "rotaie", 1: "pali", 2: "vegetazione"}

img_path = IMAGES_DIR / f"{NOME}.jpg"
label_path = LABELS_DIR / f"{NOME}.txt"

img = cv2.imread(str(img_path))
if img is None:
    print(f"[ERRORE] Immagine non trovata: {img_path}")
    exit()

h, w = img.shape[:2]
overlay = img.copy()

righe = label_path.read_text().strip().split("\n")
for riga in righe:
    if not riga.strip():
        continue
    parti = riga.split()
    classe = int(parti[0])
    coords = list(map(float, parti[1:]))
    # ricostruisci i punti del poligono (denormalizza)
    punti = []
    for i in range(0, len(coords), 2):
        x = int(coords[i] * w)
        y = int(coords[i + 1] * h)
        punti.append([x, y])
    punti = np.array(punti, dtype=np.int32)
    colore = COLORI.get(classe, (255, 255, 255))
    cv2.fillPoly(overlay, [punti], colore)

# fondi immagine originale e overlay semi-trasparente
# risultato = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
risultato = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

# legenda
y0 = 30
for classe, nome in NOMI.items():
    cv2.putText(risultato, nome, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORI[classe], 2)
    y0 += 30

out_path = BASE / f"verifica_{NOME}.jpg"
cv2.imwrite(str(out_path), risultato)
print(f"[INFO] Immagine di verifica salvata in: {out_path}")
print("Aprila e controlla che i colori coprano gli oggetti giusti.")