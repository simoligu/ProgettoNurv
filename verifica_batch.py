import cv2
import numpy as np
from pathlib import Path

BASE = Path("data/rs19_val")
IMAGES_DIR = BASE / "jpgs" / "rs19_val"
LABELS_DIR = BASE / "labels"
OUT_DIR = BASE / "verifiche"   # cartella dove salviamo tutte le verifiche

COLORI = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0)}
NOMI = {0: "rotaie", 1: "pali", 2: "vegetazione"}

OUT_DIR.mkdir(exist_ok=True)

# prendi tutte le label esistenti
label_files = sorted(LABELS_DIR.glob("*.txt"))
print(f"[INFO] Trovate {len(label_files)} label. Genero le verifiche...")

for label_path in label_files:
    nome = label_path.stem
    img_path = IMAGES_DIR / f"{nome}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]

    # sfondo: immagine originale scurita, per vedere bene i poligoni
    overlay = img.copy()

    testo = label_path.read_text().strip()
    if testo:
        for riga in testo.split("\n"):
            parti = riga.split()
            if not parti:
                continue
            classe = int(parti[0])
            coords = list(map(float, parti[1:]))
            punti = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                punti.append([x, y])
            if len(punti) < 3:
                continue
            punti = np.array(punti, dtype=np.int32)
            cv2.fillPoly(overlay, [punti], COLORI.get(classe, (255, 255, 255)))

    risultato = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    # legenda
    y0 = 30
    for classe, nome_classe in NOMI.items():
        cv2.putText(risultato, nome_classe, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORI[classe], 2)
        y0 += 30

    cv2.imwrite(str(OUT_DIR / f"{nome}_verifica.jpg"), risultato)

print(f"[INFO] Fatto. Immagini di verifica in: {OUT_DIR}")
print("Aprile cartella e sfogliale tutte insieme.")