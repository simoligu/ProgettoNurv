import cv2, numpy as np
from pathlib import Path

BASE = Path("data/rs19_val")
mask = cv2.imread(str(BASE/"uint8"/"rs19_val"/"rs00000.png"), cv2.IMREAD_GRAYSCALE)
h, w = mask.shape

# conta entrambe le classi binario
for val, nome in [(17, "rail-raised (rotaie metalliche)"), (12, "rail-track (superficie tra rotaie)")]:
    n = int(np.sum(mask == val))
    print(f"valore {val} = {nome}: {n} pixel")

# crea immagine: rail-raised in ROSSO, rail-track in GIALLO
canvas = np.zeros((h, w, 3), dtype=np.uint8)
canvas[mask == 17] = (0, 0, 255)     # rosso = rail-raised
canvas[mask == 12] = (0, 255, 255)   # giallo = rail-track

cv2.putText(canvas, "ROSSO = rail-raised (17)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
cv2.putText(canvas, "GIALLO = rail-track (12)", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

out = BASE / "confronto_binari_rs00000.jpg"
cv2.imwrite(str(out), canvas)
print(f"\nSalvata: {out}")