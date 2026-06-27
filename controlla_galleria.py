import cv2, numpy as np
from pathlib import Path

BASE = Path("data/rs19_val")
NOME = "rs00002"   # METTI il nome dell'immagine con la galleria-palo

mask = cv2.imread(str(BASE/"uint8"/"rs19_val"/f"{NOME}.png"), cv2.IMREAD_GRAYSCALE)
h, w = mask.shape

# isola SOLO i pixel etichettati come pole (5) e mostrali su sfondo nero
canvas = np.zeros((h, w, 3), dtype=np.uint8)
canvas[mask == 5] = (255, 0, 0)   # blu = pole

# conta
n = int(np.sum(mask == 5))
print(f"{NOME}: pixel etichettati come pole (5): {n} ({n/(h*w)*100:.1f}%)")

cv2.putText(canvas, "BLU = cio che il dataset chiama 'pole'", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
out = BASE / f"controlla_pole_{NOME}.jpg"
cv2.imwrite(str(out), canvas)
print(f"Salvata: {out}")