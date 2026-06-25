import cv2, numpy as np
from pathlib import Path

BASE = Path("data/rs19_val")
mask = cv2.imread(str(BASE/"uint8"/"rs19_val"/"rs00000.png"), cv2.IMREAD_GRAYSCALE)

# pixel di rotaia nella maschera ORIGINALE
pixel_orig = int(np.sum(mask == 17))
print(f"Pixel rotaia (valore 17) nella maschera originale: {pixel_orig}")

# ora simula cosa sopravvive al filtro MIN_AREA
binary = (mask == 17).astype(np.uint8) * 255
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

MIN_AREA = 200
tenuti = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
scartati = [c for c in contours if cv2.contourArea(c) < MIN_AREA]

area_tenuta = sum(cv2.contourArea(c) for c in tenuti)
area_scartata = sum(cv2.contourArea(c) for c in scartati)

print(f"Contorni totali di rotaia: {len(contours)}")
print(f"  tenuti (area >= {MIN_AREA}): {len(tenuti)} -> area {area_tenuta:.0f}")
print(f"  scartati (troppo piccoli): {len(scartati)} -> area {area_scartata:.0f}")