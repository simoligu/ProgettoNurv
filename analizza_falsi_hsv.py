# -*- coding: utf-8 -*-
"""
A quali classi del ground truth di RailSem19 appartengono i pixel che la soglia
HSV della diagnostica ereditata ([20,30,20]-[95,255,255]) classifica come
vegetazione senza che lo siano (capitolo 10). Immagine intera e terzo inferiore.

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python analizza_falsi_hsv.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

BASE = Path("data/rs19_val")
MASK_DIR = BASE / "uint8" / "rs19_val"
VAL_DIR = BASE / "images" / "val"
VEG = 8


def main():
    nomi = [l["name"] for l in json.load(open(BASE / "rs19-config.json"))["labels"]]
    conta = {"intera": np.zeros(256, np.int64), "roi33": np.zeros(256, np.int64)}
    for p in sorted(VAL_DIR.glob("*.jpg")):
        f = cv2.imread(str(p))
        g = cv2.imread(str(MASK_DIR / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if f is None or g is None:
            continue
        h = g.shape[0]
        r0 = h - int(h * 33 / 100)
        v = cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV), np.array([20, 30, 20]),
                        np.array([95, 255, 255])) > 0
        fp = v & (g != VEG)
        conta["intera"] += np.bincount(g[fp], minlength=256)
        conta["roi33"] += np.bincount(g[r0:][fp[r0:]], minlength=256)

    L = ["Pixel classificati verdi dalla soglia HSV ma non 'vegetation' nel ground truth",
         "(1700 immagini di validazione RailSem19), ripartiti per classe vera", ""]
    for zona, c in conta.items():
        tot = c.sum()
        L.append(f"{zona}: {tot} pixel")
        for i in np.argsort(c)[::-1][:8]:
            if c[i] == 0:
                break
            nome = nomi[i] if i < len(nomi) else f"valore {i}"
            L.append(f"   {nome:14s} {100 * c[i] / tot:5.1f}%")
        L.append("")
    testo = "\n".join(L)
    print(testo)
    out = Path("risultati_tesi/falsi_hsv_per_classe.txt")
    out.write_text(testo, encoding="utf-8")


if __name__ == "__main__":
    main()
