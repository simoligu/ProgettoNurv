# -*- coding: utf-8 -*-
"""
Su quali classi del ground truth di RailSem19 cadono i segmenti che la
diagnostica dei pali ereditata (commit a3f700e, corpo di run) considera pali
candidati, e quelli che fanno scattare una segnalazione (capitolo 10).

Ogni segmento e' attribuito alla classe piu' frequente fra i pixel del ground
truth che attraversa (campionati a passo di un pixel).

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python analizza_candidati_pali.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

BASE = Path("data/rs19_val")
MASK_DIR = BASE / "uint8" / "rs19_val"
VAL_DIR = BASE / "images" / "val"


def main():
    nomi = [l["name"] for l in json.load(open(BASE / "rs19-config.json"))["labels"]]
    cand = np.zeros(256, np.int64)
    alert = np.zeros(256, np.int64)
    for p in sorted(VAL_DIR.glob("*.jpg")):
        f = cv2.imread(str(p))
        g = cv2.imread(str(MASK_DIR / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if f is None or g is None:
            continue
        h, w = g.shape
        roi = np.zeros((h, w), np.uint8)
        cv2.rectangle(roi, (0, 0), (int(w * 0.30), h), 255, -1)
        cv2.rectangle(roi, (int(w * 0.70), 0), (w, h), 255, -1)
        edges = cv2.bitwise_and(cv2.Canny(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 50, 150), roi)
        L = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=30)
        if L is None:
            continue
        for x1, y1, x2, y2 in L.reshape(-1, 4):
            dx, dy = int(x2) - int(x1), int(y2) - int(y1)
            if not (80 < abs(np.degrees(np.arctan2(dy, dx))) < 100) or abs(dy) <= h * 0.25:
                continue
            n = max(abs(dx), abs(dy)) + 1
            xs = np.clip(np.rint(np.linspace(x1, x2, n)).astype(int), 0, w - 1)
            ys = np.clip(np.rint(np.linspace(y1, y2, n)).astype(int), 0, h - 1)
            c = int(np.bincount(g[ys, xs], minlength=256).argmax())
            cand[c] += 1
            if abs(np.degrees(np.arctan(dx / dy))) >= 3:
                alert[c] += 1

    righe = ["Classe prevalente lungo i segmenti candidati 'palo' della diagnostica ereditata",
             "(1700 immagini di validazione RailSem19)", ""]
    for nome_gruppo, c in (("candidati", cand), ("segnalazioni (theta >= 3)", alert)):
        tot = c.sum()
        righe.append(f"{nome_gruppo}: {tot}")
        for i in np.argsort(c)[::-1][:8]:
            if c[i] == 0:
                break
            nome = nomi[i] if i < len(nomi) else f"valore {i}"
            righe.append(f"   {nome:14s} {c[i]:6d}  {100 * c[i] / tot:5.1f}%")
        righe.append("")
    testo = "\n".join(righe)
    print(testo)
    Path("risultati_tesi/candidati_pali_per_classe.txt").write_text(testo, encoding="utf-8")


if __name__ == "__main__":
    main()
