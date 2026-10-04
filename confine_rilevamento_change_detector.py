# -*- coding: utf-8 -*-
"""
Confine di rilevamento del modulo ANOMALIA_STRUTTURALE (capitolo 8 della tesi).

Per tre configurazioni di soglia misura, con anomalie sintetiche quadrate su un
fotogramma Full HD vuoto, la funzione mask_to_boxes realmente usata dalla
pipeline:
  - il lato minimo rilevato, a intensita' 255;
  - l'intensita' minima rilevata, per un quadrato ben sopra l'area minima
    (da 2 in su: con valori massimi pari a 1 mask_to_boxes tratta l'immagine
    come maschera binaria e non applica la soglia).

Configurazioni:
  - ereditata:        diff_thresh 90,  min_area 8000
  - calibrata:        diff_thresh 150, min_area 60000
  - con compattezza:  diff_thresh 150, min_area 15000, min_compattezza 0.5

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python confine_rilevamento_change_detector.py
"""

from pathlib import Path

import numpy as np

from background import mask_to_boxes

H, W = 1080, 1920
CONFIG = [
    ("ereditata (90 / 8000)", 90, 8000, None),
    ("calibrata (150 / 60000)", 150, 60000, None),
    ("con compattezza (150 / 15000 / 0,5)", 150, 15000, 0.5),
]


def quadrato(lato, intensita):
    d = np.zeros((H, W), np.uint8)
    cy, cx = H // 2, W // 2
    d[cy - lato // 2: cy - lato // 2 + lato, cx - lato // 2: cx - lato // 2 + lato] = intensita
    return d


def rilevato(d, dt, ma, mc):
    boxes, _ = mask_to_boxes(d, diff_thresh=dt, min_area=ma, min_compattezza=mc)
    return len(boxes) > 0


def main():
    righe = ["Confine di rilevamento di mask_to_boxes con anomalie quadrate sintetiche",
             f"(fotogramma {W}x{H}, area {W * H} pixel)", ""]
    for nome, dt, ma, mc in CONFIG:
        lato_min = next((l for l in range(10, 600) if rilevato(quadrato(l, 255), dt, ma, mc)), None)
        lato_prova = max(lato_min or 0, 400)
        int_min = next((i for i in range(2, 256) if rilevato(quadrato(lato_prova, i), dt, ma, mc)), None)
        perc = 100.0 * lato_min * lato_min / (W * H) if lato_min else float("nan")
        righe.append(f"{nome}:")
        righe.append(f"   lato minimo rilevato (intensita' 255): {lato_min} px"
                     f"  = {perc:.2f}% del fotogramma")
        righe.append(f"   intensita' minima rilevata (quadrato di {lato_prova} px): {int_min}")
        righe.append("")
    testo = "\n".join(righe)
    print(testo)
    Path("risultati_tesi/confine_rilevamento_change_detector.txt").write_text(testo, encoding="utf-8")


if __name__ == "__main__":
    main()
