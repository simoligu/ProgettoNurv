# -*- coding: utf-8 -*-
"""
Selezione interattiva del bbox di un palo — versione matplotlib, per chi ha
opencv-python-headless installato (niente supporto GUI in cv2.namedWindow,
errore tipico: "The function is not implemented... Rebuild the library
with Windows, GTK+ 2.x or Cocoa support").

Stesso scopo di seleziona_bbox_palo.py, motore grafico diverso.

Lancio (dalla radice di ProgettoNurv):
    python seleziona_bbox_palo_matplotlib.py --immagine frame_verifica/frame_000510.jpg

Uso: clicca l'angolo in ALTO-SINISTRA del palo, poi quello in BASSO-DESTRA
(due click). Chiudi la finestra quando hai finito — lo script stampa il
bbox risultante.
"""

import argparse
import cv2
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--immagine", required=True)
    args = ap.parse_args()

    img_bgr = cv2.imread(args.immagine)
    if img_bgr is None:
        print(f"[ERRORE] Impossibile aprire {args.immagine}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]

    print(f"Immagine {w_img}x{h_img}.")
    print("Nella finestra che si apre: clicca l'angolo ALTO-SINISTRA del palo, poi "
          "quello BASSO-DESTRA (due click in totale). Chiudi la finestra quando hai finito.\n")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img_rgb)
    ax.set_title("Clicca 2 punti: alto-sinistra e basso-destra del palo")

    # plt.ginput blocca l'esecuzione finche' non vengono raccolti n_punti click
    # (o si preme un tasto/si chiude la finestra prima)
    punti = plt.ginput(n=2, timeout=0)  # timeout=0 -> aspetta indefinitamente
    plt.close(fig)

    if len(punti) < 2:
        print("\nSelezione incompleta (servono 2 click). Rilancia lo script.")
        return

    (x1, y1), (x2, y2) = punti
    x_min, y_min = int(round(min(x1, x2))), int(round(min(y1, y2)))
    x_max, y_max = int(round(max(x1, x2))), int(round(max(y1, y2)))
    w, h = x_max - x_min, y_max - y_min

    print(f"\n=== BBOX SELEZIONATO ===")
    print(f"x={x_min}, y={y_min}, w={w}, h={h}")
    print(f"\nDa usare cosi':")
    print(f"  --bbox {x_min},{y_min},{w},{h}")

    # mostra un'anteprima con il bbox disegnato, per conferma visiva
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.imshow(img_rgb)
    rect = plt.Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor='orange', facecolor='none')
    ax2.add_patch(rect)
    ax2.set_title(f"Bbox scelto: x={x_min}, y={y_min}, w={w}, h={h} — chiudi per uscire")
    plt.show()


if __name__ == "__main__":
    main()
