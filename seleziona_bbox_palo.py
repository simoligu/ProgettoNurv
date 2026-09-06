# -*- coding: utf-8 -*-
"""
Selezione interattiva del bbox di un palo: apre l'immagine in una finestra,
clicca l'angolo in alto a sinistra e poi quello in basso a destra del palo
(due click in totale) — lo script calcola e stampa x,y,w,h pronto da
incollare in valida_pole_tilt_rotazione_2d.py.

Comandi:
  - Click sinistro: imposta un punto (il primo poi il secondo)
  - 'r': resetta i due punti se hai sbagliato
  - 'q' o ESC: esce (chiede conferma se hai gia' due punti validi)

Lancio (dalla radice di ProgettoNurv):
    python seleziona_bbox_palo.py --immagine frame_per_validazione_pole_tilt.jpg
"""

import argparse
import cv2

punti = []


def click_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(punti) < 2:
            punti.append((x, y))
            print(f"Punto {len(punti)}: ({x}, {y})")
        else:
            print("Hai gia' due punti — premi 'r' per resettare se vuoi cambiarli.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--immagine", required=True)
    args = ap.parse_args()

    img = cv2.imread(args.immagine)
    if img is None:
        print(f"[ERRORE] Impossibile aprire {args.immagine}")
        return

    h_img, w_img = img.shape[:2]
    print(f"Immagine {w_img}x{h_img}. Clicca l'angolo in ALTO-SINISTRA del palo, poi "
          f"quello in BASSO-DESTRA. 'r' per ricominciare, 'q'/ESC per uscire.\n")

    cv2.namedWindow("Seleziona il palo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seleziona il palo", min(1600, w_img), min(900, h_img))
    cv2.setMouseCallback("Seleziona il palo", click_callback)

    while True:
        anteprima = img.copy()
        for p in punti:
            cv2.drawMarker(anteprima, p, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        if len(punti) == 2:
            (x1, y1), (x2, y2) = punti
            x_min, y_min = min(x1, x2), min(y1, y2)
            x_max, y_max = max(x1, x2), max(y1, y2)
            cv2.rectangle(anteprima, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)

        cv2.imshow("Seleziona il palo", anteprima)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            punti.clear()
            print("Punti resettati.")
        elif key == ord('q') or key == 27:  # ESC
            break

    cv2.destroyAllWindows()

    if len(punti) == 2:
        (x1, y1), (x2, y2) = punti
        x_min, y_min = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2), max(y1, y2)
        w, h = x_max - x_min, y_max - y_min
        print(f"\n=== BBOX SELEZIONATO ===")
        print(f"x={x_min}, y={y_min}, w={w}, h={h}")
        print(f"\nDa usare cosi':")
        print(f"  --bbox {x_min},{y_min},{w},{h}")
    else:
        print("\nNessun bbox completo selezionato (servono 2 punti).")


if __name__ == "__main__":
    main()
