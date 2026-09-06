# -*- coding: utf-8 -*-
"""
Valida PoleTiltAnalyzer su una foto REALE di un oggetto rigido dritto
(manico di scopa, tubo, paletto...) fotografato a un angolo di inclinazione
NOTO (misurato con la livella/goniometro dello smartphone) — senza nessuna
rotazione sintetica: il test piu' pulito possibile, perche' la foto e'
gia' una vera fotografia dell'oggetto inclinato, con la prospettiva e la
texture reali.

Pensato per essere lanciato UNA VOLTA PER FOTO, via via che ne scatti altre
a angoli diversi: ogni lancio aggiunge una riga al CSV dei risultati
(--csv_output, default valida_pole_tilt_foto_reali.csv) e ristampa il MAE
complessivo calcolato su TUTTE le foto testate finora — non serve rilanciare
tutto da capo ogni volta.

Lancio (dalla radice di ProgettoNurv, con .venv attivo), una volta per foto:
    python valida_pole_tilt_foto_reali.py --immagine foto_10gradi.jpg \
        --bbox 500,300,60,400 --angolo_vero 10 --pesi weights/pole_tilt_best.pt

--bbox: usa seleziona_bbox_palo_matplotlib.py sulla stessa foto per ottenerlo.
--angolo_vero: l'angolo che hai impostato con la livella/goniometro, in
               gradi. Usa il segno che preferisci per "sinistra"/"destra",
               basta restare coerente tra le foto (il segno assoluto non
               conta, conta la coerenza per calcolare correttamente l'errore).
"""

import argparse
import csv
from pathlib import Path

import cv2

from pole_tilt_analyzer import PoleTiltAnalyzer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--immagine", required=True)
    ap.add_argument("--bbox", required=True, help="x,y,w,h del bbox stretto attorno all'oggetto")
    ap.add_argument("--angolo_vero", type=float, required=True,
                     help="Angolo di inclinazione reale, misurato con livella/goniometro (gradi)")
    ap.add_argument("--pesi", default="weights/pole_tilt_best.pt")
    ap.add_argument("--angolo_max", type=float, default=22.0)
    ap.add_argument("--dimensione_input", type=int, default=224)
    ap.add_argument("--margine_bbox_frazione", type=float, default=0.25)
    ap.add_argument("--csv_output", default="valida_pole_tilt_foto_reali.csv")
    args = ap.parse_args()

    x, y, w, h = (int(v) for v in args.bbox.split(","))

    img = cv2.imread(args.immagine)
    if img is None:
        print(f"[ERRORE] Impossibile aprire {args.immagine}")
        return

    analyzer = PoleTiltAnalyzer(
        weights_path=args.pesi,
        angolo_max=args.angolo_max,
        dimensione_input=args.dimensione_input,
        margine_bbox_frazione=args.margine_bbox_frazione,
    )

    angolo_predetto = analyzer.predict_angle(img, (x, y, w, h))

    if angolo_predetto is None:
        print("[ERRORE] Crop non valido (bbox degenere) — controlla le coordinate del bbox.")
        return

    errore = angolo_predetto - args.angolo_vero
    print(f"\nAngolo vero (misurato): {args.angolo_vero:+.1f}°")
    print(f"Angolo stimato dal CNN: {angolo_predetto:+.2f}°")
    print(f"Errore: {errore:+.2f}°")

    # salva un ritaglio annotato per controllo visivo rapido
    annotato = img.copy()
    cv2.rectangle(annotato, (x, y), (x + w, y + h), (0, 165, 255), 3)
    testo = f"vero={args.angolo_vero:+.1f} CNN={angolo_predetto:+.2f}"
    cv2.putText(annotato, testo, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    out_img_path = Path(args.immagine).with_stem(Path(args.immagine).stem + "_annotato")
    cv2.imwrite(str(out_img_path), annotato)
    print(f"Immagine annotata salvata in: {out_img_path}")

    # accumula il risultato nel CSV (crea l'header se il file non esiste ancora)
    csv_path = Path(args.csv_output)
    esiste_gia = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not esiste_gia:
            writer.writerow(["immagine", "angolo_vero", "angolo_predetto", "errore"])
        writer.writerow([args.immagine, args.angolo_vero, angolo_predetto, errore])

    # ricalcola e mostra il MAE su tutte le foto testate finora
    with open(csv_path, newline="") as f:
        righe = list(csv.DictReader(f))
    errori = [float(r["errore"]) for r in righe]
    mae = sum(abs(e) for e in errori) / len(errori)
    print(f"\n=== MAE COMPLESSIVO su {len(righe)} foto testate finora: {mae:.2f}° ===")
    print(f"(confronta con MAE=1.24° del test originale su render Unreal sintetici)")


if __name__ == "__main__":
    main()
