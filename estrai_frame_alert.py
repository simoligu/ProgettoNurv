# -*- coding: utf-8 -*-
"""
Estrae dal video query i frame specifici in cui sono scattati alert, per verifica
visiva. Utile per controllare se SCARTAMENTO_ANOMALO / VEGETAZIONE_INVASIVA / 
PALO_INCLINATO hanno senso guardando l'immagine reale, non solo il numero nel CSV.

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python estrai_frame_alert.py --csv out/detections.csv --video data/videos/query.mp4
    python estrai_frame_alert.py --csv out/detections.csv --video data/videos/query.mp4 --label SCARTAMENTO_ANOMALO
"""

import argparse
import csv as csv_module
import shutil
from pathlib import Path
from collections import defaultdict
import cv2

# etichette del modulo DeepLab (quelle interessanti da verificare)
DEEPLAB_LABELS = {"SCARTAMENTO_ANOMALO", "PALO_INCLINATO", "VEGETAZIONE_INVASIVA"}

COLORI = {
    "SCARTAMENTO_ANOMALO": (0, 165, 255),
    "PALO_INCLINATO": (255, 0, 0),
    "VEGETAZIONE_INVASIVA": (0, 200, 0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Percorso a detections.csv")
    ap.add_argument("--video", required=True, help="Percorso al video query")
    ap.add_argument("--label", default=None,
                    help="Filtra su una sola etichetta (default: tutte quelle DeepLab)")
    ap.add_argument("--out", default="frame_verifica", help="Cartella di output")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    if out_dir.exists():
        n_prima = len(list(out_dir.glob("*.jpg")))
        shutil.rmtree(out_dir)
        print(f"[INFO] Rimossa cartella {out_dir} (conteneva {n_prima} immagini)")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output in: {out_dir}")

    labels_wanted = {args.label} if args.label else DEEPLAB_LABELS

    # raggruppa le detection per frame_idx (un frame puo' avere piu' box)
    per_frame = defaultdict(list)
    with open(args.csv, newline='') as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            label = row["class"]
            if label not in labels_wanted:
                continue
            per_frame[int(row["frame_idx"])].append(row)

    if not per_frame:
        print(f"[INFO] Nessuna detection trovata per le etichette: {labels_wanted}")
        return

    print(f"[INFO] Trovati {len(per_frame)} frame con alert DeepLab. Estraggo...")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire {args.video}")
        return

    salvati = 0
    for frame_idx, rows in sorted(per_frame.items()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] Frame {frame_idx} non leggibile, salto.")
            continue

        for row in rows:
            label = row["class"]
            x, y, w, h = int(float(row["x"])), int(float(row["y"])), int(float(row["w"])), int(float(row["h"]))
            color = COLORI.get(label, (0, 0, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            testo = f"{label} conf={row['conf']}"
            cv2.putText(frame, testo, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # etichetta con frame_idx e time_s in alto
        t_sec = rows[0]["time_s"]
        cv2.putText(frame, f"frame {frame_idx} | t={t_sec}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        salvati += 1

    cap.release()
    print(f"\n[FATTO] {salvati} frame salvati in: {out_dir}/")
    print("Apri la cartella e sfoglia le immagini per la verifica visiva.")


if __name__ == "__main__":
    main()