# -*- coding: utf-8 -*-
"""
Estrae, per ogni alert nel CSV di detection, il frame QUERY (con il bounding
box disegnato sopra) affiancato al frame REFERENCE corrispondente (trovato
tramite la mappa di sincronizzazione — stesso box disegnato nella stessa
posizione, per confronto visivo immediato).

A differenza di estrai_frame_alert.py (che mostra solo il query annotato),
questo strumento risponde alla domanda "cosa mostra il reference in quella
stessa posizione/momento?" — utile per distinguere un'anomalia vera (il
reference mostra qualcosa di diverso/assente) da un artefatto di
allineamento/luce (il reference mostra sostanzialmente la stessa scena).

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python estrai_frame_confronto_alert.py --csv out/detections.csv \
        --query data/videos/query.mp4 --reference data/videos/reference.mp4 \
        --sync_map mappa_sync.csv --label ANOMALIA_STRUTTURALE
"""

import argparse
import csv as csv_module
import shutil
from pathlib import Path
from collections import defaultdict

import cv2

from sync_map import MappaSincronizzazione

COLORE_BOX = (0, 0, 255)


def leggi_frame(cap: cv2.VideoCapture, indice: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
    ok, frame = cap.read()
    return frame if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Percorso a detections.csv")
    ap.add_argument("--query", required=True, help="Percorso al video query")
    ap.add_argument("--reference", required=True, help="Percorso al video reference")
    ap.add_argument("--sync_map", required=True, help="Percorso al CSV di sync_videos_dtw.py")
    ap.add_argument("--label", default="ANOMALIA_STRUTTURALE",
                    help="Etichetta da confrontare (default: ANOMALIA_STRUTTURALE)")
    ap.add_argument("--out", default="frame_confronto", help="Cartella di output")
    ap.add_argument("--max-frame", type=int, default=None,
                    help="Limita il numero di frame_idx distinti da esportare (utile per un "
                         "controllo rapido su un campione invece di tutto)")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    if out_dir.exists():
        n_prima = len(list(out_dir.glob("*.jpg")))
        shutil.rmtree(out_dir)
        print(f"[INFO] Rimossa cartella {out_dir} (conteneva {n_prima} immagini)")
    out_dir.mkdir(parents=True, exist_ok=True)

    mappa = MappaSincronizzazione(args.sync_map)

    per_frame = defaultdict(list)
    with open(args.csv, newline='') as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            if row["class"] != args.label:
                continue
            per_frame[int(row["frame_idx"])].append(row)

    if not per_frame:
        print(f"[INFO] Nessuna detection trovata per l'etichetta: {args.label}")
        return

    frame_idx_ordinati = sorted(per_frame.keys())
    if args.max_frame:
        frame_idx_ordinati = frame_idx_ordinati[:args.max_frame]

    print(f"[INFO] {len(frame_idx_ordinati)} frame da esportare (su {len(per_frame)} totali "
          f"con almeno una detection '{args.label}'). Output in: {out_dir}")

    query_cap = cv2.VideoCapture(args.query)
    ref_cap = cv2.VideoCapture(args.reference)
    if not query_cap.isOpened() or not ref_cap.isOpened():
        print("[ERRORE] Impossibile aprire uno dei due video.")
        return

    salvati = 0
    senza_corrispondenza = 0

    for frame_idx in frame_idx_ordinati:
        rows = per_frame[frame_idx]

        query_frame = leggi_frame(query_cap, frame_idx)
        if query_frame is None:
            print(f"[WARN] Frame query {frame_idx} non leggibile, salto.")
            continue
        h, w = query_frame.shape[:2]

        indice_ref = mappa.frame_reference_per(frame_idx)
        reference_frame = None
        if indice_ref is not None:
            reference_frame = leggi_frame(ref_cap, indice_ref)
            if reference_frame is not None:
                reference_frame = cv2.resize(reference_frame, (w, h))

        query_annotato = query_frame.copy()
        if reference_frame is not None:
            reference_annotato = reference_frame.copy()
        else:
            # nessuna corrispondenza reference valida per questo frame: pannello
            # nero con scritta esplicita, invece di ometterlo silenziosamente
            reference_annotato = cv2.putText(
                (query_frame * 0).copy(), "nessuna corrispondenza reference",
                (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            senza_corrispondenza += 1

        for row in rows:
            x, y = int(float(row["x"])), int(float(row["y"]))
            bw, bh = int(float(row["w"])), int(float(row["h"]))
            cv2.rectangle(query_annotato, (x, y), (x + bw, y + bh), COLORE_BOX, 3)
            cv2.rectangle(reference_annotato, (x, y), (x + bw, y + bh), COLORE_BOX, 3)

        testo_query = f"QUERY frame {frame_idx} | t={rows[0]['time_s']}s"
        testo_ref = (f"REFERENCE frame {indice_ref}" if indice_ref is not None
                    else "REFERENCE (nessuna corrispondenza)")
        cv2.putText(query_annotato, testo_query, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        cv2.putText(reference_annotato, testo_ref, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        combinato = cv2.hconcat([query_annotato, reference_annotato])
        out_path = out_dir / f"confronto_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), combinato)
        salvati += 1

    query_cap.release()
    ref_cap.release()

    print(f"\n[FATTO] {salvati} confronti salvati in: {out_dir}/")
    if senza_corrispondenza:
        print(f"[INFO] {senza_corrispondenza} di questi non avevano corrispondenza reference "
              f"valida nella mappa (pannello destro nero).")
    print("Ogni immagine: QUERY a sinistra, REFERENCE a destra, stesso bounding box su entrambi.")


if __name__ == "__main__":
    main()
