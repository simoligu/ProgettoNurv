# -*- coding: utf-8 -*-
"""
Estrae frame specifici da un video (per indice), per confrontare a occhio
le corrispondenze trovate da sync_videos_dtw.py.

USO — un singolo frame:
    python estrai_frame.py --video data\\videos\\reference.mp4 --frame 2085 --output ref_2085.png

USO — piu' frame in un colpo solo, utile per controllare piu' righe della
mappa di sincronizzazione insieme:
    python estrai_frame.py --video data\\videos\\query.mp4 --frame 3110 3040 1025 --output-dir confronto\\

USO — direttamente da alcune righe del CSV di sincronizzazione (estrae sia
il frame query sia il corrispondente frame reference per ogni riga scelta,
con nomi file che si affiancano facilmente in ordine alfabetico):
    python estrai_frame.py --da-mappa mappa_sync.csv --reference data\\videos\\reference.mp4 \
        --query data\\videos\\query.mp4 --righe 0 100 300 622 --output-dir confronto\\
"""

import argparse
import csv
import os

import cv2


def estrai_frame_singolo(percorso_video: str, indice_frame: int):
    cap = cv2.VideoCapture(percorso_video)
    if not cap.isOpened():
        raise IOError(f"Impossibile aprire il video: {percorso_video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, indice_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Impossibile leggere il frame {indice_frame} da {percorso_video} "
                         f"(indice fuori range?)")
    return frame


def main():
    ap = argparse.ArgumentParser(description="Estrae frame da un video, per indice")
    ap.add_argument("--video", help="Percorso al video (modalita' singolo video)")
    ap.add_argument("--frame", type=int, nargs="+",
                     help="Uno o piu' indici di frame da estrarre (modalita' singolo video)")
    ap.add_argument("--output", help="Percorso del PNG di output (solo se un unico --frame)")
    ap.add_argument("--output-dir", default=".", help="Cartella di output (default: cartella corrente)")

    ap.add_argument("--da-mappa", help="Percorso al CSV prodotto da sync_videos_dtw.py "
                                        "(modalita' confronto da mappa)")
    ap.add_argument("--reference", help="Percorso al video reference (con --da-mappa)")
    ap.add_argument("--query", help="Percorso al video query (con --da-mappa)")
    ap.add_argument("--righe", type=int, nargs="+",
                     help="Indici di RIGA del CSV da estrarre (0-based, non frame_query — "
                          "es. '--righe 0 100 300 622' prende la prima, una intermedia, "
                          "un'altra intermedia e l'ultima riga del CSV)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.da_mappa:
        # --- modalita' confronto da mappa: estrae coppie query/reference ---
        if not (args.reference and args.query and args.righe):
            print("[ERRORE] --da-mappa richiede anche --reference, --query e --righe")
            return

        with open(args.da_mappa, "r", encoding="utf-8") as f:
            righe_csv = list(csv.DictReader(f))

        for indice_riga in args.righe:
            if indice_riga >= len(righe_csv):
                print(f"[SKIP] Riga {indice_riga} fuori range (il CSV ha {len(righe_csv)} righe)")
                continue
            r = righe_csv[indice_riga]
            fq, fr = int(r["frame_query"]), int(r["frame_reference"])

            frame_q = estrai_frame_singolo(args.query, fq)
            frame_r = estrai_frame_singolo(args.reference, fr)

            nome_q = os.path.join(args.output_dir, f"riga{indice_riga:04d}_A_query_frame{fq}.png")
            nome_r = os.path.join(args.output_dir, f"riga{indice_riga:04d}_B_reference_frame{fr}.png")
            cv2.imwrite(nome_q, frame_q)
            cv2.imwrite(nome_r, frame_r)
            print(f"Riga {indice_riga}: query#{fq} -> {nome_q}")
            print(f"           reference#{fr} -> {nome_r}")

        print(f"\nFatto. Apri la cartella '{args.output_dir}' e confronta le coppie A/B "
              f"(stesso prefisso 'rigaXXXX') — sono ordinate alfabeticamente una accanto all'altra.")

    elif args.video and args.frame:
        # --- modalita' singolo video ---
        if len(args.frame) == 1 and args.output:
            frame = estrai_frame_singolo(args.video, args.frame[0])
            cv2.imwrite(args.output, frame)
            print(f"Salvato: {args.output}")
        else:
            for idx in args.frame:
                frame = estrai_frame_singolo(args.video, idx)
                nome_file = os.path.join(args.output_dir, f"frame_{idx}.png")
                cv2.imwrite(nome_file, frame)
                print(f"Salvato: {nome_file}")
    else:
        print("[ERRORE] Specifica --video + --frame, oppure --da-mappa + --reference + --query + --righe. "
              "Vedi --help per esempi.")


if __name__ == "__main__":
    main()
