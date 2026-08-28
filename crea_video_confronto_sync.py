# -*- coding: utf-8 -*-
"""
Crea un video di confronto QUERY | REFERENCE affiancati, frame per frame,
seguendo la mappa di sincronizzazione — per guardare la corrispondenza come
un video normale invece di controllare coppie di frame statici una alla
volta. Utile per un ultimo controllo "a occhio" su tutto il percorso, dopo
aver gia' verificato numericamente la mappa.

Per ogni frame RAW del query (o 1 ogni N, vedi --ogni-n-frame), pesca il
frame reference corrispondente tramite MappaSincronizzazione (la stessa
interpolazione usata dalla pipeline in produzione, quindi il video mostra
ESATTAMENTE cio' che la pipeline vedrebbe). Dove non c'e' corrispondenza
valida, il pannello reference mostra un frame nero con scritta esplicita,
invece di lasciare un buco silenzioso.

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python crea_video_confronto_sync.py --query data\\videos\\query.mp4 \
        --reference data\\videos\\reference.mp4 --sync_map mappa_sync.csv \
        --output confronto_sync.mp4

Per un test rapido su un tratto breve invece di tutto il video:
    python crea_video_confronto_sync.py --query data\\videos\\query.mp4 \
        --reference data\\videos\\reference.mp4 --sync_map mappa_sync.csv \
        --output confronto_test.mp4 --max-frame 1000
"""

import argparse

import cv2
import numpy as np

from sync_map import MappaSincronizzazione


class LettoreReferenceSequenziale:
    """
    Legge frame dal video reference in modo ottimizzato: se il prossimo
    indice richiesto e' vicino alla posizione attuale (avanti, entro una
    piccola soglia), avanza in sequenza con .read() — molto piu' veloce di
    un seek (.set()) per ogni singolo frame, che su video lunghi renderebbe
    la generazione del confronto molto lenta. Il seek resta disponibile
    come fallback per salti grandi o all'indietro (raro, dato che la mappa
    e' garantita monotona non decrescente, ma gestito per sicurezza).
    """

    def __init__(self, percorso_video: str, soglia_salto_sequenziale: int = 30):
        self.cap = cv2.VideoCapture(percorso_video)
        if not self.cap.isOpened():
            raise IOError(f"Impossibile aprire il video: {percorso_video}")
        self.posizione_attuale = -1
        self.soglia_salto_sequenziale = soglia_salto_sequenziale

    def leggi(self, indice_target: int):
        if self.posizione_attuale == -1 or indice_target < self.posizione_attuale \
                or indice_target - self.posizione_attuale > self.soglia_salto_sequenziale:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, indice_target)
            self.posizione_attuale = indice_target
        else:
            while self.posizione_attuale < indice_target:
                ok, _ = self.cap.read()
                if not ok:
                    return None
                self.posizione_attuale += 1

        ok, frame = self.cap.read()
        if not ok:
            return None
        self.posizione_attuale += 1
        return frame

    def release(self):
        self.cap.release()


def main():
    ap = argparse.ArgumentParser(description="Crea un video di confronto query/reference affiancati")
    ap.add_argument("--query", required=True, help="Percorso al video query")
    ap.add_argument("--reference", required=True, help="Percorso al video reference")
    ap.add_argument("--sync_map", required=True, help="Percorso al CSV di sync_videos_dtw.py")
    ap.add_argument("--output", default="confronto_sync.mp4", help="Percorso del video di output")
    ap.add_argument("--max-frame", type=int, default=None,
                    help="Limita il numero di frame query da processare (utile per un test rapido "
                         "su un tratto breve invece di tutto il video)")
    ap.add_argument("--ogni-n-frame", type=int, default=1,
                    help="Processa 1 frame query ogni N (default 1 = tutti, playback fluido a "
                         "velocita' naturale). Un valore piu' alto produce un video piu' corto "
                         "ma piu' 'a scatti' — utile solo per un'ispezione molto rapida.")
    ap.add_argument("--larghezza-output", type=int, default=None,
                    help="Ridimensiona ogni pannello a questa larghezza prima di affiancarli "
                         "(default: nessun ridimensionamento, usa la risoluzione nativa del "
                         "query — puo' produrre un file grande su video ad alta risoluzione)")
    args = ap.parse_args()

    print(f"[INFO] Carico la mappa di sincronizzazione da {args.sync_map}...")
    mappa = MappaSincronizzazione(args.sync_map)

    query_cap = cv2.VideoCapture(args.query)
    if not query_cap.isOpened():
        print(f"[ERRORE] Impossibile aprire il video query: {args.query}")
        return
    lettore_ref = LettoreReferenceSequenziale(args.reference)

    fps = query_cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(query_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(query_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.larghezza_output:
        scala = args.larghezza_output / w
        w_out, h_out = args.larghezza_output, int(round(h * scala))
    else:
        w_out, h_out = w, h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps / max(1, args.ogni_n_frame), (w_out * 2, h_out))
    if not writer.isOpened():
        print(f"[ERRORE] Impossibile aprire il video in scrittura: {args.output}")
        return

    print(f"[INFO] Genero il confronto (fps={fps:.1f}, risoluzione pannello {w_out}x{h_out})...")

    idx = 0
    scritti = 0
    saltati_no_corrispondenza = 0
    while True:
        ok, qf = query_cap.read()
        if not ok:
            break
        if args.max_frame is not None and idx >= args.max_frame:
            break

        if idx % args.ogni_n_frame == 0:
            indice_ref = mappa.frame_reference_per(idx)

            if indice_ref is not None:
                rf = lettore_ref.leggi(indice_ref)
                testo_ref = f"REFERENCE frame {indice_ref}"
            else:
                rf = None
                testo_ref = "REFERENCE: nessuna corrispondenza"
                saltati_no_corrispondenza += 1

            if rf is None:
                rf = np.zeros((h, w, 3), dtype=np.uint8)

            qf_out = cv2.resize(qf, (w_out, h_out)) if args.larghezza_output else qf.copy()
            rf_out = cv2.resize(rf, (w_out, h_out))

            cv2.putText(qf_out, f"QUERY frame {idx} | t={idx / fps:.2f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(rf_out, testo_ref,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            combinato = cv2.hconcat([qf_out, rf_out])
            writer.write(combinato)
            scritti += 1

            if scritti % 300 == 0:
                print(f"  ...{scritti} frame scritti (query frame {idx}, t={idx / fps:.1f}s)")

        idx += 1

    query_cap.release()
    lettore_ref.release()
    writer.release()

    print(f"\n[FATTO] {scritti} frame scritti in: {args.output}")
    if saltati_no_corrispondenza:
        print(f"[INFO] {saltati_no_corrispondenza} di questi mostrano 'nessuna corrispondenza' "
              f"(pannello reference nero) — coerente con i tratti scartati dalla mappa.")


if __name__ == "__main__":
    main()
