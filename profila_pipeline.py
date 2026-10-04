"""
Profilazione dei tempi della pipeline, blocco per blocco.

Costruisce la pipeline esattamente come main.py (stessi video, stesso
sample_step, stessi modelli) e ne misura i tempi DALL'ESTERNO, avvolgendo con
un cronometro le funzioni che compongono ciascun blocco del ciclo principale:
nessuna riga di pipeline.py viene modificata.

Blocchi misurati:
  - allineamento omografico   FrameAligner.compute_homography + cv2.warpPerspective
  - sottrazione dello sfondo  ChangeDetector.detect_changes_gray + normalizzazione di luminanza
  - lettura del riferimento   lettura a indice del fotogramma reference (mappa di sync)
  - YOLO (ostacoli)           inferenza del rilevatore di ostacoli, su ogni fotogramma
  - DeepLab                   segmentazione e analisi (scartamento, vegetazione, pali)
      di cui regressore pali  stima CNN dell'inclinazione, chiamata dentro l'analisi
  - invio degli allarmi       codifica JPEG del fotogramma + richiesta HTTP
  - altro                     differenza col totale: decodifica del video query,
                              CLAHE, disegno e scrittura del video annotato, CSV

Gli allarmi vengono inviati a un piccolo server locale che risponde subito 200,
cosi' il tempo di rete non dipende dal fatto che la dashboard sia accesa.

Uso:
    python profila_pipeline.py --tratta 1
Output: tabella a schermo e in risultati_tesi/profilazione_tempi_pipeline.txt
"""
import argparse
import os
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import torch
from ultralytics import YOLO

import sync_videos_dtw
from alignment import FrameAligner
from detection import ChangeDetector
from pipeline import AnomalyDetectionPipeline
from sync_cache import mappa_e_valida, salva_metadati_mappa

TEMPI = defaultdict(float)
CHIAMATE = defaultdict(int)


def cronometra(nome, funzione):
    def avvolta(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return funzione(*args, **kwargs)
        finally:
            TEMPI[nome] += time.perf_counter() - t0
            CHIAMATE[nome] += 1
    return avvolta


class ClassificatoreCronometrato:
    """Avvolge il modello YOLO mantenendone l'interfaccia usata dalla pipeline."""

    def __init__(self, modello):
        self._modello = modello
        self.names = modello.names

    def __call__(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return self._modello(*args, **kwargs)
        finally:
            TEMPI["YOLO (ostacoli)"] += time.perf_counter() - t0
            CHIAMATE["YOLO (ostacoli)"] += 1


class _RispostaImmediata(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tratta", type=int, required=True)
    ap.add_argument("--reference", default="data/videos/reference.mp4")
    ap.add_argument("--query", default="data/videos/query.mp4")
    ap.add_argument("--out_dir", default="out/profilazione")
    ap.add_argument("--porta_alert", type=int, default=8091)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- mappa di sincronizzazione dedicata, per non toccare quella del progetto ---
    mappa = os.path.join(args.out_dir, "mappa_sync_profilazione.csv")
    if not mappa_e_valida(mappa, args.reference, args.query):
        print("[profilazione] genero la mappa di sincronizzazione (tempo escluso dalla misura)")
        sync_videos_dtw.main(["--reference", args.reference, "--query", args.query,
                              "--output", mappa, "--step", "5",
                              "--fattore-margine-raffinamento", "4.0",
                              "--min-run-plateau-interno", "50"])
        salva_metadati_mappa(mappa, args.reference, args.query)

    # --- server locale che accetta gli allarmi e risponde subito ---
    server = HTTPServer(("127.0.0.1", args.porta_alert), _RispostaImmediata)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[profilazione] dispositivo per le reti: {device}")

    pipeline = AnomalyDetectionPipeline(
        reference_video=args.reference,
        query_video=args.query,
        out_dir=args.out_dir,
        sample_step=8,
        use_classifier=True,
        classifier=ClassificatoreCronometrato(YOLO("yolov8n.pt")),
        alert_endpoint=f"http://127.0.0.1:{args.porta_alert}/api/alerts",
        tratta_id=args.tratta,
        deeplab_weights="runs_seg/deeplab_hires/best.pt",
        deeplab_imgsz=896,
        seg_step=30,
        gauge_tolerance=0.15,
        pole_tilt_weights="weights/pole_tilt_best.pt",
        pole_tilt_angolo_max=22.0,
        sync_map_csv=mappa,
        diff_thresh=150,
        min_area=15000,
        min_compattezza=0.5,
        usa_temporal_tracker=True,
    )

    # --- cronometri attorno ai blocchi, senza modificare pipeline.py ---
    FrameAligner.compute_homography = cronometra(
        "allineamento omografico", FrameAligner.compute_homography)
    cv2.warpPerspective = cronometra("allineamento omografico", cv2.warpPerspective)
    ChangeDetector.detect_changes_gray = staticmethod(cronometra(
        "sottrazione dello sfondo", ChangeDetector.detect_changes_gray))
    AnomalyDetectionPipeline._normalizza_luminanza = staticmethod(cronometra(
        "sottrazione dello sfondo", AnomalyDetectionPipeline._normalizza_luminanza))
    pipeline._leggi_frame_reference_per_indice = cronometra(
        "lettura del riferimento", pipeline._leggi_frame_reference_per_indice)
    pipeline.analyzer.analyze = cronometra("DeepLab", pipeline.analyzer.analyze)
    if pipeline.pole_tilt_analyzer is not None:
        pipeline.pole_tilt_analyzer.predict_angle = cronometra(
            "di cui regressore pali", pipeline.pole_tilt_analyzer.predict_angle)
    pipeline.dispatch_alert = cronometra("invio degli allarmi", pipeline.dispatch_alert)

    t0 = time.perf_counter()
    pipeline.run()
    totale = time.perf_counter() - t0
    server.shutdown()

    blocchi = ["allineamento omografico", "sottrazione dello sfondo", "lettura del riferimento",
               "YOLO (ostacoli)", "DeepLab", "invio degli allarmi"]
    misurato = sum(TEMPI[b] for b in blocchi)
    TEMPI["altro"] = totale - misurato

    righe = [f"Profilazione della pipeline — query: {args.query}, reference: {args.reference}",
             f"Dispositivo per le reti: {device}; sample_step=8, seg_step=30, sync_map attiva",
             f"Tempo totale di run(): {totale:.1f} s ({int(totale // 60)} min {totale % 60:.0f} s)",
             "",
             f"{'blocco':28s} {'secondi':>9s} {'% totale':>9s} {'chiamate':>9s} {'ms/chiamata':>12s}"]
    for b in blocchi + ["di cui regressore pali", "altro"]:
        n = CHIAMATE.get(b, 0)
        ms = (TEMPI[b] / n * 1000) if n else float("nan")
        righe.append(f"{b:28s} {TEMPI[b]:9.1f} {TEMPI[b] / totale * 100:8.1f}% {n:9d} {ms:12.1f}")
    testo = "\n".join(righe)
    print("\n" + testo)
    os.makedirs("risultati_tesi", exist_ok=True)
    with open("risultati_tesi/profilazione_tempi_pipeline.txt", "w", encoding="utf-8") as f:
        f.write(testo + "\n")


if __name__ == "__main__":
    main()
