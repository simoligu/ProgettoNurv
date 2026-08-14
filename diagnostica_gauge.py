# -*- coding: utf-8 -*-
"""
Diagnostica GAUGE (scartamento) — capisce perche' SCARTAMENTO_ANOMALO non scatta
mai, nemmeno con tolleranza stretta (5%).

Rilancia la stessa logica di _analyze_gauge() ma con stampa verbosa per ogni
frame analizzato: quante righe hanno dato una misura valida, la dispersione
(coefficiente di variazione), se il filtro di robustezza scarta la misura, e
la deviazione dal baseline. Permette di capire SE il problema e':
  (a) il filtro di robustezza (CV > 20%) scarta quasi tutte le misure -> non
      arriviamo mai al controllo della tolleranza
  (b) lo scartamento misurato e' semplicemente sempre vicino al baseline
      (nessuna vera anomalia nel video, comportamento corretto)
  (c) le righe valide sono troppo poche (<5) per considerare la misura affidabile

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python diagnostica_gauge.py --weights runs_seg/deeplab_hires/best.pt --video data/videos/query.mp4
    python diagnostica_gauge.py --weights runs_seg/deeplab_hires/best.pt --video data/videos/query.mp4 --seg_step 150 --n_test 15
"""

import argparse
import numpy as np
import cv2
import torch

from deeplab_analyzer import DeepLabAnalyzer, CL_ROTAIE
from video_io import VideoIO


def analizza_riga_per_riga(analyzer, class_map, h, w):
    """Replica la scansione di _analyze_gauge ma restituisce TUTTI i dettagli
    intermedi invece del solo risultato finale, per la diagnostica. Usa gli
    stessi parametri configurabili dell'analyzer, cosi' la diagnostica
    riflette sempre il comportamento reale di produzione."""
    rail_mask = (class_map == CL_ROTAIE).astype(np.uint8)
    y_start = int(h * analyzer.gauge_roi_top_frac)
    y_end = int(h * analyzer.gauge_roi_bottom_frac)
    rows_to_check = np.linspace(y_start, y_end, num=20, dtype=int)

    distances = []
    for y in rows_to_check:
        row = rail_mask[y, :]
        runs = DeepLabAnalyzer._find_runs(row)
        if len(runs) >= 2:
            runs_sorted = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)
            r1, r2 = sorted(runs_sorted[:2], key=lambda r: r[0])
            gap = r2[0] - r1[1]
            if gap > 10:
                distances.append(gap)

    return distances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--ref", default="data/videos/reference.mp4",
                    help="Video di reference, usato per l'auto-calibrazione dello "
                         "scartamento — SENZA questo passaggio il confronto usa il "
                         "default hardcoded (150px) invece del baseline vero, "
                         "producendo deviazioni fuorvianti.")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--seg_step", type=int, default=150)
    ap.add_argument("--n_test", type=int, default=15, help="Numero di frame campionati da testare")
    ap.add_argument("--sample_step", type=int, default=8,
                    help="Passo di campionamento del reference, deve coincidere con "
                         "quello usato da main.py (default 8, come in main.py)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Carico modello ({device})...")
    analyzer = DeepLabAnalyzer(weights_path=args.weights, imgsz=args.imgsz)

    # --- AUTO-CALIBRAZIONE (replica ESATTAMENTE cosa fa pipeline.run()) ---
    # IMPORTANTE: ref_anchor NON e' il centro dell'intero reference video, ma
    # il centro dei primi 200 frame campionati ogni 8 (VideoIO.sample_frames),
    # quindi copre solo i primi ~53 secondi del reference indipendentemente
    # da quanto e' lungo il video intero. Usare naivamente total_frames//2
    # (come nella versione precedente di questo script) sceglie un frame
    # DIVERSO se il reference e' piu' lungo di ~53s, calibrando su un
    # baseline diverso da quello che la pipeline reale userebbe davvero.
    print(f"[INFO] Campionamento reference (step={args.sample_step}, max 200 frame, "
          f"come in pipeline.py)...")
    ref_frames_tuple = VideoIO.sample_frames(args.ref, max_frames=200, step=args.sample_step)
    ref_frame = ref_frames_tuple[len(ref_frames_tuple) // 2][1]
    ref_idx_originale = ref_frames_tuple[len(ref_frames_tuple) // 2][0]
    print(f"[INFO] ref_anchor = frame originale #{ref_idx_originale} del reference "
          f"(su {len(ref_frames_tuple)} campionati)")

    print(f"[INFO] Calibrazione dal reference...")
    ref_class_map = analyzer.segment(ref_frame)
    baseline = analyzer.measure_gauge(ref_class_map)
    if baseline is not None:
        analyzer.expected_gauge_px = baseline
        print(f"[INFO] Baseline calibrato: {baseline:.0f}px (sostituisce il default {150}px)")
    else:
        print(f"[WARN] Calibrazione fallita (rotaie non rilevate nel reference), "
              f"resto sul default {analyzer.expected_gauge_px}px — i risultati sotto "
              f"potrebbero essere fuorvianti.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # campiona ai multipli di seg_step, come farebbe la pipeline reale
    frame_indices = [i * args.seg_step for i in range(1, args.n_test + 1) if i * args.seg_step < total_frames]

    print(f"[INFO] Baseline attuale (expected_gauge_px): {analyzer.expected_gauge_px:.0f}px")
    print(f"[INFO] Fascia di scansione: {analyzer.gauge_roi_top_frac*100:.0f}%-{analyzer.gauge_roi_bottom_frac*100:.0f}% dell'altezza")
    print(f"[INFO] Soglia CV robustezza: {analyzer.gauge_cv_threshold*100:.0f}%")
    print(f"[INFO] Testo {len(frame_indices)} frame ai multipli di seg_step={args.seg_step}\n")

    print(f"{'Frame':>7}{'Righe valide':>14}{'Mediana(px)':>13}{'CV%':>8}{'Filtro CV':>11}{'Deviazione':>12}{'Alert 5%':>10}{'Alert 15%':>11}")
    print("-" * 90)

    n_filtrate_da_cv = 0
    n_poche_righe = 0
    n_misure_valide = 0
    n_alert_5 = 0
    n_alert_15 = 0

    for fidx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (w, h))  # dimensioni invariate, coerenza con pipeline
        class_map = analyzer.segment(frame_resized)

        distances = analizza_riga_per_riga(analyzer, class_map, h, w)

        if len(distances) < 5:
            n_poche_righe += 1
            print(f"{fidx:>7}{len(distances):>14}{'—':>13}{'—':>8}{'POCHE RIGHE':>11}{'—':>12}{'—':>10}{'—':>11}")
            continue

        gauge = float(np.median(distances))
        std_dev = float(np.std(distances))
        cv_pct = 100 * std_dev / gauge if gauge > 0 else 0

        cv_filtra = cv_pct > (analyzer.gauge_cv_threshold * 100)
        if cv_filtra:
            n_filtrate_da_cv += 1

        deviation = abs(gauge - analyzer.expected_gauge_px)
        dev_pct = 100 * deviation / analyzer.expected_gauge_px

        alert_5 = (not cv_filtra) and (dev_pct > 5.0)
        alert_15 = (not cv_filtra) and (dev_pct > 15.0)

        if not cv_filtra:
            n_misure_valide += 1
            if alert_5:
                n_alert_5 += 1
            if alert_15:
                n_alert_15 += 1

        filtro_str = "SCARTATA" if cv_filtra else "ok"
        print(f"{fidx:>7}{len(distances):>14}{gauge:>13.0f}{cv_pct:>8.1f}{filtro_str:>11}"
              f"{dev_pct:>11.1f}%{str(alert_5):>10}{str(alert_15):>11}")

    print("-" * 90)
    print(f"\nRIEPILOGO su {len(frame_indices)} frame testati:")
    print(f"  Scartati per poche righe valide (<5):              {n_poche_righe}")
    print(f"  Scartati dal filtro robustezza (CV>{analyzer.gauge_cv_threshold*100:.0f}%):        {n_filtrate_da_cv}")
    print(f"  Misure considerate valide/affidabili:              {n_misure_valide}")
    print(f"  Di cui avrebbero generato alert a tolleranza 5%:   {n_alert_5}")
    print(f"  Di cui avrebbero generato alert a tolleranza 15%:  {n_alert_15}")

    print("\nINTERPRETAZIONE:")
    if n_poche_righe > len(frame_indices) * 0.5:
        print("  -> La MAGGIOR PARTE dei frame non trova abbastanza righe con due")
        print("     rotaie rilevate (serve un minimo di 5 su 20 scansionate).")
        print("     Possibile causa: risoluzione/qualita' della segmentazione nella")
        print("     fascia di scansione scelta per questo video specifico.")
    elif n_misure_valide > 0 and (n_alert_5 > 0 or n_alert_15 > 0):
        print(f"  -> Ci sono {n_misure_valide} misure valide, e {n_alert_5} di esse")
        print(f"     avrebbero generato alert anche a tolleranza 15% ({n_alert_15}).")
        print("     Il meccanismo E' SENSIBILE: non e' 'spento'. Se main.py con gli")
        print("     stessi parametri non genera alert, il problema e' altrove (es.")
        print("     nel throttling degli alert dentro pipeline.run(), o i frame")
        print("     effettivamente campionati da main.py non coincidono con questi).")
    elif n_filtrate_da_cv > n_misure_valide:
        print(f"  -> Il filtro di robustezza (coefficiente di variazione > {analyzer.gauge_cv_threshold*100:.0f}%)")
        print("     scarta la MAGGIOR PARTE delle misure. Probabile causa:")
        print("     scambi/deviatori frequenti in questo video, che disturbano la")
        print("     scansione riga per riga quasi sempre, non solo occasionalmente.")
        print("     Il filtro e' probabilmente ancora troppo aggressivo per questo")
        print("     contesto specifico — valuta di alzarlo ulteriormente.")
    elif n_misure_valide > 0:
        print("  -> Ci sono misure valide e affidabili, ma la deviazione dal")
        print("     baseline resta sempre sotto le soglie testate. Questo significa")
        print("     che lo scartamento e' effettivamente STABILE in questo video:")
        print("     comportamento CORRETTO, non un problema del filtro.")
    else:
        print("  -> Nessuna misura valida in nessun frame testato. Da indagare")
        print("     ulteriormente con --n_test piu' alto o controllo visivo.")


if __name__ == "__main__":
    main()