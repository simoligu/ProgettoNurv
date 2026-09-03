from pipeline import AnomalyDetectionPipeline
import torch
import os
from ultralytics import YOLO
import argparse

import sync_videos_dtw
from sync_cache import mappa_e_valida, salva_metadati_mappa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline NURV di rilevamento anomalie")
    parser.add_argument('--tratta', type=int, required=True,
                        help="ID della tratta sul sito a cui appartengono gli alert")
    parser.add_argument('--deeplab', type=str, default='runs_seg/deeplab_hires/best.pt',
                        help="Percorso ai pesi DeepLabV3+ (default: runs_seg/deeplab_hires/best.pt)")
    parser.add_argument('--imgsz', type=int, default=896,
                        help="Risoluzione di inferenza DeepLab (default: 896)")
    parser.add_argument('--seg_step', type=int, default=30,
                        help="Analisi strutturale ogni N frame (default: 30)")
    parser.add_argument('--gauge_tolerance', type=float, default=0.15,
                        help="Tolleranza percentuale sullo scartamento (default: 0.15 = 15%%). "
                             "Abbassala (es. 0.05) per un test di sanita': verifica che il "
                             "meccanismo di alert torni sensibile con soglia piu' stretta.")
    # --- NUOVO: raffinamento CNN dell'inclinazione dei pali ---
    parser.add_argument('--pole_tilt_weights', type=str, default='weights/pole_tilt_best.pt',
                        help="Percorso al checkpoint .pt del regressore CNN per l'inclinazione "
                             "pali (vedi train_pole_tilt.py). Default: weights/pole_tilt_best.pt "
                             "(coerente con --deeplab). Passa --pole_tilt_weights '' (stringa "
                             "vuota) per disattivarlo esplicitamente e tornare alla sola stima "
                             "geometrica DeepLab (minAreaRect) — utile se stai lavorando da una "
                             "macchina priva dei pesi (es. lab senza accesso al desktop di casa).")
    parser.add_argument('--pole_tilt_angolo_max', type=float, default=22.0,
                        help="Deve coincidere con ANGOLO_MAX usato in train_pole_tilt.py (default: 22.0)")
    # --- NUOVO: mappa di sincronizzazione reference/query ---
    parser.add_argument('--sync_map', type=str, default=None,
                        help="Percorso alla mappa di sincronizzazione (CSV). Se omesso, il modulo "
                             "di background-subtraction usa l'unico background mediato fisso "
                             "(comportamento originale, nessuna sincronizzazione). Se fornito: "
                             "se il file esiste gia' ED e' stato generato dagli stessi identici "
                             "video reference/query correnti (vedi sync_cache.py), viene riusato "
                             "cosi' com'e'; altrimenti viene generato automaticamente chiamando "
                             "sync_videos_dtw.py con i parametri calibrati di default, senza "
                             "bisogno di lanciarlo a parte. Poi confronta ogni frame query col "
                             "frame reference corretto e salta il modulo sui frame senza "
                             "corrispondenza valida.")
    parser.add_argument('--forza_risincronizzazione', action='store_true',
                        help="Rigenera sempre la mappa (ignora la cache basata sui metadati "
                             "dei video), anche se --sync_map punta a un file gia' valido per "
                             "i video correnti. Utile se hai cambiato i parametri di sync a mano "
                             "e vuoi essere sicuro che vengano riapplicati.")
    # --- NUOVO: soglie del modulo di background-subtraction (De Paolis) ---
    parser.add_argument('--diff_thresh', type=int, default=150,
                        help="Soglia di differenza pixel per il modulo ANOMALIA_STRUTTURALE "
                             "(default storico: 90). Il default qui e' 150, dalla calibrazione "
                             "empirica su questo progetto (vedi calibra_soglie_change_detector.py) "
                             "— verifica sempre con test_change_detector_sintetico.py o simile "
                             "prima di modificare ulteriormente.")
    parser.add_argument('--min_area', type=int, default=60000,
                        help="Area minima (in pixel) di un blob di differenza per generare un "
                             "alert ANOMALIA_STRUTTURALE (default storico: 8000). Il default qui "
                             "e' 60000, dalla stessa calibrazione empirica.")
    parser.add_argument('--min_compattezza', type=float, default=None,
                        help="Filtro di FORMA aggiuntivo (0-1): scarta i blob la cui area e' "
                             "meno di questa frazione della loro bounding box — un rumore da "
                             "disallineamento tende a formare linee sottili e allungate lungo i "
                             "contorni (compattezza bassa), un'anomalia vera tende a essere piu' "
                             "compatta. Default None (disattivato). Se attivato, valuta di "
                             "abbassare anche --min_area di conseguenza (vedi calibrazione).")
    # --- NUOVO: filtro di persistenza spaziale/temporale (TemporalTracker) ---
    parser.add_argument('--temporal_tracker', action='store_true',
                        help="Attiva il filtro di persistenza per ANOMALIA_STRUTTURALE: un "
                             "candidato entra nel CSV/video/alert solo se ricompare in "
                             "posizione simile per almeno --temporal_min_occorrenze controlli "
                             "consecutivi. Default disattivato (comportamento invariato).")
    parser.add_argument('--temporal_iou_threshold', type=float, default=0.25,
                        help="Sovrapposizione minima (IoU) tra bbox consecutivi per "
                             "considerarli la stessa anomalia (default: 0.25)")
    parser.add_argument('--temporal_min_occorrenze', type=int, default=2,
                        help="Quante volte l'anomalia deve ricomparire prima di essere "
                             "confermata (default: 2)")
    parser.add_argument('--temporal_finestra_frame', type=int, default=450,
                        help="Quanti frame indietro si cerca ancora una corrispondenza prima "
                             "che la traccia scada (default: 450)")
    args = parser.parse_args()

    reference_video = 'data/videos/reference.mp4'
    query_video = 'data/videos/query.mp4'
    out_dir = 'out'

    # --- Auto-sync: genera/riusa la mappa solo se --sync_map e' stato passato
    # (comportamento opt-in, invariato per chi non lo usa) ---
    if args.sync_map is not None:
        serve_rigenerare = args.forza_risincronizzazione or not mappa_e_valida(
            args.sync_map, reference_video, query_video)

        if serve_rigenerare:
            print(f"[main] Genero la mappa di sincronizzazione in: {args.sync_map} "
                  f"(reference={reference_video}, query={query_video})")
            # Parametri calibrati empiricamente su questo progetto (vedi diario) —
            # non esposti come flag di main.py per non duplicare la superficie CLI
            # di sync_videos_dtw.py; usa quello script direttamente se ti serve
            # toccarli per un caso particolare.
            sync_videos_dtw.main([
                "--reference", reference_video,
                "--query", query_video,
                "--output", args.sync_map,
                "--step", "5",
                "--fattore-margine-raffinamento", "4.0",
                "--min-run-plateau-interno", "50",
            ])
            salva_metadati_mappa(args.sync_map, reference_video, query_video)

    USE_CLASSIFIER = True

    # Selezione del device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    classifier = None
    if USE_CLASSIFIER:
        try:
            classifier = YOLO('yolov8n.pt')
            print(f"[INFO] Modello YOLOv8 caricato correttamente.")
        except Exception as e:
            print(f"[ERROR] Errore durante l'inizializzazione di YOLO: {e}")
            USE_CLASSIFIER = False

    ALERT_ENDPOINT = "http://localhost:8080/api/alerts"

    pipeline = AnomalyDetectionPipeline(
        reference_video=reference_video,
        query_video=query_video,
        out_dir=out_dir,
        sample_step=8,
        use_classifier=USE_CLASSIFIER,
        classifier=classifier,
        alert_endpoint=ALERT_ENDPOINT,
        tratta_id=args.tratta,
        # --- DeepLab ---
        deeplab_weights=args.deeplab,
        deeplab_imgsz=args.imgsz,
        seg_step=args.seg_step,
        gauge_tolerance=args.gauge_tolerance,
        # --- PoleTilt (opzionale: None se --pole_tilt_weights non passato) ---
        pole_tilt_weights=args.pole_tilt_weights,
        pole_tilt_angolo_max=args.pole_tilt_angolo_max,
        # --- Mappa di sincronizzazione (opzionale: None se --sync_map non passato) ---
        sync_map_csv=args.sync_map,
        # --- Soglie background-subtraction (calibrate empiricamente, vedi --help) ---
        diff_thresh=args.diff_thresh,
        min_area=args.min_area,
        min_compattezza=args.min_compattezza,
        # --- TemporalTracker (opzionale: disattivo se --temporal_tracker non passato) ---
        usa_temporal_tracker=args.temporal_tracker,
        temporal_iou_threshold=args.temporal_iou_threshold,
        temporal_min_occorrenze=args.temporal_min_occorrenze,
        temporal_finestra_frame=args.temporal_finestra_frame,
    )

    print("--- Avvio Pipeline di Rilevamento Anomalie ---")
    pipeline.run()
    print("--- Pipeline Terminata ---")