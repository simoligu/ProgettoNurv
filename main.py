from pipeline import AnomalyDetectionPipeline
import torch
import os
from ultralytics import YOLO
import argparse

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
    parser.add_argument('--pole_tilt_weights', type=str, default=None,
                        help="Percorso al checkpoint .pt del regressore CNN per l'inclinazione "
                             "pali (vedi train_pole_tilt.py). Se omesso, gli alert PALO_INCLINATO "
                             "restano basati solo sulla stima geometrica DeepLab (minAreaRect).")
    parser.add_argument('--pole_tilt_angolo_max', type=float, default=22.0,
                        help="Deve coincidere con ANGOLO_MAX usato in train_pole_tilt.py (default: 22.0)")
    # --- NUOVO: mappa di sincronizzazione reference/query ---
    parser.add_argument('--sync_map', type=str, default=None,
                        help="Percorso al CSV prodotto da sync_videos_dtw.py. Se omesso, il modulo "
                             "di background-subtraction usa l'unico background mediato fisso "
                             "(comportamento originale). Se fornito, confronta ogni frame query "
                             "col frame reference corretto e salta il modulo sui frame senza "
                             "corrispondenza valida.")
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
    args = parser.parse_args()

    reference_video = 'data/videos/reference.mp4'
    query_video = 'data/videos/query.mp4'
    out_dir = 'out'

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
    )

    print("--- Avvio Pipeline di Rilevamento Anomalie ---")
    pipeline.run()
    print("--- Pipeline Terminata ---")