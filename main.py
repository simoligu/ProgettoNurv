from pipeline import AnomalyDetectionPipeline
from classifier import SimpleClassifier
import torch
import os
from ultralytics import YOLO
import requests
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline NURV di rilevamento anomalie")
    parser.add_argument('--tratta', type=int, required=True, help="ID della tratta sul sito a cui appartengono gli alert")
    args = parser.parse_args()

    reference_video = 'data/videos/reference.mp4'
    query_video = 'data/videos/query.mp4'
    out_dir = 'out'

    # Imposta a True solo dopo aver addestrato il tuo classifier.pth
    USE_CLASSIFIER = True

    # Selezione del device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    classifier = None
    if USE_CLASSIFIER:
        try:
            # Carichiamo il modello YOLOv8 ufficiale
            # Scaricherà automaticamente il file .pt al primo avvio
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
        tratta_id=args.tratta
    )

    print("--- Avvio Pipeline di Rilevamento Anomalie ---")
    pipeline.run()
    print("--- Pipeline Terminata ---")