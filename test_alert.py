import requests
import json

# === URL del sito Java ===
ALERT_ENDPOINT = "http://localhost:8080/api/alerts"

# === Dati di test simulati ===
alert = {
    "project": "ProgettoNurv",
    "frame_idx": 123,
    "time_s": 12.34,
    "bbox": {"x": 50, "y": 60, "w": 120, "h": 80},
    "area": 9600,
    "label": "Dilatazione Ferrovia",
    "conf": 0.95,
    "severity": "CRITICA",
    "details": "Simulazione: Rilevata anomalia strutturale (Dilatazione)",
    "source_video": "query.mp4"
}

print(f"Invio alert di test a {ALERT_ENDPOINT}...")
try:
    response = requests.post(ALERT_ENDPOINT, json=alert, timeout=5)
    print(f"✅ Status code: {response.status_code}")
    print("Risposta server:", response.text)
except requests.exceptions.RequestException as e:
    print(f"❌ Errore di connessione: Assicurati che il server Java sia in esecuzione su {ALERT_ENDPOINT}. Errore: {e}")