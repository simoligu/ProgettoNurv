import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

print(f"Sto per usare il token: {TELEGRAM_TOKEN}")      #DEBUG



# URL del tuo controller Java
URL = "http://localhost:8080/api/alerts"

test_alert = {
    "project": "ProgettoNurv",
    "label": "PERSONA_SUI_BINARI",
    "severity": "CRITICA",      # Questo deve far apparire il badge ROSSO
    "details": "Test logica dinamica - Verifica Colori",
    "conf": 0.98,
    "frame_idx": 1,
    "time_s": 0.0,
    "source_video": "test.mp4"
}

print(f"Inviando test a {URL}...")
try:
    r = requests.post(URL, json=test_alert, timeout=5)
    print(f"Risposta server: {r.status_code}")
    if r.status_code == 201 or r.status_code == 200:
        print("✅ SUCCESSO! Vai sul sito e ricarica la pagina /ia/alerts")
    else:
        print(f"❌ Errore Java: {r.text}")
except Exception as e:
    print(f"❌ Errore di connessione: {e}")