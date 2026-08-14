# -*- coding: utf-8 -*-
"""
Test isolato: verifica che l'endpoint /api/alerts riceva, decodifichi e salvi
correttamente un'immagine allegata, senza dover lanciare l'intera pipeline.

Lancio (con il server Java gia' avviato su localhost:8080):
    python test_alert_con_immagine.py --image data/rs19_val/jpgs/rs19_val/rs00001.jpg --tratta 1
"""

import argparse
import base64
import requests

URL = "http://localhost:8080/api/alerts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Percorso a un'immagine JPEG qualsiasi")
    ap.add_argument("--tratta", type=int, required=True, help="ID di una tratta esistente nel DB")
    ap.add_argument("--severity", default="CRITICA", help="Severita' (CRITICA per testare anche Telegram)")
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    print(f"[INFO] Immagine codificata: {len(img_b64)} caratteri base64 "
          f"({len(img_b64) * 3 // 4} byte circa)")

    alert = {
        "project": "ProgettoNurv",
        "frame_idx": 1,
        "time_s": 0.0,
        "bbox": {"x": 10, "y": 10, "w": 100, "h": 100},
        "area": 10000,
        "label": "VEGETAZIONE_INVASIVA",
        "conf": 0.9,
        "severity": args.severity,
        "details": "Test invio immagine — script isolato",
        "source_video": "test.mp4",
        "trattaId": args.tratta,
        "frame_b64": img_b64,
    }

    print(f"[INFO] Invio a {URL}...")
    try:
        r = requests.post(URL, json=alert, timeout=10)
        print(f"[RISULTATO] Status: {r.status_code}")
        print(f"[RISULTATO] Risposta: {r.text}")
        if r.status_code == 201:
            print("\n✅ SUCCESSO — l'alert con immagine e' stato accettato dal server.")
            print("   Prossimo passo: controlla il DB (colonna frame_image sulla riga")
            print("   appena creata) e prova ad aprire /anomalia/{id}/frame nel browser.")
        else:
            print("\n❌ Il server ha risposto con un errore. Controlla i log Java.")
    except requests.exceptions.ConnectionError:
        print("\n❌ Impossibile connettersi al server. E' avviato su localhost:8080?")
    except Exception as e:
        print(f"\n❌ Errore: {e}")


if __name__ == "__main__":
    main()
