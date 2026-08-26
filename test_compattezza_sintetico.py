# -*- coding: utf-8 -*-
"""
Validazione sintetica del filtro di COMPATTEZZA (min_compattezza), aggiunto a
mask_to_boxes per distinguere forma-rumore (linee sottili da disallineamento
omografico lungo i contorni) da forma-anomalia (blob compatti/tondeggianti).

Testa DUE famiglie di forme sintetiche, alla stessa area nominale:
  - QUADRATI (rappresentano anomalie vere: frana, cedimento, accumulo — forma
    compatta)
  - RETTANGOLI ALLUNGATI a vari rapporti d'aspetto (rappresentano il rumore da
    disallineamento: una linea di differenza lungo un contorno lungo e sottile)

Obiettivo: verificare se, aggiungendo il filtro di compattezza, si puo'
ABBASSARE min_area rispetto alla calibrazione precedente (150/60000) senza
riaprire la porta al rumore — cioe' riguadagnare sensibilita' alle anomalie
piccole ma compatte, continuando a scartare il rumore allungato anche se
supera l'area minima piu' bassa.

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python test_compattezza_sintetico.py
"""

import numpy as np
import cv2
from background import mask_to_boxes

H, W = 1080, 1920
AREA_FRAME = H * W
DIFF_THRESH = 150  # gia' validato in precedenza, non ri-testato qui
INTENSITA_FORMA = 200  # ben sopra diff_thresh, isola la domanda "forma" da quella "intensita'"


def costruisci_quadrato(area_nominale: int) -> np.ndarray:
    diff = np.zeros((H, W), dtype=np.uint8)
    lato = int(round(area_nominale ** 0.5))
    cy, cx = H // 2, W // 2
    y0, y1 = cy - lato // 2, cy + lato // 2
    x0, x1 = cx - lato // 2, cx + lato // 2
    diff[y0:y1, x0:x1] = INTENSITA_FORMA
    return diff, lato * lato


def costruisci_linea_sottile(area_nominale: int, rapporto_aspetto: int) -> np.ndarray:
    """
    Simula il rumore REALE da disallineamento omografico: una linea sottile
    (spessore fisso) lungo un contorno — NON un rettangolo pieno (che avrebbe
    sempre compattezza 1.0 per costruzione, indipendentemente da quanto e'
    allungato, dato che riempie esattamente il proprio bounding box).

    rapporto_aspetto qui controlla lo spessore della linea rispetto alla sua
    lunghezza (piu' alto = piu' sottile), a parita' di area totale.
    """
    diff = np.zeros((H, W), dtype=np.uint8)
    spessore = max(2, int(round((area_nominale / rapporto_aspetto) ** 0.5)))
    lunghezza = max(10, area_nominale // spessore)
    cy, cx = H // 2, W // 2
    # linea leggermente diagonale (come un bordo di edificio/binario non
    # perfettamente orizzontale/verticale nell'inquadratura)
    dx, dy = lunghezza // 2, lunghezza // 8
    p1 = (max(0, cx - dx), max(0, cy - dy))
    p2 = (min(W - 1, cx + dx), min(H - 1, cy + dy))
    cv2.line(diff, p1, p2, INTENSITA_FORMA, spessore)
    return diff, None


def rilevato(diff: np.ndarray, min_area: int, min_compattezza) -> bool:
    boxes, _ = mask_to_boxes(diff, diff_thresh=DIFF_THRESH, min_area=min_area,
                             min_compattezza=min_compattezza)
    if not boxes:
        return False
    cy, cx = H // 2, W // 2
    for (x, y, w, h, area) in boxes:
        if x <= cx <= x + w and y <= cy <= y + h:
            return True
    return False


def main():
    print("=" * 100)
    print(" VALIDAZIONE SINTETICA — filtro di compattezza (forma) per ANOMALIA_STRUTTURALE")
    print("=" * 100)

    aree_test = [15000, 30000, 45000, 60000, 90000]
    rapporti_aspetto = [3, 6, 10, 20]  # quanto e' "allungato" il rumore simulato

    configurazioni = [
        {"nome": "SOLO area (min_area=60000, come calibrazione precedente)",
         "min_area": 60000, "min_compattezza": None},
        {"nome": "area PIU' BASSA + compattezza (min_area=15000, min_compattezza=0.5)",
         "min_area": 15000, "min_compattezza": 0.5},
        {"nome": "area PIU' BASSA + compattezza piu' permissiva (min_area=15000, min_compattezza=0.3)",
         "min_area": 15000, "min_compattezza": 0.3},
    ]

    for config in configurazioni:
        print(f"\n--- {config['nome']} ---")

        print("QUADRATI (anomalia-tipo, atteso: rilevati anche piccoli):")
        header = f"{'area_nom':>10}" + "".join(f"{'lato' + str(i):>10}" for i in [""])
        riga_intestazione = f"{'area':>10}{'lato(px)':>10}{'rilevato':>10}"
        print(riga_intestazione)
        for area in aree_test:
            diff, area_reale = costruisci_quadrato(area)
            lato = int(round(area ** 0.5))
            ril = rilevato(diff, config["min_area"], config["min_compattezza"])
            print(f"{area:>10}{lato:>10}{('SI' if ril else '--'):>10}")

        print("\nLINEE SOTTILI (rumore-tipo, stessa area nominale, atteso: MAI rilevate):")
        header2 = f"{'area':>10}" + "".join(f"{'rapp.'+str(r):>10}" for r in rapporti_aspetto)
        print(header2)
        for area in aree_test:
            riga = f"{area:>10}"
            for rapporto in rapporti_aspetto:
                diff, _ = costruisci_linea_sottile(area, rapporto)
                ril = rilevato(diff, config["min_area"], config["min_compattezza"])
                riga += f"{('SI' if ril else '--'):>10}"
            print(riga)

    print("\n" + "=" * 100)
    print("Come leggere: nella tabella QUADRATI, 'SI' e' un successo (anomalia compatta")
    print("rilevata). Nella tabella LINEE SOTTILI, 'SI' e' un FALLIMENTO (rumore allungato")
    print("rilevato per errore, esattamente cio' che il filtro di compattezza dovrebbe evitare).")
    print()
    print("Confronta le tre configurazioni: se 'area piu' bassa + compattezza' rileva piu'")
    print("quadrati piccoli della configurazione storica, MA continua a scartare tutti i")
    print("rettangoli allungati, il filtro di forma funziona come sperato — permette di")
    print("riguadagnare sensibilita' senza riaprire la porta al rumore.")


if __name__ == "__main__":
    main()
