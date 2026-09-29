# -*- coding: utf-8 -*-
"""
Corregge il segno di 'angolo_gradi' nelle etichette del dataset asset2.

PERCHE' SERVE:
La camera nella zona dell'asset2 guarda in direzione quasi opposta lungo il
binario rispetto al dataset originale (lo stesso motivo per cui PUNTO_BINARIO_A/B
sono stati scambiati in genera_dataset_pali_asset2.py per correggere lo
spostamento della camera). Quello scambio risolve il movimento, ma ha un
effetto collaterale sul SIGNIFICATO VISIVO del segno dell'angolo: la stessa
rotazione applicata in Unreal (stesso valore numerico "pitch") appare come
inclinazione verso sinistra in un dataset e verso destra nell'altro, perche'
la camera guarda il binario da capo opposto.

Prova empirica (dalla diagnostica del training): confrontando angolo reale e
predetto sul test set, l'ENTITA' e' quasi sempre corretta ma il SEGNO e'
concorde solo nell'8% dei casi (contro il 50% atteso se fosse casuale) — la
firma di un'inversione sistematica, non di un modello che non generalizza.

COSA FA QUESTO SCRIPT:
Nega il campo "angolo_gradi" in etichette.json per il dataset asset2 (sia la
versione raw sia quella gia' ritagliata da crop_dataset.py), con backup del
file originale. NON tocca le immagini ne' i nomi dei file (che restano con
l'angolo "vecchio" nel nome — mismatch solo cosmetico, il training legge il
valore dal JSON, non dal nome del file).

USO:
    python fix_segno_asset2.py
"""

import json
import os
import shutil

CARTELLE_DA_CORREGGERE = [
    r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_asset2_test",
    r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_asset2_test_crop",
]


def correggi_cartella(cartella: str):
    percorso_json = os.path.join(cartella, "etichette.json")
    if not os.path.exists(percorso_json):
        print(f"[SKIP] Non trovato: {percorso_json}")
        return

    percorso_backup = percorso_json + ".backup_prima_fix_segno"
    if os.path.exists(percorso_backup):
        print(f"[ATTENZIONE] Backup gia' esistente ({percorso_backup}) — "
              f"lo script e' gia' stato eseguito su questa cartella? Salto per sicurezza, "
              f"cancella il backup a mano se vuoi rieseguire.")
        return

    shutil.copy2(percorso_json, percorso_backup)
    print(f"Backup salvato: {percorso_backup}")

    with open(percorso_json, "r", encoding="utf-8") as f:
        etichette = json.load(f)

    for rec in etichette:
        rec["angolo_gradi"] = -rec["angolo_gradi"]

    with open(percorso_json, "w", encoding="utf-8") as f:
        json.dump(etichette, f, indent=2)

    print(f"Corrette {len(etichette)} etichette in: {percorso_json}\n")


if __name__ == "__main__":
    for cartella in CARTELLE_DA_CORREGGERE:
        correggi_cartella(cartella)
    print("Fatto. Rilancia la valutazione (--eval-only) o il training per verificare il fix.")
