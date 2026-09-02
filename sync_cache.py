# -*- coding: utf-8 -*-
"""
Gestisce la cache della mappa di sincronizzazione: evita di rilanciare
sync_videos_dtw.py (operazione lenta, minuti su video lunghi) se esiste
gia' una mappa valida generata dagli stessi identici video reference/query.

Il fingerprint usato e' leggero (path assoluto + dimensione file + data di
ultima modifica), non un hash del contenuto — sufficiente per il caso
d'uso reale (accorgersi di essersi dimenticati di rigenerare la mappa dopo
aver sostituito un video), non pensato per essere a prova di manomissione.
Se sposti/rinomini un video senza modificarne il contenuto, il path cambia
e la cache verra' considerata invalida anche se il contenuto e' identico:
scelta consapevole, preferendo un falso "serve rigenerare" occasionale a
un falso "va bene cosi'" che userebbe una mappa sbagliata.
"""

import json
import os
from pathlib import Path
from typing import Optional


def _fingerprint_video(percorso_video: str) -> dict:
    p = Path(percorso_video).resolve()
    stat = p.stat()
    return {
        "path": str(p),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),  # troncato al secondo, sufficiente qui
    }


def _percorso_meta(percorso_mappa: str) -> str:
    return percorso_mappa + ".meta.json"


def mappa_e_valida(percorso_mappa: str, reference_video: str, query_video: str) -> bool:
    """
    True se esiste gia' una mappa di sincronizzazione con un file di
    metadati a fianco (<mappa>.meta.json) che combacia ESATTAMENTE con il
    fingerprint dei due video passati ora. False in ogni altro caso
    (mappa assente, metadati assenti/corrotti, o fingerprint diverso) —
    in caso di dubbio si preferisce rigenerare piuttosto che rischiare di
    riusare una mappa non piu' valida.
    """
    if not os.path.isfile(percorso_mappa):
        return False

    percorso_meta = _percorso_meta(percorso_mappa)
    if not os.path.isfile(percorso_meta):
        print(f"[sync_cache] Mappa presente ma senza metadati ({percorso_meta} mancante) "
              f"— probabilmente generata prima dell'introduzione della cache. Rigenero.")
        return False

    try:
        with open(percorso_meta, "r", encoding="utf-8") as f:
            meta_salvati = json.load(f)
        meta_attuali = {
            "reference": _fingerprint_video(reference_video),
            "query": _fingerprint_video(query_video),
        }
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"[sync_cache] Impossibile leggere/confrontare i metadati ({e}) — rigenero.")
        return False

    if meta_salvati == meta_attuali:
        print(f"[sync_cache] Mappa esistente ({percorso_mappa}) generata dagli stessi identici "
              f"video (reference/query invariati per path, dimensione e data) — la riuso.")
        return True

    print(f"[sync_cache] La mappa esistente ({percorso_mappa}) risulta generata da video "
          f"diversi da quelli attuali (path/dimensione/data non combaciano) — rigenero.")
    return False


def salva_metadati_mappa(percorso_mappa: str, reference_video: str, query_video: str) -> None:
    """Da chiamare subito dopo aver (ri)generato la mappa, per registrare da quali
    video e' stata prodotta."""
    meta = {
        "reference": _fingerprint_video(reference_video),
        "query": _fingerprint_video(query_video),
    }
    with open(_percorso_meta(percorso_mappa), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
