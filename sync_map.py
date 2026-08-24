# -*- coding: utf-8 -*-
"""
MappaSincronizzazione — carica e interroga il CSV prodotto da
sync_videos_dtw.py, dando per ogni frame RAW del query (non campionato) il
frame RAW corrispondente del reference, con interpolazione lineare tra i
campioni piu' vicini (il CSV contiene una corrispondenza ogni --step frame,
non per ogni singolo frame).

Ritorna None per un frame query se:
  - cade PRIMA del primo campione o DOPO l'ultimo (fuori dal range coperto
    dalla mappa)
  - cade tra due campioni di cui almeno uno e' stato scartato come privo di
    vera corrispondenza (frame_reference vuoto nel CSV) — in quel caso non
    si interpola tra un valore valido e uno assente, si segnala "nessuna
    corrispondenza" per prudenza
"""

import bisect
import csv
from typing import List, Optional, Tuple


class MappaSincronizzazione:
    def __init__(self, percorso_csv: str):
        chiavi: List[int] = []
        valori: List[Optional[int]] = []

        with open(percorso_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for riga in reader:
                fq = int(riga["frame_query"])
                fr_str = riga["frame_reference"]
                fr = int(fr_str) if fr_str.strip() != "" else None
                chiavi.append(fq)
                valori.append(fr)

        if not chiavi:
            raise ValueError(f"Mappa di sincronizzazione vuota: {percorso_csv}")

        # ordina per sicurezza (dovrebbero gia' esserlo, ma non si sa mai)
        ordinati = sorted(zip(chiavi, valori), key=lambda t: t[0])
        self._chiavi = [c for c, _ in ordinati]
        self._valori = [v for _, v in ordinati]

        n_valide = sum(1 for v in self._valori if v is not None)
        print(f"[MappaSincronizzazione] Caricata da {percorso_csv}: "
              f"{len(self._chiavi)} campioni ({n_valide} con corrispondenza valida)")

    def frame_reference_per(self, frame_query_raw: int) -> Optional[int]:
        """
        Ritorna il frame reference corrispondente a un dato frame query
        (indice raw, non campionato), interpolando linearmente tra i due
        campioni piu' vicini. None se fuori range o se la corrispondenza
        piu' vicina e' stata scartata come non valida.
        """
        i = bisect.bisect_left(self._chiavi, frame_query_raw)

        # fuori range prima del primo campione o dopo l'ultimo
        if i == 0 and frame_query_raw < self._chiavi[0]:
            return None
        if i >= len(self._chiavi):
            return None

        # corrispondenza esatta su un campione
        if self._chiavi[i] == frame_query_raw:
            return self._valori[i]

        # interpolazione tra il campione precedente (i-1) e quello attuale (i)
        if i == 0:
            return None  # non dovrebbe succedere dato il controllo sopra, difensivo
        q0, q1 = self._chiavi[i - 1], self._chiavi[i]
        r0, r1 = self._valori[i - 1], self._valori[i]

        if r0 is None or r1 is None:
            return None  # almeno un estremo privo di corrispondenza: non interpolare

        frazione = (frame_query_raw - q0) / (q1 - q0)
        return round(r0 + frazione * (r1 - r0))