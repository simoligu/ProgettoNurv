# -*- coding: utf-8 -*-
"""
Sincronizzazione temporale reference/query — stessa tratta, velocita' diverse.

PROBLEMA:
reference_video e query_video riprendono la stessa tratta ferroviaria, ma il
treno viaggia a velocita' diverse nei due video (accelerazioni/rallentamenti
non uniformi) — quindi "secondo X" nel query NON corrisponde alla stessa
scena fisica di "secondo X" nel reference. Serve una mappa frame-per-frame
di corrispondenza, non un semplice offset costante.

APPROCCIO — Dynamic Time Warping (DTW) su feature visive:
Per ogni frame campionato di entrambi i video si calcola una "impronta"
visiva (istogramma colore + densita' di bordi, scelta per essere ragionevol-
mente robusta a piccole differenze di luce/esposizione tra i due passaggi).
Si confrontano tutte le impronte query contro tutte le impronte reference
(matrice di costo), e si cerca il percorso di corrispondenza MONOTONO
(il treno va sempre avanti lungo la tratta, mai indietro) che minimizza la
differenza visiva cumulata lungo il percorso — questo e' esattamente cio'
che fa DTW, ed e' lo strumento giusto perche' gestisce naturalmente tratti
piu' lenti/veloci senza assumere un rapporto di velocita' costante.

OUTPUT: un CSV con la mappa frame_query -> frame_reference (piu' i tempi in
secondi di entrambi), da usare direttamente nella pipeline per confrontare
sempre i frame corretti tra loro — invece di un nuovo file video (nessuna
perdita di qualita' da re-encoding, nessun lavoro manuale di editing).

USO:
    python sync_videos_dtw.py --reference ref.mp4 --query query.mp4 \
        --output mappa_sync.csv --step 5

Il parametro --step campiona 1 frame ogni N (default 5): non serve
analizzare ogni singolo frame per la sincronizzazione, e ridurre il numero
di frame accelera parecchio sia il calcolo delle feature sia DTW.
"""

import argparse
import csv
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ==================== FEATURE PER FRAME ====================

def calcola_feature_frame(frame_bgr: np.ndarray, dimensione: int = 64) -> np.ndarray:
    """
    Impronta visiva compatta di un frame, pensata per il confronto DTW:
    - istogramma Hue+Saturation in HSV (32+32 bin): ragionevolmente robusto
      a differenze di luminosita'/esposizione tra i due passaggi (a
      differenza di un istogramma sui livelli di grigio, che ne risente di
      piu'), utile perche' reference e query potrebbero essere stati ripresi
      in condizioni di luce leggermente diverse
    - densita' di bordi (Sobel, 16 bin per riga orizzontale): cattura la
      "struttura" della scena (curve, oggetti, orizzonte), complementare al
      solo colore

    Il frame viene ridimensionato a un quadrato piccolo prima del calcolo,
    per velocita' — la sincronizzazione non richiede dettaglio pixel-level.
    """
    piccolo = cv2.resize(frame_bgr, (dimensione, dimensione), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(piccolo, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()

    gray = cv2.cvtColor(piccolo, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    hist_edge = np.histogram(np.abs(sobel), bins=16, range=(0, 255))[0].astype(np.float32)

    feature = np.concatenate([hist_h, hist_s, hist_edge]).astype(np.float32)
    norma = np.linalg.norm(feature)
    if norma > 1e-6:
        feature = feature / norma
    return feature


def estrai_feature_video(percorso_video: str, step: int,
                          max_frame: int = None) -> Tuple[List[np.ndarray], List[int], float]:
    """
    Campiona un frame ogni `step` e ne calcola la feature.
    Ritorna (lista_feature, lista_indici_frame_originali, fps).
    """
    cap = cv2.VideoCapture(percorso_video)
    if not cap.isOpened():
        raise IOError(f"Impossibile aprire il video: {percorso_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    totale_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frame is not None:
        totale_frame = min(totale_frame, max_frame)

    features = []
    indici = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frame is not None and idx >= max_frame):
            break
        if idx % step == 0:
            features.append(calcola_feature_frame(frame))
            indici.append(idx)
        idx += 1
    cap.release()
    return features, indici, fps


# ==================== DTW CON BANDA DI SAKOE-CHIBA ====================

def dtw_allineamento(feat_query: List[np.ndarray], feat_reference: List[np.ndarray],
                      banda_frazione: float = 0.15) -> List[Tuple[int, int]]:
    """
    Dynamic Time Warping tra due sequenze di feature, con vincolo di banda
    (Sakoe-Chiba): limita quanto il percorso puo' scostarsi dalla diagonale,
    per evitare corrispondenze assurde (es. l'inizio del query abbinato
    alla fine del reference) ed accelerare il calcolo su sequenze lunghe.
    banda_frazione=0.15 permette un rapporto di velocita' tra i due video
    fino a circa +/-15% della lunghezza totale della sequenza in qualunque
    punto — alzalo se sospetti differenze di velocita' molto piu' marcate
    in tratti brevi.

    Ritorna il percorso di allineamento come lista di coppie
    (indice_query, indice_reference), in ordine crescente di indice_query.
    """
    n, m = len(feat_query), len(feat_reference)

    # IMPORTANTE: la banda deve accomodare ALMENO la differenza di lunghezza
    # complessiva tra le due sequenze (abs(n-m)) — altrimenti il percorso non
    # riesce nemmeno teoricamente a collegare l'inizio (0,0) alla fine (n,m),
    # perche' il punto finale richiede uno scostamento dalla diagonale pari
    # esattamente a quella differenza. banda_frazione aggiunge margine extra
    # per la variazione LOCALE di velocita' oltre a quella differenza globale.
    banda = max(1, abs(n - m) + int(max(n, m) * banda_frazione))

    INF = float("inf")
    costo = np.full((n + 1, m + 1), INF, dtype=np.float64)
    costo[0, 0] = 0.0

    # matrice di costo locale (distanza euclidea tra feature normalizzate,
    # in [0,2] dato che le feature hanno norma 1)
    for i in range(1, n + 1):
        j_min = max(1, i - banda)
        j_max = min(m, i + banda)
        for j in range(j_min, j_max + 1):
            dist = float(np.linalg.norm(feat_query[i - 1] - feat_reference[j - 1]))
            costo[i, j] = dist + min(costo[i - 1, j], costo[i, j - 1], costo[i - 1, j - 1])

    # backtracking dal fondo della matrice — DEVE arrivare fino a (0,0), non
    # fermarsi appena uno dei due indici tocca zero: altrimenti si perde la
    # corrispondenza per tutta la parte iniziale della sequenza piu' lunga
    # (bug scoperto empiricamente: con sequenze di lunghezza molto diversa,
    # uno dei due indici puo' azzerarsi ben prima dell'altro).
    percorso = []
    i, j = n, m
    while i > 0 and j > 0:
        percorso.append((i - 1, j - 1))
        passi = [
            (costo[i - 1, j - 1], i - 1, j - 1),
            (costo[i - 1, j], i - 1, j),
            (costo[i, j - 1], i, j - 1),
        ]
        _, i, j = min(passi, key=lambda p: p[0])

    # completa il percorso lungo l'asse rimasto (se uno dei due indici ha
    # raggiunto zero prima dell'altro): i frame residui vengono abbinati al
    # frame 0 dell'altra sequenza — la parte iniziale del video piu' lungo
    # in questo tratto non ha un vero corrispondente 1:1, ma resta comunque
    # tracciata nella mappa invece di sparire silenziosamente
    while i > 0:
        i -= 1
        percorso.append((i, 0))
    while j > 0:
        j -= 1
        percorso.append((0, j))

    percorso.reverse()
    return percorso


def comprimi_percorso(percorso: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    DTW puo' abbinare piu' frame query allo stesso frame reference (o
    viceversa) nei tratti dove le velocita' divergono molto — per l'uso a
    valle (una corrispondenza per frame query) si tiene una sola coppia per
    ogni indice query (l'ultima, cioe' quella piu' avanzata nel reference
    per quell'istante query — scelta arbitraria ma coerente).
    """
    ultimo_per_query = {}
    for q, r in percorso:
        ultimo_per_query[q] = r
    return sorted(ultimo_per_query.items())


def scarta_bordi_senza_corrispondenza(percorso: List[Tuple[int, int]],
                                       feat_query: List[np.ndarray],
                                       feat_reference: List[np.ndarray],
                                       soglia_moltiplicatore: float = 2.5,
                                       min_run: int = 3) -> List[Tuple[int, Optional[int]]]:
    """
    Il DTW "classico" e' vincolato a collegare sempre l'inizio esatto con la
    fine esatta di entrambe le sequenze — se il reference e' fisicamente
    piu' corto (finisce prima sulla tratta) o inizia in un punto diverso,
    la coda/testa del percorso finisce FORZATA su un abbinamento privo di
    vero significato (verificato empiricamente: costo locale molto piu' alto
    del resto del percorso, invece di restare basso come nel tratto dove il
    contenuto corrisponde davvero).

    Questa funzione calcola il costo locale (distanza euclidea tra feature)
    di ogni corrispondenza gia' trovata, e AZZERA (sostituisce con None) i
    tratti CONTIGUI a inizio o fine del percorso il cui costo supera
    `soglia_moltiplicatore` volte la mediana globale — solo ai bordi, mai
    nel mezzo (un costo alto isolato in mezzo al percorso e' piu' probabile
    rumore visivo momentaneo che assenza di corrispondenza, e va tenuto).

    Ritorna la stessa lista (indice_query, indice_reference) ma con
    indice_reference = None dove la corrispondenza e' stata giudicata non
    valida — cosi' chi usa la mappa a valle (es. la pipeline) puo' scegliere
    di saltare il confronto col reference per quei frame, invece di
    confrontarli con un frame reference sbagliato.
    """
    costi = [float(np.linalg.norm(feat_query[q] - feat_reference[r])) for q, r in percorso]
    mediana = float(np.median(costi))
    soglia = mediana * soglia_moltiplicatore

    risultato: List[Tuple[int, Optional[int]]] = [(q, r) for q, r in percorso]
    n = len(risultato)

    # scarta dalla TESTA finche' il costo resta sopra soglia (run contiguo)
    i = 0
    while i < n and costi[i] > soglia:
        i += 1
    if i >= min_run:
        for k in range(i):
            q, _ = risultato[k]
            risultato[k] = (q, None)

    # scarta dalla CODA finche' il costo resta sopra soglia (run contiguo)
    j = n - 1
    while j >= 0 and costi[j] > soglia:
        j -= 1
    if (n - 1 - j) >= min_run:
        for k in range(j + 1, n):
            q, _ = risultato[k]
            risultato[k] = (q, None)

    return risultato


def scarta_coda_riferimento_esaurito(percorso: List[Tuple[int, int]],
                                      indice_max_reference: int,
                                      indice_min_reference: int = 0,
                                      min_run: int = 5) -> List[Tuple[int, Optional[int]]]:
    """
    Rilevamento MIRATO per il caso "il reference finisce fisicamente prima
    del query" (o inizia dopo): piu' affidabile del costo statistico su
    contenuto reale, dove le differenze di costo tra corrispondenze vere e
    forzate possono essere troppo deboli per una soglia (verificato
    empiricamente sul dataset reale — costo testa/coda solo ~1.4x quello
    del tratto centrale, contro un rapporto atteso ben piu' netto).

    Il percorso DTW e' vincolato a non decrescere sull'indice reference —
    quindi se il reference e' fisicamente esaurito, il percorso puo' SOLO
    ripetere il suo ultimo frame disponibile per tutti i restanti frame
    query: non e' un'ipotesi statistica, e' una conseguenza geometrica
    diretta del vincolo di monotonia. Un singolo tocco dell'estremo e'
    normale (l'ultimo frame puo' davvero corrispondere una volta); un
    BLOCCO PROLUNGATO sullo stesso estremo (>= min_run frame query di
    fila) e' il segnale che il reference non ha piu' contenuto nuovo da
    offrire, e va scartato — stesso ragionamento simmetrico applicato
    all'estremo opposto (indice_min_reference, es. se e' il query a
    iniziare prima che il reference abbia iniziato a coprire la tratta).
    """
    risultato: List[Tuple[int, Optional[int]]] = [(q, r) for q, r in percorso]
    n = len(risultato)

    # blocco in TESTA sull'estremo minimo del reference
    i = 0
    while i < n and risultato[i][1] == indice_min_reference:
        i += 1
    if i >= min_run:
        for k in range(i):
            q, _ = risultato[k]
            risultato[k] = (q, None)

    # blocco in CODA sull'estremo massimo del reference
    j = n - 1
    while j >= 0 and risultato[j][1] == indice_max_reference:
        j -= 1
    if (n - 1 - j) >= min_run:
        for k in range(j + 1, n):
            q, _ = risultato[k]
            risultato[k] = (q, None)

    return risultato


def trova_plateau_interni(percorso: List[Tuple[int, int]],
                           min_run: int = 15) -> List[Tuple[int, int, int]]:
    """
    Individua (senza modificare nulla) i blocchi nel percorso dove piu' di
    min_run corrispondenze consecutive puntano allo stesso identico frame
    reference — tipico segnale di ambiguita' locale del DTW grezzo (contenuto
    visivo poco distintivo in quel tratto, non necessariamente un rallentamento
    reale del treno).

    Ritorna una lista di tuple (indice_inizio, indice_fine, indice_reference)
    — indici POSIZIONALI dentro `percorso` (0-based), non frame_query/reference
    grezzi — chi chiama questa funzione converte secondo necessita'.
    """
    plateau = []
    n = len(percorso)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and percorso[j + 1][1] == percorso[i][1]:
            j += 1
        lunghezza_run = j - i + 1
        if lunghezza_run >= min_run:
            plateau.append((i, j, percorso[i][1]))
        i = j + 1
    return plateau


def fondi_plateau_vicini(plateau: List[Tuple[int, int, int]],
                          distanza_max: int = 10) -> List[Tuple[int, int, int]]:
    """
    Due plateau ADIACENTI O SOVRAPPOSTI (il gap tra la fine dell'uno e
    l'inizio del successivo e' <= distanza_max campioni) vengono fusi in un
    unico plateau, invece di essere raffinati separatamente con due finestre
    di ricerca indipendenti — verificato empiricamente: raffinare
    separatamente due plateau che si sovrappongono anche solo di 2-3
    campioni produce un salto incoerente esattamente al punto di
    sovrapposizione (una finestra "vince" sull'altra in modo arbitrario).

    Il valore r (terzo elemento della tupla) del plateau fuso non ha piu'
    un significato preciso (i plateau uniti potevano avere valori diversi)
    — resta solo a scopo di log/debug, raffina_plateau_locale non lo usa
    piu' direttamente (usa il contesto reale nel percorso).
    """
    if not plateau:
        return []

    plateau_ordinati = sorted(plateau, key=lambda p: p[0])
    fusi = [plateau_ordinati[0]]

    for i, j, r in plateau_ordinati[1:]:
        i_prec, j_prec, r_prec = fusi[-1]
        if i - j_prec <= distanza_max:
            # fonde: estende il plateau precedente fino alla fine di questo
            fusi[-1] = (i_prec, max(j_prec, j), r_prec)
        else:
            fusi.append((i, j, r))

    return fusi


def rileva_plateau_interni(percorso: List[Tuple[int, int]],
                            min_run: int = 15) -> List[Tuple[int, Optional[int]]]:
    """
    Estende scarta_coda_riferimento_esaurito(): quella funzione rileva un
    blocco SOLO agli estremi del percorso (reference esaurito prima/dopo).
    Questa rileva un blocco OVUNQUE nel percorso — tipicamente in un tratto
    dove il treno rallenta/si ferma (es. una stazione): per diversi secondi
    la scena cambia pochissimo, gli istogrammi diventano quasi indistinguibili
    tra frame consecutivi, e DTW puo' "bloccarsi" su un unico frame reference
    per un tratto di query, anche se la posizione fisica nel frattempo si
    sposta leggermente — risultato: un vero sfasamento spaziale (non solo
    fotometrico), verificato empiricamente su questo progetto: un blocco di
    3+ frame query consecutivi mappati sullo stesso identico frame reference,
    con contenuto visibilmente piu' vicino/grande nel reference rispetto al
    query nello stesso istante.

    NOTA: questa funzione SCARTA (frame_reference=None) — utile se preferisci
    non fidarti di quel tratto piuttosto che raffinarlo. Per un raffinamento
    locale invece di uno scarto, vedi raffina_plateau_locale() in main().

    min_run e' volutamente PIU' ALTO del default usato per la coda (5): un
    breve blocco di 2-4 frame puo' essere legittimo (il treno rallenta
    davvero un po'), va scartato solo un blocco abbastanza lungo da indicare
    un problema reale di precisione, non normale variazione di velocita'.

    Il PRIMO frame di ogni blocco viene tenuto (e' comunque un match valido,
    il problema nasce dal SECONDO in poi, quando il reference smette di
    avanzare mentre il query continua) — solo dal secondo in poi vengono
    marcati come None.
    """
    risultato: List[Tuple[int, Optional[int]]] = [(q, r) for q, r in percorso]
    for i, j, _r_plateau in trova_plateau_interni(percorso, min_run=min_run):
        for k in range(i + 1, j + 1):
            q, _ = risultato[k]
            risultato[k] = (q, None)
    return risultato



def _estrai_feature_finestra(percorso_video: str, frame_inizio: int, frame_fine: int):
    """
    Come estrai_feature_video, ma con SEEK diretto a frame_inizio invece di
    rileggere il video dall'inizio — essenziale per un raffinamento locale su
    un plateau che puo' trovarsi a meta'/fine di un video lungo (altrimenti,
    per ogni plateau raffinato, si rileggerebbero tutti i frame precedenti).
    """
    cap = cv2.VideoCapture(percorso_video)
    if not cap.isOpened():
        raise IOError(f"Impossibile aprire il video: {percorso_video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inizio)

    features, indici = [], []
    idx = frame_inizio
    while idx <= frame_fine:
        ok, frame = cap.read()
        if not ok:
            break
        features.append(calcola_feature_frame(frame))
        indici.append(idx)
        idx += 1
    cap.release()
    return features, indici


def rileva_violazione_monotonia(righe_finali: List[list]) -> int:
    """
    Criterio automatico più affidabile di rileva_esaurimento_automatico:
    invece di controllare "sei vicino all'estremo del reference" (euristica
    statistica, puo' avere falsi positivi/negativi), controlla un FATTO
    FISICO inequivocabile — il reference non puo' MAI scendere sotto un
    valore gia' raggiunto in precedenza (il treno non torna indietro sulla
    tratta). Vale OVUNQUE nel percorso, non solo vicino agli estremi — un
    passo indietro e' fisicamente impossibile a meta' video esattamente
    come alla fine.

    Due tipi di violazione, distinti automaticamente:
      1. CALO CHE SI RIPRENDE: il valore scende, ma entro poche righe la
         sequenza torna a raggiungere/superare il massimo gia' visto prima
         del calo — tipico di un raccordo imperfetto fra un tratto raffinato
         e i dati grezzi adiacenti (scoperto empiricamente: la fine di un
         plateau raffinato puo' non incastrarsi perfettamente con cio' che
         viene subito dopo). In questo caso si scarta SOLO il tratto
         intermedio (dal calo fino al punto di recupero, escluso), non tutto
         il resto del video.
      2. CALO CHE NON SI RIPRENDE PIU': la sequenza non torna mai a
         raggiungere il massimo precedente fino alla fine del video — segno
         che da li' in poi il reference non ha piu' vero contenuto
         corrispondente (es. la vera coda senza corrispondenza, dove il DTW
         grezzo residuo "spalma" il disavanzo invece di fermarsi). In questo
         caso si scarta tutto da quel punto fino alla fine.

    NOTA IMPORTANTE (limite noto): quando il calo si riprende (caso 1), lo
    script assume che il valore PIU' ALTO (raggiunto prima del calo) sia
    quello corretto, e scarta il tratto piu' basso intermedio — un'assunzione
    ragionevole ma non verificata algoritmicamente: in linea di principio
    potrebbe essere il tratto piu' basso quello giusto, e il raffinamento
    precedente ad aver sovrastimato. Se dopo aver usato la mappa noti ancora
    un tratto sospetto vicino a un punto dove questo criterio e' intervenuto,
    vale la pena un controllo visivo mirato con estrai_frame.py.

    Ritorna il numero di righe scartate.
    """
    indici_validi = [k for k, r in enumerate(righe_finali) if r[1] is not None]
    n_scartate = 0
    massimo_visto = -1
    pos = 0
    while pos < len(indici_validi):
        k = indici_validi[pos]
        fr = righe_finali[k][1]

        if fr >= massimo_visto:
            massimo_visto = fr
            pos += 1
            continue

        # violazione: fr < massimo_visto — cerca il primo punto successivo
        # (se esiste) in cui la sequenza recupera (torna a >= massimo_visto)
        punto_recupero = None
        for pos2 in range(pos + 1, len(indici_validi)):
            k2 = indici_validi[pos2]
            if righe_finali[k2][1] >= massimo_visto:
                punto_recupero = pos2
                break

        fine_scarto = punto_recupero if punto_recupero is not None else len(indici_validi)
        for pos3 in range(pos, fine_scarto):
            k3 = indici_validi[pos3]
            righe_finali[k3][1] = None
            righe_finali[k3][2] = None
            n_scartate += 1

        if punto_recupero is None:
            break  # non recupera piu': scartato tutto fino alla fine, finito
        pos = punto_recupero  # riprende il controllo normale da qui

    return n_scartate


def rileva_esaurimento_automatico(righe_finali: List[list], indice_max_reference: int,
                                   indice_min_reference: int = 0,
                                   min_run: int = 10, tolleranza: int = 5) -> Tuple[int, int]:
    """
    Sostituisce lo scarto manuale basato su un numero di secondi deciso a
    occhio (--scarta-ultimi/primi-secondi-query) con un rilevamento
    AUTOMATICO — applicato ai dati GIA' RAFFINATI (molto piu' affidabili
    della mappa grezza campionata, che si era rivelata inaffidabile per
    questo scopo su dati reali).

    Scansiona righe_finali dai due estremi: se un tratto (testa o coda) resta
    bloccato entro `tolleranza` frame dal vero estremo disponibile del
    reference per almeno `min_run` righe consecutive, e' un segnale oggettivo
    che il reference e' fisicamente esaurito li' — non serve che un umano
    guardi i frame e indovini un numero di secondi, funziona automaticamente
    per qualunque coppia di video (durate diverse, tratte diverse).

    Ritorna (n_scartate_testa, n_scartate_coda) — le righe interessate
    vengono modificate IN PLACE (frame_reference/costo impostati a None).
    """
    n = len(righe_finali)
    if n == 0:
        return 0, 0

    # --- CODA: quante righe finali restano vicine al MASSIMO reference? ---
    j = n - 1
    while j >= 0:
        fr = righe_finali[j][1]
        if fr is None or fr < indice_max_reference - tolleranza:
            break
        j -= 1
    lunghezza_coda = n - 1 - j
    n_coda = 0
    if lunghezza_coda >= min_run:
        for k in range(j + 1, n):
            if righe_finali[k][1] is not None:
                righe_finali[k][1] = None
                righe_finali[k][2] = None
                n_coda += 1

    # --- TESTA: quante righe iniziali restano vicine al MINIMO reference? ---
    i = 0
    while i < n:
        fr = righe_finali[i][1]
        if fr is None or fr > indice_min_reference + tolleranza:
            break
        i += 1
    lunghezza_testa = i
    n_testa = 0
    if lunghezza_testa >= min_run:
        for k in range(0, i):
            if righe_finali[k][1] is not None:
                righe_finali[k][1] = None
                righe_finali[k][2] = None
                n_testa += 1

    return n_testa, n_coda


def raffina_plateau_locale(percorso_globale: List[Tuple[int, int]],
                            plateau: Tuple[int, int, int],
                            idx_query: List[int], idx_ref: List[int],
                            percorso_video_query: str, percorso_video_ref: str,
                            fattore_margine: float = 2.0,
                            margine_ref_max_campioni: int = 80,
                            contesto_extra_campioni: int = 10) -> dict:
    """
    Invece di scartare un plateau (rileva_plateau_interni), rilancia DTW a
    grana FINE (un frame ogni raw, non campionato ogni --step) in una finestra
    ristretta attorno al plateau — costo computazionale trascurabile perche'
    la finestra e' piccola (una frazione del video intero), ma la precisione
    aumenta parecchio: il DTW grezzo campionato puo' perdere sfumature che
    diventano visibili guardando ogni singolo frame invece che 1 ogni 5.

    La finestra REFERENCE si basa sui valori r osservati nel percorso grezzo
    appena PRIMA e DOPO il plateau (contesto_extra_campioni campioni ai due
    lati), non solo su un margine simmetrico attorno al valore centrale — cosi'
    gestisce bene anche due plateau adiacenti con valori diversi (il contesto
    dell'uno include naturalmente l'altro), invece di trattarli come due
    finestre scollegate che possono produrre un salto al confine.

    margine_ref_max_campioni mette un TETTO al margine aggiuntivo, a prescindere
    da quanto e' grande il plateau — senza questo, un plateau molto esteso (es.
    15+ secondi, verificato empiricamente su questo progetto) fa esplodere il
    margine fino a coprire l'intero video reference, vanificando lo scopo di un
    raffinamento LOCALE ed economico.

    Ritorna un dizionario {frame_query_raw: frame_reference_raw} con la
    corrispondenza raffinata SOLO per i frame query raw coperti dal plateau
    (compreso tra idx_query[i] e idx_query[j] inclusi) — chi chiama questa
    funzione fonde questo dizionario nella mappa finale, sostituendo i valori
    grezzi/scartati in quel tratto.
    """
    i, j, r_centro = plateau

    # finestra sui frame QUERY: tutti i frame raw (non campionati) tra l'inizio
    # e la fine del plateau, con un margine di sicurezza ai due lati
    frame_q_inizio = idx_query[max(0, i - 1)]
    frame_q_fine = idx_query[min(len(idx_query) - 1, j + 1)]
    margine_q = int((frame_q_fine - frame_q_inizio) * 0.2)
    frame_q_inizio = max(0, frame_q_inizio - margine_q)

    # finestra sui frame REFERENCE: guarda i valori r EFFETTIVAMENTE osservati
    # nel percorso grezzo un po' prima/dopo il plateau, invece di assumere
    # simmetria attorno a un solo valore centrale
    ctx_min = max(0, i - contesto_extra_campioni)
    ctx_max = min(len(percorso_globale) - 1, j + contesto_extra_campioni)
    valori_r_contesto = [percorso_globale[k][1] for k in range(ctx_min, ctx_max + 1)]
    r_min_osservato = min(valori_r_contesto)
    r_max_osservato = max(valori_r_contesto)

    ampiezza_plateau_campioni = j - i + 1
    margine_ref_campioni = min(
        max(5, int(ampiezza_plateau_campioni * fattore_margine)),
        margine_ref_max_campioni,
    )

    # se il plateau e' gia' vicino a un VERO estremo del reference (indice
    # 0 o l'ultimo disponibile), la finestra si estende fino a quell'estremo
    # SENZA il tetto — il tetto serve solo a evitare l'esplosione generica
    # (plateau interno, nessun vero confine vicino), non a impedire di
    # raggiungere un confine reale che sappiamo esistere. Verificato
    # empiricamente: con il tetto applicato anche qui, un plateau vicino
    # alla vera fine del reference veniva raffinato con una finestra che si
    # fermava PRIMA del vero ultimo frame, dando l'impressione (sbagliata)
    # che la corrispondenza si esaurisse per divergenza di contenuto invece
    # che per un limite artificiale della finestra di ricerca.
    #
    # IMPORTANTE: la vicinanza va giudicata sulla POSIZIONE del plateau
    # nella timeline del QUERY (i, j rispetto a percorso_globale), NON sul
    # valore di reference osservato — un plateau a META' del query puo'
    # avere per altri motivi un valore di reference numericamente basso
    # senza che questo significhi "vicino al vero inizio del reference".
    # Bug reale scoperto empiricamente: un plateau a query~700 su un totale
    # di 3113 (tutt'altro che vicino all'inizio) veniva esteso fino a
    # reference=0 solo perche' il suo valore osservato era numericamente
    # piccolo — causava una violazione di monotonia con i dati non
    # raffinati appena prima, che invece avevano gia' un reference piu' alto.
    margine_posizionale = max(10, int(len(percorso_globale) * 0.20))
    vicino_al_minimo = i <= margine_posizionale
    vicino_al_massimo = j >= len(percorso_globale) - 1 - margine_posizionale

    if vicino_al_minimo:
        idx_r_min_campione = 0
    else:
        idx_r_min_campione = max(0, r_min_osservato - margine_ref_campioni)

    if vicino_al_massimo:
        idx_r_max_campione = len(idx_ref) - 1
    else:
        idx_r_max_campione = min(len(idx_ref) - 1, r_max_osservato + margine_ref_campioni)

    # VINCOLO DI CONTINUITA': la finestra non puo' mai scendere sotto il
    # valore GIA' STABILITO immediatamente prima del plateau nel percorso
    # grezzo — senza questo vincolo, il DTW locale e' libero di scegliere
    # un valore basso all'interno di una finestra ampia, anche se questo
    # contraddice (viola la monotonia con) il dato appena precedente non
    # toccato dal raffinamento. Bug reale scoperto empiricamente: una
    # finestra "ragionevole" (non estesa fino a 0) ha comunque prodotto una
    # violazione di monotonia al primo frame del plateau, perche' nessun
    # vincolo la legava al valore immediatamente precedente.
    if i > 0:
        valore_precedente = percorso_globale[i - 1][1]
        idx_r_min_campione = max(idx_r_min_campione, valore_precedente)
        idx_r_max_campione = max(idx_r_max_campione, idx_r_min_campione)

    frame_r_inizio = idx_ref[idx_r_min_campione]
    frame_r_fine = idx_ref[idx_r_max_campione]

    print(f"    Raffino plateau: query raw [{frame_q_inizio}-{frame_q_fine}] "
          f"vs reference raw [{frame_r_inizio}-{frame_r_fine}]"
          f"{' (esteso fino al vero estremo reference)' if vicino_al_massimo or vicino_al_minimo else ''}...")

    # estrae feature FRAME PER FRAME (ogni raw frame, non campionato) nelle
    # due finestre ristrette, con seek diretto (niente rilettura dall'inizio)
    feat_q_fine, idx_q_fine = _estrai_feature_finestra(
        percorso_video_query, frame_q_inizio, frame_q_fine)
    feat_r_fine, idx_r_fine = _estrai_feature_finestra(
        percorso_video_ref, frame_r_inizio, frame_r_fine)

    if len(feat_q_fine) < 2 or len(feat_r_fine) < 2:
        print("    [WARN] Finestra troppo piccola per un raffinamento affidabile, salto.")
        return {}

    # DTW locale, banda larga (le finestre sono gia' piccole e mirate, non
    # serve un vincolo stretto qui — il problema che stiamo risolvendo e'
    # proprio la scarsa precisione del DTW grezzo su questo tratto)
    percorso_fine = dtw_allineamento(feat_q_fine, feat_r_fine, banda_frazione=0.8)
    percorso_fine = comprimi_percorso(percorso_fine)

    mappa_raffinata = {}
    for i_q_fine, i_r_fine in percorso_fine:
        frame_q_raw = idx_q_fine[i_q_fine]
        frame_r_raw = idx_r_fine[i_r_fine]
        mappa_raffinata[frame_q_raw] = frame_r_raw

    return mappa_raffinata


def unisci_scarti(*liste_percorso: List[Tuple[int, Optional[int]]]) -> List[Tuple[int, Optional[int]]]:
    """Combina piu' liste di scarto (stesso ordine/lunghezza): una
    corrispondenza risulta scartata (None) se lo e' in ALMENO UNA delle
    liste — i due criteri (costo anomalo, blocco sull'estremo) si
    rafforzano a vicenda invece di doverne scegliere uno solo."""
    base = list(liste_percorso[0])
    for altra in liste_percorso[1:]:
        for k in range(len(base)):
            if altra[k][1] is None:
                base[k] = (base[k][0], None)
    return base


# ==================== MAIN ====================

def main():
    ap = argparse.ArgumentParser(description="Sincronizza temporalmente reference/query via DTW")
    ap.add_argument("--reference", required=True, help="Percorso al video reference")
    ap.add_argument("--query", required=True, help="Percorso al video query")
    ap.add_argument("--output", required=True, help="Percorso del CSV di output (mappa di sincronizzazione)")
    ap.add_argument("--step", type=int, default=5,
                     help="Campiona 1 frame ogni N per il calcolo (default 5)")
    ap.add_argument("--banda-frazione", type=float, default=0.15,
                     help="Vincolo di banda DTW, vedi dtw_allineamento() (default 0.15)")
    ap.add_argument("--max-frame", type=int, default=None,
                     help="Limite di frame da processare per video (utile per un test rapido)")
    ap.add_argument("--soglia-costo-multiplo", type=float, default=2.5,
                     help="Vedi scarta_bordi_senza_corrispondenza(): moltiplicatore sulla "
                          "mediana del costo, oltre il quale un tratto di bordo viene scartato "
                          "come privo di vera corrispondenza (default 2.5)")
    ap.add_argument("--min-run-scarto", type=int, default=3,
                     help="Numero minimo di corrispondenze contigue a costo anomalo richieste "
                          "prima di scartare un bordo (default 3, evita di scartare per un "
                          "singolo frame rumoroso isolato)")
    ap.add_argument("--nessuno-scarto-bordi", action="store_true",
                     help="Disattiva lo scarto automatico dei bordi a costo anomalo/geometrico "
                          "— comportamento precedente, ogni frame query ottiene sempre una "
                          "corrispondenza (anche se forzata/priva di senso ai bordi)")
    ap.add_argument("--scarta-ultimi-secondi-query", type=float, default=0.0,
                     help="Scarta manualmente gli ultimi N secondi del QUERY (frame_reference "
                          "vuoto), indipendentemente da cosa rilevano i criteri automatici. "
                          "Utile quando i criteri automatici (costo, blocco geometrico) non "
                          "riescono a distinguere in modo affidabile il tratto senza vera "
                          "corrispondenza — determina il valore giusto con un controllo visivo "
                          "(vedi estrai_frame.py --da-mappa) invece di indovinarlo. NON PIU' "
                          "NECESSARIO in condizioni normali: il rilevamento automatico "
                          "dell'esaurimento reference (vedi --nessuno-scarto-coda-automatico) "
                          "copre gia' questo caso senza bisogno di un input manuale — usa "
                          "questo flag solo come eccezione/override per casi particolari.")
    ap.add_argument("--scarta-primi-secondi-query", type=float, default=0.0,
                     help="Come --scarta-ultimi-secondi-query, ma per l'INIZIO del query.")
    ap.add_argument("--nessuno-scarto-coda-automatico", action="store_true",
                     help="Disattiva il rilevamento AUTOMATICO dell'esaurimento del reference "
                          "(testa/coda bloccate vicino agli estremi del reference sui dati "
                          "raffinati). Attivo per default — sostituisce lo scarto manuale come "
                          "meccanismo primario, funziona senza bisogno di determinare un "
                          "numero di secondi a occhio per ogni nuova coppia di video.")
    ap.add_argument("--min-run-scarto-automatico", type=int, default=10,
                     help="Numero minimo di righe consecutive bloccate vicino all'estremo "
                          "reference prima di considerarlo esaurito (default 10).")
    ap.add_argument("--tolleranza-scarto-automatico", type=int, default=5,
                     help="Quanto vicino (in frame reference) all'estremo va considerato "
                          "'bloccato' per il rilevamento automatico (default 5).")
    ap.add_argument("--min-run-plateau-interno", type=int, default=15,
                     help="Rileva blocchi (stesso frame reference per piu' frame query "
                          "consecutivi) OVUNQUE nel percorso, non solo agli estremi — tipico "
                          "di un tratto dove il DTW campionato perde precisione (contenuto "
                          "visivo poco distintivo, non necessariamente un rallentamento "
                          "reale). Default 15. 0 disattiva questo criterio.")
    ap.add_argument("--scarta-plateau", action="store_true",
                     help="Per i plateau rilevati, SCARTA il tratto (frame_reference vuoto) "
                          "invece di raffinarlo con un secondo passaggio DTW frame-per-frame "
                          "(comportamento di default). Usa questo se preferisci non fidarti "
                          "affatto di quel tratto piuttosto che tentare un raffinamento.")
    ap.add_argument("--fattore-margine-raffinamento", type=float, default=2.0,
                     help="Ampiezza della finestra di ricerca reference durante il "
                          "raffinamento locale di un plateau (vedi raffina_plateau_locale), "
                          "come multiplo dell'ampiezza del plateau stesso. Default 2.0. Se "
                          "noti SALTI improvvisi nella mappa raffinata (es. un frame che "
                          "punta a un reference molto lontano dai vicini), la finestra era "
                          "probabilmente troppo stretta — prova un valore piu' alto (es. 4.0).")
    ap.add_argument("--margine-ref-max-campioni", type=int, default=80,
                     help="Tetto massimo al margine di ricerca reference durante il "
                          "raffinamento (default 80 campioni) — indipendente da quanto e' "
                          "grande il plateau. Senza questo tetto, un plateau molto esteso "
                          "(15+ secondi, verificato empiricamente) fa esplodere la finestra "
                          "di ricerca fino a coprire l'intero reference, vanificando lo "
                          "scopo di un raffinamento locale ed economico. Se un plateau molto "
                          "grande viene rilevato, e' probabilmente un problema DIVERSO (piu' "
                          "serio) da quello che il raffinamento locale puo' risolvere da solo "
                          "— vedi il messaggio di avviso stampato in quel caso.")
    args = ap.parse_args()

    print(f"Estraggo feature dal reference: {args.reference}")
    feat_ref, idx_ref, fps_ref = estrai_feature_video(args.reference, args.step, args.max_frame)
    print(f"  {len(feat_ref)} frame campionati (fps={fps_ref:.2f})")

    print(f"Estraggo feature dal query: {args.query}")
    feat_query, idx_query, fps_query = estrai_feature_video(args.query, args.step, args.max_frame)
    print(f"  {len(feat_query)} frame campionati (fps={fps_query:.2f})")

    if len(feat_ref) < 2 or len(feat_query) < 2:
        print("[ERRORE] Troppo pochi frame campionati — controlla i video/il parametro --step.")
        sys.exit(1)

    print("Eseguo DTW (puo' richiedere qualche minuto su video lunghi)...")
    percorso = dtw_allineamento(feat_query, feat_ref, args.banda_frazione)
    percorso = comprimi_percorso(percorso)
    print(f"  Percorso di allineamento: {len(percorso)} corrispondenze")

    # costo locale per ogni corrispondenza — calcolato SEMPRE (indipendentemente
    # da --nessuno-scarto-bordi), scritto come colonna diagnostica nel CSV: utile
    # per scegliere --soglia-costo-multiplo sui dati reali invece che a intuito
    costi = [float(np.linalg.norm(feat_query[q] - feat_ref[r])) for q, r in percorso]
    mediana_costo = float(np.median(costi))
    print(f"  Costo locale: min={min(costi):.4f} mediana={mediana_costo:.4f} "
          f"media={float(np.mean(costi)):.4f} max={max(costi):.4f} "
          f"(soglia scarto attuale: {mediana_costo * args.soglia_costo_multiplo:.4f})")

    if args.nessuno_scarto_bordi:
        percorso_finale = [(q, r) for q, r in percorso]
    else:
        percorso_costo = scarta_bordi_senza_corrispondenza(
            percorso, feat_query, feat_ref,
            soglia_moltiplicatore=args.soglia_costo_multiplo,
            min_run=args.min_run_scarto,
        )
        # criterio geometrico: rileva un blocco prolungato sull'estremo del
        # reference (segnale affidabile anche quando il costo statistico e'
        # troppo debole per distinguersi dal rumore, come verificato su dati
        # reali) — indice_max_reference = m-1 dato che gli indici in percorso
        # sono 0-based sull'array di feature del reference
        percorso_geometrico = scarta_coda_riferimento_esaurito(
            percorso, indice_max_reference=len(feat_ref) - 1, indice_min_reference=0,
            min_run=args.min_run_scarto,
        )
        liste_da_unire = [percorso_costo, percorso_geometrico]

        # i plateau interni, per default, vengono RAFFINATI (vedi sotto, dopo
        # aver calcolato percorso_finale) invece che scartati — solo se
        # --scarta-plateau e' esplicitamente richiesto si aggiungono qui
        # all'unione degli scarti, come gli altri criteri
        if args.min_run_plateau_interno > 0 and args.scarta_plateau:
            percorso_plateau = rileva_plateau_interni(
                percorso, min_run=args.min_run_plateau_interno)
            liste_da_unire.append(percorso_plateau)

        percorso_finale = unisci_scarti(*liste_da_unire)

    # scarto MANUALE (in secondi), applicato indipendentemente dai criteri
    # automatici sopra (funziona anche con --nessuno-scarto-bordi) — basato
    # sul controllo visivo dell'utente: piu' affidabile quando il segnale
    # statistico/geometrico e' troppo debole per essere rilevato in modo
    # automatico su contenuto reale (verificato su questo stesso progetto:
    # ne' il costo ne' il blocco geometrico si sono dimostrati sufficienti)
    if args.scarta_ultimi_secondi_query > 0 or args.scarta_primi_secondi_query > 0:
        tempo_totale_query = idx_query[-1] / fps_query
        soglia_fine = tempo_totale_query - args.scarta_ultimi_secondi_query
        soglia_inizio = args.scarta_primi_secondi_query

        percorso_manuale = []
        for q, r in percorso_finale:
            frame_q = idx_query[q]
            t_q = frame_q / fps_query
            if t_q < soglia_inizio or t_q > soglia_fine:
                percorso_manuale.append((q, None))
            else:
                percorso_manuale.append((q, r))
        percorso_finale = percorso_manuale

    n_scartate = sum(1 for _, r in percorso_finale if r is None)
    if n_scartate > 0:
        print(f"  [ATTENZIONE] {n_scartate} corrispondenze scartate come prive "
              f"di vera corrispondenza (costo anomalo, reference esaurito, "
              f"e/o scarto manuale) — quei frame query avranno frame_reference vuoto nel CSV.")
    else:
        print("  Nessun tratto di bordo scartato (nessun criterio ha rilevato problemi).")

    # costruisco le righe finali (indicizzate per frame RAW, non piu' per
    # posizione campionata) partendo dalla mappa grossolana gia' calcolata
    righe_finali = []
    for (i_query_camp, i_ref_camp), costo_riga in zip(percorso_finale, costi):
        frame_q = idx_query[i_query_camp]
        frame_r = idx_ref[i_ref_camp] if i_ref_camp is not None else None
        righe_finali.append([frame_q, frame_r, round(costo_riga, 5)])

    # raffinamento dei plateau interni (default): sostituisce le poche righe
    # grossolane in quel tratto con molte righe fini (una per frame raw),
    # ottenute rilanciando DTW frame-per-frame in una finestra ristretta —
    # invece di scartare il tratto, lo si rende piu' preciso
    if args.min_run_plateau_interno > 0 and not args.scarta_plateau:
        plateau_trovati = trova_plateau_interni(percorso, min_run=args.min_run_plateau_interno)
        n_prima_fusione = len(plateau_trovati)
        print(f"\n  Plateau individuati PRIMA della fusione (confini VERI, senza margine): "
              f"{[(idx_query[i], idx_query[j]) for i, j, _ in plateau_trovati]}")
        plateau_trovati = fondi_plateau_vicini(plateau_trovati, distanza_max=10)
        if len(plateau_trovati) < n_prima_fusione:
            print(f"  ({n_prima_fusione} plateau individuati, fusi in {len(plateau_trovati)} "
                  f"perche' adiacenti/sovrapposti — evita raffinamenti incoerenti al confine)")
        print(f"  Plateau DOPO la fusione (confini VERI, senza margine): "
              f"{[(idx_query[i], idx_query[j]) for i, j, _ in plateau_trovati]}")
        if plateau_trovati:
            print(f"\n  Trovati {len(plateau_trovati)} plateau interni — raffino ciascuno "
                  f"con un secondo passaggio DTW frame-per-frame...")
        for plateau in plateau_trovati:
            i_pl_check, j_pl_check, _ = plateau
            lunghezza_campioni = j_pl_check - i_pl_check + 1
            durata_stimata_s = lunghezza_campioni * args.step / fps_query
            if durata_stimata_s > 5.0:
                print(f"    [ATTENZIONE] Plateau insolitamente grande "
                      f"(~{durata_stimata_s:.1f}s di query) — potrebbe essere un problema "
                      f"DIVERSO (piu' serio) da una semplice ambiguita' locale del DTW: "
                      f"un tratto dove reference e query divergono davvero parecchio. Il "
                      f"raffinamento procede comunque (con il tetto sul margine), ma "
                      f"controlla il risultato con un confronto visivo prima di fidartene.")

            mappa_raffinata = raffina_plateau_locale(
                percorso, plateau, idx_query, idx_ref, args.query, args.reference,
                fattore_margine=args.fattore_margine_raffinamento,
                margine_ref_max_campioni=args.margine_ref_max_campioni)
            if not mappa_raffinata:
                print("    Raffinamento non riuscito per questo plateau, resta il valore grossolano.")
                continue

            i_pl, j_pl, _ = plateau
            # IMPORTANTE: uso il range EFFETTIVO delle chiavi raffinate (non i
            # confini stretti del plateau) per rimuovere le righe grossolane —
            # raffina_plateau_locale usa un margine di sicurezza che estende
            # leggermente oltre i confini del plateau, quindi rimuovere solo
            # in base ai confini stretti lascerebbe residui grossolani
            # duplicati/sovrapposti alle nuove righe fini
            frame_q_min = min(mappa_raffinata.keys())
            frame_q_max = max(mappa_raffinata.keys())
            # rimuovo le righe grossolane nel range coperto dal plateau...
            righe_finali = [r for r in righe_finali if not (frame_q_min <= r[0] <= frame_q_max)]
            # ...e le sostituisco con le righe fini raffinate (costo_locale
            # non disponibile a questa granularita' diversa, lasciato vuoto)
            for frame_q_raw, frame_r_raw in sorted(mappa_raffinata.items()):
                righe_finali.append([frame_q_raw, frame_r_raw, None])

        print(f"  Fatto — {len(righe_finali)} righe totali dopo il raffinamento "
              f"(erano {len(percorso_finale)} prima).")

    # l'ordinamento avviene SEMPRE (indipendentemente da se il raffinamento
    # plateau e' stato eseguito) — necessario prima del rilevamento
    # automatico dell'esaurimento qui sotto
    righe_finali.sort(key=lambda r: r[0])

    # rilevamento AUTOMATICO — due criteri, applicati in ordine:
    # 1) violazione di monotonia (fatto fisico inequivocabile: il reference
    #    non puo' mai tornare indietro — vedi rileva_violazione_monotonia)
    # 2) vicinanza all'estremo (euristica statistica, cattura casi che il
    #    criterio 1 non vede, es. un tratto che resta bloccato senza mai
    #    violare la monotonia)
    # Entrambi funzionano senza bisogno che un umano guardi i frame e
    # indovini un numero di secondi — validi per qualunque coppia di video.
    if not args.nessuno_scarto_coda_automatico:
        n_monotonia = rileva_violazione_monotonia(righe_finali)
        if n_monotonia > 0:
            print(f"\n  [RILEVAMENTO AUTOMATICO] Violazione di monotonia rilevata "
                  f"(il reference tornava indietro) — {n_monotonia} righe scartate da "
                  f"quel punto in poi (nessun input manuale richiesto).")

        n_testa_auto, n_coda_auto = rileva_esaurimento_automatico(
            righe_finali, indice_max_reference=idx_ref[-1], indice_min_reference=idx_ref[0],
            min_run=args.min_run_scarto_automatico, tolleranza=args.tolleranza_scarto_automatico)
        if n_testa_auto > 0 or n_coda_auto > 0:
            print(f"  [RILEVAMENTO AUTOMATICO] Reference esaurito: {n_testa_auto} righe "
                  f"scartate in testa, {n_coda_auto} in coda (nessun input manuale richiesto).")


    # RI-APPLICO lo scarto manuale (in secondi) DOPO il raffinamento dei
    # plateau: un plateau puo' estendersi dentro la zona che l'utente ha
    # gia' deciso di scartare (es. verificato empiricamente su questo
    # progetto — un plateau enorme, sintomo di reference che si esaurisce
    # gradualmente, che si estendeva ben dentro gli ultimi N secondi gia'
    # scartati). Senza questo secondo passaggio, il raffinamento "resuscita"
    # con un valore fasullo (il migliore trovabile in una finestra comunque
    # limitata) proprio i frame che l'utente aveva gia' deciso, a ragione,
    # di non fidarsi — lo scarto manuale deve avere sempre l'ultima parola.
    if args.scarta_ultimi_secondi_query > 0 or args.scarta_primi_secondi_query > 0:
        tempo_totale_query_raw = idx_query[-1] / fps_query
        soglia_fine_raw = tempo_totale_query_raw - args.scarta_ultimi_secondi_query
        soglia_inizio_raw = args.scarta_primi_secondi_query

        n_ri_scartate = 0
        for riga in righe_finali:
            t_q = riga[0] / fps_query
            if (t_q < soglia_inizio_raw or t_q > soglia_fine_raw) and riga[1] is not None:
                riga[1] = None
                riga[2] = None
                n_ri_scartate += 1
        if n_ri_scartate > 0:
            print(f"  [ATTENZIONE] {n_ri_scartate} righe raffinate erano ricadute nella "
                  f"zona di scarto manuale — ri-scartate (lo scarto manuale ha sempre "
                  f"l'ultima parola sul raffinamento).")

    print(f"\nScrivo la mappa in: {args.output}")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_query", "tempo_query_s", "frame_reference", "tempo_reference_s", "costo_locale"])
        for frame_q, frame_r, costo in righe_finali:
            costo_str = costo if costo is not None else ""
            if frame_r is None:
                writer.writerow([frame_q, round(frame_q / fps_query, 3), "", "", costo_str])
            else:
                writer.writerow([
                    frame_q, round(frame_q / fps_query, 3),
                    frame_r, round(frame_r / fps_ref, 3),
                    costo_str,
                ])

    print("\n=== FATTO ===")
    print(f"Mappa di sincronizzazione salvata in: {args.output}")
    print("Ogni riga: un frame del query -> il frame del reference che gli corrisponde "
          "piu' da vicino visivamente (risoluzione mista: campionata per la maggior parte "
          "del video, frame-per-frame nei tratti raffinati).")


if __name__ == "__main__":
    main()