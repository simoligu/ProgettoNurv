# -*- coding: utf-8 -*-
"""
Generazione dataset sintetico — pali inclinati (Unreal Engine 5.7).
VERSIONE ASSET 2 — stesso identico script della versione principale
(genera_dataset_pali.py), con SOLO i parametri specifici del nuovo asset
modificati (vedi i blocchi marcati "<<< DA MODIFICARE" qui sotto).

PERCHE' UNO SCRIPT SEPARATO invece di un flag nello stesso file: questo
asset e' pensato per restare SOLO nel test/validation set, mai mescolato
nel training — tenerlo in un file fisicamente separato riduce il rischio
di lanciarlo per sbaglio nella cartella di output del dataset principale
o di confondere le due configurazioni durante la sperimentazione.

Logica invariata rispetto alla V3 (nuvole, motion blur, distanza target
assoluta, salvataggio incrementale etichette): vedi commenti nel resto del
file per i dettagli, qui sono ripetuti solo dove serve capire cosa cambia.

COME LANCIARLO: identico allo script principale, nella riga di comando
Python dell'editor:
    exec(open(r"C:\percorso\genera_dataset_pali_asset2.py").read())
"""

import unreal
import os
import glob
import shutil
import random
import json
import time
import math

# ==================== CONFIGURAZIONE (modifica qui) ====================

NOMI_PIVOT = ["PolePivot_NuovoAsset"]
# <<< DA MODIFICARE: inserisci qui i nomi esatti (label Outliner) dei pivot
# duplicati per il nuovo asset importato da Sketchfab. Se importi un solo
# palo e lo duplichi piu' volte per avere varieta' di posizione lungo il
# binario (come avete fatto con Base2/3/4), elenca tutti i duplicati qui.
# Se ne tieni solo uno, la lista puo' restare con un solo elemento — perde
# solo la varieta' di distanza di partenza, non il senso del test.

ASSE_ROTAZIONE = "pitch"   # cambia in "roll" se l'inclinazione visiva è sbagliata

# --- movimento del viewport tra uno scatto e l'altro ---
# Simula il treno che avanza/retrocede lungo il binario, cambiando la
# distanza apparente dei pali tra uno scatto e l'altro — senza bisogno di
# Camera Actor o Sequencer. Lo spostamento avviene lungo la direzione in cui
# il viewport sta gia' guardando nel momento in cui lanci lo script (quindi
# posiziona bene il viewport PRIMA di lanciare, come hai gia' fatto finora).
MUOVI_CAMERA = True
# NOTA: il viewport parte gia' posizionato al BORDO del terreno esteso —
# arretrare (allontanarsi dal palo) mostrerebbe il vuoto oltre il bordo.
# Per questo lo spostamento e' SOLO IN AVANTI (verso il palo): la frazione
# minima e' 0 (nessun arretramento), non negativa.
FRAZIONE_CAMERA_MIN = 0.0     # 0 = mai piu' indietro del punto di partenza

# --- distanza di inquadratura basata sulla dimensione reale del palo,
# non piu' una frazione fissa della distanza di partenza (vedi spiegazione
# in cima al file). Se disattivato, lo script torna al comportamento V2
# (frazione fissa casuale tra FRAZIONE_CAMERA_MIN e FRAZIONE_CAMERA_MAX).
USA_DISTANZA_TARGET_ASSOLUTA = True

FRAZIONE_CAMERA_MAX = 0.65
# ^ usata SOLO se USA_DISTANZA_TARGET_ASSOLUTA = False (fallback al
# comportamento precedente).

FOV_ORIZZONTALE_GRADI = 90.0
# ^ FOV orizzontale del viewport editor. Se non sei sicuro del valore esatto,
# controllalo in Editor Preferences > Viewports > Field of View (default
# standard dell'editor UE = 90). Un valore sbagliato qui sfalsa la distanza
# target calcolata (FOV piu' stretto → serve stare piu' lontani per la
# stessa inquadratura, e viceversa) — se le immagini generate risultano
# sistematicamente troppo vicine o troppo lontane rispetto a quanto atteso,
# e' il primo parametro da verificare.

ALTEZZA_PALO_MANUALE = None
# <<< DA MODIFICARE: il valore 92990.0 usato nello script principale e'
# specifico dell'asset attuale — NON riusarlo qui invariato. Il nuovo asset
# ha quasi certamente una scala diversa (soprattutto se importato via glTF/
# GLB, dove avete gia' un problema di scala noto nel progetto). Ripeti la
# stessa procedura gia' fatta per l'asset principale: duplica un attore alla
# base e uno in cima al nuovo palo, poi in console Python:
#     import unreal
#     sel = unreal.EditorLevelLibrary.get_selected_level_actors()
#     altezza = abs(sel[0].get_actor_location().z - sel[1].get_actor_location().z)
#     print(f"Altezza misurata: {altezza:.2f} unita' Unreal")
# e incolla qui il valore stampato. Lasciato a None finche' non lo misuri —
# NON lanciare lo script con questo a None per una run vera (vedi warning
# che lo script stampa in log se lo fai comunque, spiegato piu' sotto).
#
# ATTENZIONE: _altezza_attore() (usata se questo resta None) misura il
# bounding box dell'INTERO attore, che potrebbe includere bracci/cavi o
# altre parti della gerarchia oltre al solo montante — stesso problema gia'
# diagnosticato per l'asset principale (braccio della catenaria). Se questo
# nuovo modello ha una struttura ad albero simile (palo + accessori come
# tag "wire" suggerisce), misura SOLO il componente del montante principale,
# non l'intero attore.

FATTORE_MARGINE_INQUADRATURA = 1.8
# ^ Margine di sicurezza sopra il calcolo geometrico "esatto": 1.0
# inquadrerebbe il palo esattamente al bordo del frame verticale (rischioso,
# poco margine per rotazioni/imperfezioni di misura); 1.4 lascia ~40% di
# spazio in piu' sopra/sotto cosi' la base e la sommita' restano ben dentro
# il frame anche con inclinazioni forti vicino ai +/-22 gradi.

VARIAZIONE_DISTANZA_TARGET = 0.15
# ^ Variabilita' (+/- 15%) intorno alla distanza target calcolata, per dare
# un minimo di diversita' di inquadratura tra uno scatto e l'altro invece di
# un'immagine sempre identica in scala.

FRAZIONE_CAMERA_MAX_CAP = 0.90
# ^ tetto di sicurezza assoluto sulla frazione di avvicinamento derivata dal
# calcolo a ritroso, usato come clamp quando la distanza target calcolata
# richiederebbe un avvicinamento estremo (pali molto lontani) — evita
# spostamenti "sparati" oltre questo limite in un colpo solo.

# --- posizione/rotazione di partenza FISSA (opzionale). Se impostate,
# lo script posiziona automaticamente il viewport li' all'avvio di ogni run —
# non serve piu' riposizionarlo a mano ogni volta. Per catturare i valori
# giusti: posiziona il viewport dove vuoi, poi nella console Python:
#     subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
#     loc, rot = subsystem.get_level_viewport_camera_info()
#     print(loc); print(rot)
# e incolla i numeri qui sotto. Se lasciate a None, lo script usa la
# posizione ATTUALE del viewport (comportamento precedente).
POSIZIONE_INIZIALE_FISSA = unreal.Vector(3367.633837, -5981442.977994, 81735.589591)
ROTAZIONE_INIZIALE_FISSA = unreal.Rotator(pitch=1.200100, yaw=-1708.798147, roll=-0.000001)

# --- direzione del binario calcolata dalla geometria REALE, non piu'
# stimata dall'orientamento (impreciso) del viewport. Presa da due segmenti
# di binario duplicati, uno vicino e uno lontano, sulla stessa linea.
PUNTO_BINARIO_A = unreal.Vector(90.0, -3683600.0, 0.0)      # BinarioVerticale_Test10
PUNTO_BINARIO_B = unreal.Vector(0.0, 0.0, 0.0)              # BinarioVerticale_Test
# <<< SCAMBIATI rispetto allo script principale: nella zona del nuovo asset
# la camera guarda in verso opposto lungo lo stesso binario, quindi la
# direzione A->B va invertita (altrimenti la proiezione della distanza
# palo-camera esce negativa e lo script non avvicina mai la camera, come
# verificato nel log del primo test: "-745155 <= ... — nessun avvicinamento").

ANGOLO_MIN = -22
ANGOLO_MAX = 22
NUM_IMMAGINI = 300
# <<< DA MODIFICARE (eventualmente): questo e' un TEST SET, non un dataset
# di training — non serve la stessa numerosita' (2000) del dataset
# principale. Qualche centinaio di immagini bastano per misurare la
# generalizzazione del modello su un asset mai visto in training. Il numero
# esatto non e' critico quanto per il training; alza/abbassa secondo quanto
# tempo hai a disposizione.

CARTELLA_OUTPUT = r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_asset2_test"
# <<< DA MODIFICARE: cartella DIVERSA da quella del dataset principale — la
# separazione fisica e' voluta, cosi' non rischi di mescolare per sbaglio le
# immagini dei due asset nella stessa cartella (e nello stesso split di
# training) quando poi organizzi train/val/test.

FRAMES_ATTESA_RENDER = 8    # frame da aspettare dopo la rotazione, prima di scattare
FRAMES_ATTESA_FILE_MAX = 900  # tetto massimo di frame di attesa per il file (circa 15s a 60fps)

# --- risoluzione FISSA dello screenshot, indipendente da quanto e'
# grande il pannello viewport in quel momento. Puoi usare:
#   - "1920x1080", "2560x1440", "3840x2160" ecc. — risoluzione esplicita
#   - "2" o "4" — moltiplicatore della risoluzione attuale del viewport
# Risoluzioni piu' alte = file piu' pesanti e render leggermente piu' lento
# per scatto, ma qualita' visiva nettamente migliore.
# NOTA: se usi un moltiplicatore ("2"/"4") invece di una risoluzione
# esplicita, l'aspect ratio per il calcolo della distanza target viene
# stimato a 16:9 di default (vedi ASPECT_RATIO_FALLBACK piu' sotto),
# perche' la risoluzione reale dipende dal pannello viewport in quel momento.
RISOLUZIONE_SCREENSHOT = "1920x1080"
ASPECT_RATIO_FALLBACK = 16.0 / 9.0

# --- variazione della temperatura colore della luce (DirectionalLight)
# tra uno scatto e l'altro, per dare varieta' di illuminazione al dataset
# (da tonalita' calda/tramonto a fredda/mezzogiorno). La temperatura e' in
# Kelvin: valori bassi (~3000K) = luce calda/arancione, valori alti
# (~9000K) = luce fredda/bluastra. Un valore "neutro" tipico e' ~6500K.
VARIA_TEMPERATURA_LUCE = True
NOME_DIRECTIONAL_LIGHT = "DirectionalLight"   # nome nell'Outliner
TEMPERATURA_MIN = 3500.0
TEMPERATURA_MAX = 9000.0

# --- disattivazione motion blur per la durata del run ---
# Il motion blur dell'editor sfocava i contorni nelle immagini generate,
# probabilmente in relazione allo spostamento repentino del viewport tra
# uno scatto e l'altro (_sposta_viewport) — un difetto serio per un
# dataset dove servono contorni netti per stimare l'inclinazione.
# Disattivato via console command r.MotionBlurQuality, ripristinato al
# valore originale (assunto = default editor, 4) a fine run.
DISATTIVA_MOTION_BLUR = True
MOTION_BLUR_QUALITY_ORIGINALE = 4
# ^ Valore a cui viene ripristinato r.MotionBlurQuality a fine run. Non e'
# possibile leggere il valore attuale del cvar da Python in modo affidabile
# in questa versione dell'API, quindi va indicato qui a mano SOLO se lo
# avevi cambiato tu manualmente dal default (4) prima di lanciare lo script.

# --- rotazione casuale delle nuvole volumetriche per ogni scatto ---
# Il Volumetric Cloud di default genera le nuvole da un pattern procedurale
# letto dal materiale: non e' geometria interrogabile per sapere a priori
# se una nuvola cade dietro al palo in un dato scatto. Ruotando l'attore
# (yaw casuale) a ogni scatto, il pattern di nuvole scorre rispetto alla
# direzione della camera — non elimina il problema con garanzia assoluta
# (resta casuale), ma evita che lo stesso addensamento resti fisso sempre
# nella stessa posizione dietro al palo per l'intero run.
VARIA_ROTAZIONE_NUVOLE = True
NOME_VOLUMETRIC_CLOUD = "VolumetricCloud"   # nome nell'Outliner — verifica che coincida

# --- salvataggio incrementale delle etichette ---
# Su run lunghe (migliaia di scatti, ore di generazione), scrivere il JSON
# solo alla fine e' rischioso: se l'editor si chiude o il PC si interrompe
# a meta', si perdono TUTTE le etichette anche se le immagini sono gia'
# salvate su disco una per una. Con questo, il file viene sovrascritto ogni
# N scatti, cosi' in caso di interruzione resta comunque il progresso fatto
# fino all'ultimo salvataggio.
SALVA_ETICHETTE_OGNI_N = 50

# ==================== STATO INTERNO (non toccare) ====================

_state = {
    "attori": {},        # {nome: attore} — tutti i pali trovati all'avvio
    "attore_corrente": None,
    "nome_corrente": None,
    "indice": 0,
    "fase": "avvio",
    "angolo_corrente": None,
    "etichette": [],
    "falliti": 0,
    "frame_contatore": 0,
    "cartella_saved": None,
    "prima_set": None,
    "handle": None,
    "camera_location_base": None,
    "camera_rotation_base": None,
    "direzione_binario": None,
    "componente_luce": None,
    "temperatura_originale": None,
    "use_temperature_originale": None,
    "temperatura_corrente": None,
    "distanza_target_base": None,
    "altezza_palo_misurata": None,
    "distanza_lungo_binario_corrente": None,
    "distanza_target_corrente": None,
    "frazione_corrente": None,
    "attore_nuvole": None,
    "rotazione_nuvole_originale": None,
}


def _trova_attore_per_nome(nome_label):
    tutti = unreal.EditorLevelLibrary.get_all_level_actors()
    for attore in tutti:
        if attore.get_actor_label() == nome_label:
            return attore
    return None


def _trova_componente_luce_direzionale():
    """Trova l'attore DirectionalLight e ritorna il suo componente luce
    (dove vivono le proprieta' come temperature, intensity, ecc.)."""
    attore = _trova_attore_per_nome(NOME_DIRECTIONAL_LIGHT)
    if attore is None:
        return None
    componente = attore.get_component_by_class(unreal.DirectionalLightComponent)
    return componente


def _imposta_rotazione(attore, angolo_gradi):
    if ASSE_ROTAZIONE == "pitch":
        rot = unreal.Rotator(roll=0.0, pitch=angolo_gradi, yaw=0.0)
    elif ASSE_ROTAZIONE == "roll":
        rot = unreal.Rotator(roll=angolo_gradi, pitch=0.0, yaw=0.0)
    else:
        rot = unreal.Rotator(roll=0.0, pitch=0.0, yaw=angolo_gradi)
    attore.set_actor_relative_rotation(rot, False, False)


def _distanza(a, b):
    """Distanza euclidea tra due unreal.Vector, calcolata a mano
    (in questa versione dell'API, Vector non ha un metodo .size()/.length())."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _direzione_da_due_punti(a, b):
    """Direzione ORIZZONTALE normalizzata (Z=0) da un punto a un altro —
    usata per ricavare la direzione del binario dalla geometria REALE
    (due segmenti di binario), invece che dall'orientamento del viewport
    (che puo' essere leggermente disallineato)."""
    dx = b.x - a.x
    dy = b.y - a.y
    lunghezza = (dx * dx + dy * dy) ** 0.5
    if lunghezza < 0.0001:
        return unreal.Vector(0.0, 0.0, 0.0)
    return unreal.Vector(dx / lunghezza, dy / lunghezza, 0.0)


def _direzione_orizzontale_da_rotazione(rotation):
    """Vettore direzione ORIZZONTALE (normalizzato, Z=0) nella direzione in
    cui punta 'rotation'. E' la direzione FISSA del binario (calcolata una
    volta sola dalla rotazione iniziale del viewport) — la camera si muove
    SEMPRE lungo questa linea, mai lateralmente verso un palo specifico."""
    forward = rotation.get_forward_vector()
    lunghezza = (forward.x ** 2 + forward.y ** 2) ** 0.5
    if lunghezza < 0.0001:
        return unreal.Vector(0.0, 0.0, 0.0)
    return unreal.Vector(forward.x / lunghezza, forward.y / lunghezza, 0.0)


def _distanza_lungo_direzione(da, verso, direzione_normalizzata):
    """Quanta strada c'e' DA PERCORRERE LUNGO IL BINARIO (proiezione, non
    distanza in linea retta) per essere alla stessa 'profondita'' del punto
    'verso', partendo da 'da'. Usata per calibrare l'avvicinamento sul palo
    scelto anche se e' leggermente fuori asse rispetto al binario — senza
    questo, un palo spostato lateralmente falserebbe la percentuale se si
    usasse la distanza in linea retta."""
    dx = verso.x - da.x
    dy = verso.y - da.y
    return dx * direzione_normalizzata.x + dy * direzione_normalizzata.y


def _cattura_posizione_viewport():
    """Legge posizione/rotazione ATTUALI del viewport, da usare come baseline
    per gli spostamenti successivi (avanti/indietro lungo lo sguardo)."""
    subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    location, rotation = subsystem.get_level_viewport_camera_info()
    return location, rotation


def _sposta_viewport(location_base, rotation_base, direzione_orizzontale, offset_avanti):
    """Sposta il viewport lungo la direzione ORIZZONTALE specificata (verso
    il palo scelto per questo scatto), di offset_avanti unita'. La camera
    resta sempre alla stessa altezza (Z invariata) e mantiene lo sguardo
    originale (rotation_base non cambia) — cambia solo la posizione."""
    subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    nuova_location = location_base + direzione_orizzontale * offset_avanti
    subsystem.set_level_viewport_camera_info(nuova_location, rotation_base)
    unreal.log(f"   [debug camera] offset applicato: {offset_avanti:.0f} unita' "
              f"(direzione verso il palo scelto)")


def _altezza_attore(attore):
    """Altezza (asse Z) del bounding box dell'attore, in unita' del mondo.
    Va misurata mentre il palo e' ancora a rotazione 0 (dritto) — chiamata
    quindi in fase 'avvio', prima di qualunque rotazione random."""
    origine, estensione = attore.get_actor_bounds(only_colliding_components=False)
    return estensione.z * 2.0


def _aspect_ratio_da_config():
    """Ricava l'aspect ratio (larghezza/altezza) dalla stringa di
    risoluzione configurata, se e' una risoluzione esplicita del tipo
    'LARGHEZZAxALTEZZA'. Se e' invece un moltiplicatore ('2', '4') non c'e'
    modo di saperlo a priori (dipende dal pannello viewport in quel
    momento) — si usa ASPECT_RATIO_FALLBACK."""
    valore = RISOLUZIONE_SCREENSHOT.lower().strip()
    if "x" in valore:
        try:
            larghezza_str, altezza_str = valore.split("x")
            larghezza = float(larghezza_str)
            altezza = float(altezza_str)
            if larghezza > 0 and altezza > 0:
                return larghezza / altezza
        except Exception:
            pass
    return ASPECT_RATIO_FALLBACK


def _calcola_distanza_target(altezza_palo, aspect_ratio):
    """Distanza assoluta alla quale l'intero palo (altezza 'altezza_palo')
    occupa il frame verticale della camera, con il margine di sicurezza
    configurato. Formula: altezza / (2 * tan(FOV_verticale / 2)), con
    FOV_verticale derivato dal FOV orizzontale e dall'aspect ratio."""
    fov_o_rad = math.radians(FOV_ORIZZONTALE_GRADI)
    fov_v_rad = 2.0 * math.atan(math.tan(fov_o_rad / 2.0) / aspect_ratio)
    distanza_esatta = altezza_palo / (2.0 * math.tan(fov_v_rad / 2.0))
    return distanza_esatta * FATTORE_MARGINE_INQUADRATURA


def _tick(delta_time):
    """Richiamata dal motore a ogni frame reale — qui vive tutta la logica."""
    s = _state
    fase = s["fase"]

    if fase == "avvio":
        # --- pulizia automatica della cartella di output ---
        if os.path.exists(CARTELLA_OUTPUT):
            shutil.rmtree(CARTELLA_OUTPUT)
            unreal.log(f"Cartella di output pulita: {CARTELLA_OUTPUT}")

        trovati = []
        non_trovati = []
        for nome in NOMI_PIVOT:
            attore = _trova_attore_per_nome(nome)
            if attore is None:
                non_trovati.append(nome)
                continue
            s["attori"][nome] = attore
            trovati.append(nome)
            _imposta_rotazione(attore, 0.0)  # assicura che parta dritto

        # --- riepilogo ESPLICITO, impossibile perderselo nel log ---
        unreal.log("=" * 60)
        unreal.log(f"PALI TROVATI ({len(trovati)}/{len(NOMI_PIVOT)}): {trovati}")
        if non_trovati:
            unreal.log_error(f"PALI NON TROVATI (controlla i nomi esatti "
                            f"nell'Outliner!): {non_trovati}")
        unreal.log("=" * 60)

        if not s["attori"]:
            unreal.log_error("Nessun palo trovato tra quelli elencati in NOMI_PIVOT! Controlla i nomi.")
            s["fase"] = "fine"
            return

        # --- misura l'altezza del palo e calcola la distanza target ---
        # (fatto qui, con tutti i pali ancora a rotazione 0, cosi' il
        # bounding box misurato e' quello "a riposo" e non e' inquinato da
        # un'eventuale inclinazione residua da un run precedente)
        if USA_DISTANZA_TARGET_ASSOLUTA:
            if ALTEZZA_PALO_MANUALE is not None:
                altezza_palo = ALTEZZA_PALO_MANUALE
                fonte_altezza = "ALTEZZA_PALO_MANUALE (override manuale)"
            else:
                primo_nome = trovati[0]
                primo_attore = s["attori"][primo_nome]
                altezza_palo = _altezza_attore(primo_attore)
                fonte_altezza = f"bounding box intero attore '{primo_nome}'"
                unreal.log_warning(
                    "[distanza target] ALTEZZA_PALO_MANUALE non impostata: uso il "
                    "bounding box dell'intero attore, che include anche il braccio "
                    "della catenaria (ParteVolantePalo_Test) e probabilmente "
                    "SOVRASTIMA l'altezza reale del palo — la camera potrebbe restare "
                    "piu' lontana del necessario. Misura l'altezza del solo componente "
                    "Pole_Test nel Details panel e impostala in ALTEZZA_PALO_MANUALE "
                    "per una stima corretta."
                )

            aspect_ratio = _aspect_ratio_da_config()
            distanza_target = _calcola_distanza_target(altezza_palo, aspect_ratio)

            s["altezza_palo_misurata"] = altezza_palo
            s["distanza_target_base"] = distanza_target

            unreal.log(f"[distanza target] altezza palo usata ({fonte_altezza}): "
                      f"{altezza_palo:.1f} unita'")
            unreal.log(f"[distanza target] aspect ratio usato: {aspect_ratio:.3f} "
                      f"(FOV orizzontale {FOV_ORIZZONTALE_GRADI:.0f}°)")
            unreal.log(f"[distanza target] distanza di inquadratura calcolata: "
                      f"{distanza_target:.0f} unita' (margine {FATTORE_MARGINE_INQUADRATURA:.2f}x)")

        # --- setup variazione temperatura luce ---
        if VARIA_TEMPERATURA_LUCE:
            s["componente_luce"] = _trova_componente_luce_direzionale()
            if s["componente_luce"] is not None:
                s["temperatura_originale"] = s["componente_luce"].get_editor_property("temperature")
                s["use_temperature_originale"] = s["componente_luce"].get_editor_property("use_temperature")
                s["componente_luce"].set_editor_property("use_temperature", True)
                unreal.log(f"Variazione temperatura luce attiva ({TEMPERATURA_MIN:.0f}K-{TEMPERATURA_MAX:.0f}K). "
                          f"Valore originale salvato: {s['temperatura_originale']:.0f}K")
            else:
                unreal.log_warning(f"DirectionalLight '{NOME_DIRECTIONAL_LIGHT}' non trovato — "
                                  f"variazione temperatura disattivata per questo run.")

        os.makedirs(CARTELLA_OUTPUT, exist_ok=True)
        s["cartella_saved"] = os.path.join(
            unreal.SystemLibrary.get_project_saved_directory(), "Screenshots"
        )

        # --- disattivazione motion blur ---
        if DISATTIVA_MOTION_BLUR:
            unreal.SystemLibrary.execute_console_command(
                unreal.EditorLevelLibrary.get_editor_world(), "r.MotionBlurQuality 0"
            )
            unreal.log("Motion blur disattivato (r.MotionBlurQuality 0) per la durata del run.")

        # --- setup rotazione casuale nuvole ---
        if VARIA_ROTAZIONE_NUVOLE:
            s["attore_nuvole"] = _trova_attore_per_nome(NOME_VOLUMETRIC_CLOUD)
            if s["attore_nuvole"] is not None:
                s["rotazione_nuvole_originale"] = s["attore_nuvole"].get_actor_rotation()
                unreal.log(f"Rotazione nuvole casuale attiva (attore '{NOME_VOLUMETRIC_CLOUD}' trovato). "
                          f"Rotazione originale salvata: {s['rotazione_nuvole_originale']}")
            else:
                unreal.log_warning(f"Attore nuvole '{NOME_VOLUMETRIC_CLOUD}' non trovato — "
                                  f"controlla il nome nell'Outliner. Rotazione nuvole disattivata "
                                  f"per questo run.")

        if MUOVI_CAMERA:
            if POSIZIONE_INIZIALE_FISSA is not None and ROTAZIONE_INIZIALE_FISSA is not None:
                # posiziona automaticamente il viewport al punto fisso configurato
                subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
                subsystem.set_level_viewport_camera_info(
                    POSIZIONE_INIZIALE_FISSA, ROTAZIONE_INIZIALE_FISSA
                )
                s["camera_location_base"] = POSIZIONE_INIZIALE_FISSA
                s["camera_rotation_base"] = ROTAZIONE_INIZIALE_FISSA
                unreal.log("Viewport posizionato automaticamente alla posizione fissa configurata.")
            else:
                s["camera_location_base"], s["camera_rotation_base"] = _cattura_posizione_viewport()
                unreal.log("Nessuna posizione fissa configurata — uso la posizione "
                          "ATTUALE del viewport (posizionalo bene prima di lanciare).")

            s["direzione_binario"] = _direzione_da_due_punti(PUNTO_BINARIO_A, PUNTO_BINARIO_B)
            unreal.log(f"Direzione binario (da geometria reale): {s['direzione_binario']}")

        unreal.log(f"=== Avvio generazione: {NUM_IMMAGINI} immagini, "
                  f"{len(s['attori'])} pali disponibili, "
                  f"range {ANGOLO_MIN}/{ANGOLO_MAX} gradi, asse={ASSE_ROTAZIONE} ===")
        s["fase"] = "imposta_rotazione"
        return

    if fase == "imposta_rotazione":
        if s["indice"] >= NUM_IMMAGINI:
            s["fase"] = "fine"
            return
        # scegli un palo a caso tra quelli disponibili per questo scatto
        nome_scelto = random.choice(list(s["attori"].keys()))
        s["nome_corrente"] = nome_scelto
        s["attore_corrente"] = s["attori"][nome_scelto]

        angolo = round(random.uniform(ANGOLO_MIN, ANGOLO_MAX), 1)
        s["angolo_corrente"] = angolo
        _imposta_rotazione(s["attore_corrente"], angolo)

        # varia la temperatura colore della luce per questo scatto
        if VARIA_TEMPERATURA_LUCE and s["componente_luce"] is not None:
            temperatura = round(random.uniform(TEMPERATURA_MIN, TEMPERATURA_MAX), 0)
            s["componente_luce"].set_editor_property("temperature", temperatura)
            s["temperatura_corrente"] = temperatura
        else:
            s["temperatura_corrente"] = None

        # ruota le nuvole a caso per questo scatto, cosi' lo stesso
        # addensamento non resta sempre nella stessa posizione dietro al palo
        if VARIA_ROTAZIONE_NUVOLE and s["attore_nuvole"] is not None:
            yaw_casuale = random.uniform(0.0, 360.0)
            nuova_rotazione_nuvole = unreal.Rotator(
                roll=s["rotazione_nuvole_originale"].roll,
                pitch=s["rotazione_nuvole_originale"].pitch,
                yaw=yaw_casuale,
            )
            s["attore_nuvole"].set_actor_rotation(nuova_rotazione_nuvole, False)

        # sposta il viewport LUNGO IL BINARIO (direzione fissa, mai
        # lateralmente), avvicinandosi in proporzione a quanta strada
        # separa la camera dalla "profondita'" del palo scelto
        s["distanza_lungo_binario_corrente"] = None
        s["distanza_target_corrente"] = None
        s["frazione_corrente"] = None

        if MUOVI_CAMERA and s["camera_location_base"] is not None:
            posizione_palo = s["attore_corrente"].get_actor_location()
            distanza_lungo_binario = _distanza_lungo_direzione(
                s["camera_location_base"], posizione_palo, s["direzione_binario"]
            )
            s["distanza_lungo_binario_corrente"] = distanza_lungo_binario

            if USA_DISTANZA_TARGET_ASSOLUTA and s["distanza_target_base"] is not None:
                variazione = random.uniform(
                    1.0 - VARIAZIONE_DISTANZA_TARGET, 1.0 + VARIAZIONE_DISTANZA_TARGET
                )
                distanza_target = s["distanza_target_base"] * variazione
                s["distanza_target_corrente"] = distanza_target

                if distanza_lungo_binario <= distanza_target:
                    # il palo e' gia' piu' vicino (o uguale) del target:
                    # nessun avvicinamento ulteriore (niente arretramento,
                    # vedi nota su FRAZIONE_CAMERA_MIN)
                    frazione = 0.0
                    unreal.log(f"   [debug camera] '{nome_scelto}' gia' entro la distanza target "
                              f"({distanza_lungo_binario:.0f} <= {distanza_target:.0f} unita') "
                              f"— nessun avvicinamento")
                else:
                    frazione = 1.0 - (distanza_target / distanza_lungo_binario)
                    frazione = max(FRAZIONE_CAMERA_MIN, min(frazione, FRAZIONE_CAMERA_MAX_CAP))
                    unreal.log(f"   [debug camera] '{nome_scelto}': partenza={distanza_lungo_binario:.0f}, "
                              f"target={distanza_target:.0f}, frazione derivata={frazione:.3f}")
            else:
                frazione = random.uniform(FRAZIONE_CAMERA_MIN, FRAZIONE_CAMERA_MAX)

            s["frazione_corrente"] = frazione
            offset = distanza_lungo_binario * frazione
            offset = max(0.0, offset)  # clamp difensivo: mai in negativo

            _sposta_viewport(s["camera_location_base"], s["camera_rotation_base"],
                            s["direzione_binario"], offset)

        s["frame_contatore"] = 0
        s["fase"] = "attesa_render"
        return

    if fase == "attesa_render":
        # aspetta qualche frame REALE (non bloccante) perche' il viewport
        # si aggiorni visivamente dopo la rotazione, prima di scattare
        s["frame_contatore"] += 1
        if s["frame_contatore"] >= FRAMES_ATTESA_RENDER:
            s["prima_set"] = set(glob.glob(
                os.path.join(s["cartella_saved"], "**", "*.png"), recursive=True
            ))
            unreal.SystemLibrary.execute_console_command(
                unreal.EditorLevelLibrary.get_editor_world(), f"HighResShot {RISOLUZIONE_SCREENSHOT}"
            )
            s["frame_contatore"] = 0
            s["fase"] = "attesa_file"
        return

    if fase == "attesa_file":
        s["frame_contatore"] += 1
        dopo = set(glob.glob(
            os.path.join(s["cartella_saved"], "**", "*.png"), recursive=True
        ))
        nuovi = dopo - s["prima_set"]

        if nuovi:
            percorso = list(nuovi)[0]
            i = s["indice"]
            # NOTA sul segno: la rotazione 3D applicata al palo (s['angolo_corrente'])
            # resta quella random generata sopra — non importa il verso fisico
            # applicato in Unreal, quello che conta e' che l'ETICHETTA SALVATA
            # rappresenti in modo coerente l'inclinazione come appare sullo
            # schermo. La camera in questa zona guarda in verso quasi opposto
            # lungo il binario rispetto al dataset originale (stesso motivo per
            # cui PUNTO_BINARIO_A/B sono scambiati qui sopra): questo inverte
            # sinistra/destra sullo schermo a parita' di rotazione 3D applicata,
            # quindi l'etichetta salvata va negata per restare coerente con la
            # convenzione del dataset originale (confermato empiricamente: senza
            # questa correzione, il modello addestrato sul dataset originale
            # indovinava il segno giusto solo nell'8% dei casi su questo asset).
            angolo = -s["angolo_corrente"]
            nome_file = f"palo_{i:04d}_{s['nome_corrente']}_angolo_{angolo:+06.1f}.png"
            destinazione = os.path.join(CARTELLA_OUTPUT, nome_file)
            try:
                # piccola attesa di sicurezza: il file potrebbe essere ancora
                # in fase di scrittura su disco nell'istante in cui compare
                time.sleep(0.1)
                shutil.move(percorso, destinazione)
                s["etichette"].append({
                    "file": nome_file,
                    "palo": s["nome_corrente"],
                    "angolo_gradi": angolo,
                    "temperatura_luce_K": s["temperatura_corrente"],
                    "distanza_lungo_binario_iniziale": s["distanza_lungo_binario_corrente"],
                    "distanza_target": s["distanza_target_corrente"],
                    "frazione_applicata": s["frazione_corrente"],
                })
                unreal.log(f"[{i+1}/{NUM_IMMAGINI}] palo={s['nome_corrente']} "
                          f"angolo={angolo:+.1f}° -> {nome_file}")

                # salvataggio incrementale: sovrascrive il JSON ogni N
                # scatti, cosi' un'interruzione a meta' run non fa perdere
                # tutte le etichette gia' raccolte
                if SALVA_ETICHETTE_OGNI_N > 0 and len(s["etichette"]) % SALVA_ETICHETTE_OGNI_N == 0:
                    percorso_json_parziale = os.path.join(CARTELLA_OUTPUT, "etichette.json")
                    with open(percorso_json_parziale, "w") as f:
                        json.dump(s["etichette"], f, indent=2)
                    unreal.log(f"   [salvataggio incrementale] etichette.json aggiornato "
                              f"({len(s['etichette'])} record)")
            except Exception as e:
                unreal.log_warning(f"[{i+1}/{NUM_IMMAGINI}] Errore spostando il file: {e}")
                s["falliti"] += 1

            # IMPORTANTE: riporta questo palo a 0 gradi subito dopo lo scatto,
            # cosi' negli scatti successivi resta dritto — altrimenti i pali
            # gia' usati resterebbero storti sullo sfondo, mostrando piu' di
            # un'anomalia nella stessa immagine (contraddirebbe l'etichetta
            # singola associata a ciascun file).
            _imposta_rotazione(s["attore_corrente"], 0.0)

            s["indice"] += 1
            s["fase"] = "imposta_rotazione"

        elif s["frame_contatore"] >= FRAMES_ATTESA_FILE_MAX:
            unreal.log_warning(
                f"[{s['indice']+1}/{NUM_IMMAGINI}] Screenshot non trovato entro il "
                f"limite di frame — salto. Se succede spesso, alza FRAMES_ATTESA_FILE_MAX."
            )
            s["falliti"] += 1
            s["indice"] += 1
            s["fase"] = "imposta_rotazione"
        return

    if fase == "fine":
        percorso_json = os.path.join(CARTELLA_OUTPUT, "etichette.json")
        with open(percorso_json, "w") as f:
            json.dump(s["etichette"], f, indent=2)

        for attore in s["attori"].values():
            _imposta_rotazione(attore, 0.0)  # assicura che tutti i pali finiscano dritti

        if MUOVI_CAMERA and s["camera_location_base"] is not None:
            subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
            subsystem.set_level_viewport_camera_info(
                s["camera_location_base"], s["camera_rotation_base"]
            )
            unreal.log("Viewport riportato alla posizione di partenza.")

        if VARIA_TEMPERATURA_LUCE and s["componente_luce"] is not None:
            s["componente_luce"].set_editor_property("temperature", s["temperatura_originale"])
            s["componente_luce"].set_editor_property("use_temperature", s["use_temperature_originale"])
            unreal.log(f"Temperatura luce ripristinata al valore originale ({s['temperatura_originale']:.0f}K).")

        if DISATTIVA_MOTION_BLUR:
            unreal.SystemLibrary.execute_console_command(
                unreal.EditorLevelLibrary.get_editor_world(),
                f"r.MotionBlurQuality {MOTION_BLUR_QUALITY_ORIGINALE}"
            )
            unreal.log(f"Motion blur ripristinato (r.MotionBlurQuality {MOTION_BLUR_QUALITY_ORIGINALE}).")

        if VARIA_ROTAZIONE_NUVOLE and s["attore_nuvole"] is not None:
            s["attore_nuvole"].set_actor_rotation(s["rotazione_nuvole_originale"], False)
            unreal.log("Rotazione nuvole ripristinata al valore originale.")

        unreal.log(f"=== FATTO: {len(s['etichette'])} immagini generate "
                  f"({s['falliti']} fallite) in {CARTELLA_OUTPUT} ===")
        unreal.log(f"Etichette salvate in: {percorso_json}")

        unreal.unregister_slate_post_tick_callback(s["handle"])
        s["fase"] = "terminato"
        return


# --- avvio: registra il tick callback e ritorna subito il controllo ---
_state["handle"] = unreal.register_slate_post_tick_callback(_tick)
unreal.log("Generazione avviata in background (un frame alla volta). "
          "Controlla l'Output Log per i progressi. Non toccare il viewport.")