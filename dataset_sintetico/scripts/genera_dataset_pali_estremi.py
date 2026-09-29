# -*- coding: utf-8 -*-
"""
Generazione dataset sintetico — pali inclinati (Unreal Engine 5.7).
VERSIONE "ESTREMI" — batch mirato solo su inclinazioni |angolo| >= 15 gradi,
per rinforzare la fascia dove il modello attuale e' piu' debole (MAE 3.14°
su 15-22° contro 0.63-1.00° sulle fasce piu' basse).

MESCOLA PALI DI DUE ASSET DIVERSI nella stessa scena/scatto: alcuni pivot
sono duplicati dell'asset originale, altri dell'asset3 — tutti posizionati
vicini nello stesso punto della scena, un solo POSIZIONE_INIZIALE_FISSA per
tutti. A differenza degli script precedenti, qui ALTEZZA_PALO_MANUALE (un
singolo valore) NON basta piu': i due asset hanno altezze diverse (92990 per
l'originale, 117040 per l'asset3), quindi la distanza di inquadratura deve
essere calcolata SEPARATAMENTE per ciascun pivot, in base al suo asset.
Vedi ALTEZZE_PALI qui sotto (dizionario nome_pivot -> altezza), che
sostituisce ALTEZZA_PALO_MANUALE.

CAMPIONAMENTO ANGOLO CON SOGLIA MINIMA: invece di un range continuo
-22/+22, qui si scarta (rejection sampling) qualunque angolo con
|angolo| < ANGOLO_MODULO_MIN, ripetendo l'estrazione finche' non esce un
valore nella fascia voluta. Il segno resta casuale (positivo o negativo).

QUESTO DATASET VA SOMMATO al training esistente (dataset_originale_crop +
dataset_asset3_crop), non li sostituisce — l'obiettivo e' oversampling
della fascia estrema, non un nuovo dataset a se stante.

COME LANCIARLO: identico agli altri, nella riga di comando Python
dell'editor:
    exec(open(r"C:\percorso\genera_dataset_pali_estremi.py").read())
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

NOMI_PIVOT = ["PolePivot_Base5", "PolePivot_Base6",
              "Electric_Pole4", "Electric_Pole5"]
# <<< DA MODIFICARE: nomi esatti (label Outliner) dei 4 (o quanti ne
# duplichi) pivot piazzati per questo batch — mix di duplicati dell'asset
# originale e dell'asset3, tutti vicini nella stessa zona della scena.

ALTEZZE_PALI = {
    "PolePivot_Base5": 92990.0,        # duplicato asset originale
    "PolePivot_Base6": 92990.0,        # duplicato asset originale
    "Electric_Pole4": 117040.0,   # duplicato asset3
    "Electric_Pole5": 117040.0,   # duplicato asset3
}
# <<< DA MODIFICARE: ogni pivot elencato in NOMI_PIVOT DEVE avere una voce
# qui con l'altezza corretta del SUO asset (92990.0 per i duplicati
# dell'asset originale, 117040.0 per i duplicati dell'asset3) — sono i
# valori gia' misurati in precedenza, non serve rimisurarli se stai
# duplicando gli stessi identici asset. Se aggiungi pivot di un asset
# diverso da questi due, misuragli l'altezza con la solita procedura prima
# di aggiungerlo qui.

ASSE_ROTAZIONE = "pitch"   # cambia in "roll" se l'inclinazione visiva è sbagliata

MUOVI_CAMERA = True
FRAZIONE_CAMERA_MIN = 0.0

USA_DISTANZA_TARGET_ASSOLUTA = True
FRAZIONE_CAMERA_MAX = 0.65   # usata SOLO se USA_DISTANZA_TARGET_ASSOLUTA = False

FOV_ORIZZONTALE_GRADI = 90.0

FATTORE_MARGINE_INQUADRATURA = 1.8
VARIAZIONE_DISTANZA_TARGET = 0.15
FRAZIONE_CAMERA_MAX_CAP = 0.90

# --- posizione/rotazione di partenza FISSA. Cattura con lo stesso comando
# gia' usato negli altri script:
#     subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
#     loc, rot = subsystem.get_level_viewport_camera_info()
#     print(loc); print(rot)
POSIZIONE_INIZIALE_FISSA = unreal.Vector(-479.380163, -3184983.712027, 52162.036756)
ROTAZIONE_INIZIALE_FISSA = unreal.Rotator(pitch=6.200100, yaw=-1170.398096, roll=-0.000001)
# <<< DA MODIFICARE: incolla qui i valori catturati. Finche' restano None,
# lo script usa la posizione ATTUALE del viewport al lancio.

# --- direzione del binario: stessa retta di sempre. VERIFICA COMUNQUE con
# il batch di test piccolo (vedi ANGOLO_MODULO_MIN sotto) prima di un batch
# grande — stessa euristica gia' spiegata negli script asset2/asset3: se
# "partenza" nel log viene negativo, scambia questi due punti.
PUNTO_BINARIO_A = unreal.Vector(0.0, 0.0, 0.0)              # BinarioVerticale_Test
PUNTO_BINARIO_B = unreal.Vector(90.0, -3683600.0, 0.0)      # BinarioVerticale_Test10

INVERTI_SEGNO_ETICHETTA = False
# <<< DA VERIFICARE: stessa logica gia' vista per asset2/asset3 — attiva
# SOLO se il batch di test rivela l'inversione sinistra/destra.

ANGOLO_MIN = -22
ANGOLO_MAX = 22
ANGOLO_MODULO_MIN = 15.0
# <<< Soglia minima di |angolo|: qualunque candidato con |angolo| inferiore
# viene scartato e ri-estratto (rejection sampling), finche' non esce un
# valore nella fascia [15,22] o [-22,-15]. Il segno resta casuale.

NUM_IMMAGINI = 450
# <<< Batch di oversampling mirato — non serve la numerosita' di un
# dataset "base", questo va solo a rinforzare la coda della distribuzione
# gia' esistente.

CARTELLA_OUTPUT = r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_estremi"

FRAMES_ATTESA_RENDER = 8
FRAMES_ATTESA_FILE_MAX = 900

RISOLUZIONE_SCREENSHOT = "1920x1080"
ASPECT_RATIO_FALLBACK = 16.0 / 9.0

VARIA_TEMPERATURA_LUCE = True
NOME_DIRECTIONAL_LIGHT = "DirectionalLight"
TEMPERATURA_MIN = 3500.0
TEMPERATURA_MAX = 9000.0

DISATTIVA_MOTION_BLUR = True
MOTION_BLUR_QUALITY_ORIGINALE = 4

VARIA_ROTAZIONE_NUVOLE = True
NOME_VOLUMETRIC_CLOUD = "VolumetricCloud"

SALVA_ETICHETTE_OGNI_N = 50

# ==================== STATO INTERNO (non toccare) ====================

_state = {
    "attori": {},
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
    "distanza_target_base_per_palo": {},   # <<< dizionario, non piu' un singolo valore
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
    attore = _trova_attore_per_nome(NOME_DIRECTIONAL_LIGHT)
    if attore is None:
        return None
    return attore.get_component_by_class(unreal.DirectionalLightComponent)


def _imposta_rotazione(attore, angolo_gradi):
    if ASSE_ROTAZIONE == "pitch":
        rot = unreal.Rotator(roll=0.0, pitch=angolo_gradi, yaw=0.0)
    elif ASSE_ROTAZIONE == "roll":
        rot = unreal.Rotator(roll=angolo_gradi, pitch=0.0, yaw=0.0)
    else:
        rot = unreal.Rotator(roll=0.0, pitch=0.0, yaw=angolo_gradi)
    attore.set_actor_relative_rotation(rot, False, False)


def _campiona_angolo_estremo():
    """Rejection sampling: ripete l'estrazione uniforme su [ANGOLO_MIN,
    ANGOLO_MAX] finche' non esce un valore con |angolo| >= ANGOLO_MODULO_MIN.
    Il numero di tentativi e' irrilevante in pratica (poche iterazioni in
    media, dato che la fascia accettata e' comunque una frazione consistente
    del range totale), ma un tetto di sicurezza evita loop infiniti in caso
    di configurazione errata (es. ANGOLO_MODULO_MIN > ANGOLO_MAX)."""
    for _ in range(1000):
        candidato = round(random.uniform(ANGOLO_MIN, ANGOLO_MAX), 1)
        if abs(candidato) >= ANGOLO_MODULO_MIN:
            return candidato
    unreal.log_error(f"_campiona_angolo_estremo: nessun valore valido trovato in 1000 "
                      f"tentativi — controlla ANGOLO_MODULO_MIN ({ANGOLO_MODULO_MIN}) "
                      f"rispetto a ANGOLO_MIN/MAX ({ANGOLO_MIN}/{ANGOLO_MAX}).")
    return ANGOLO_MAX  # fallback estremo, non dovrebbe mai essere raggiunto


def _direzione_da_due_punti(a, b):
    dx = b.x - a.x
    dy = b.y - a.y
    lunghezza = (dx * dx + dy * dy) ** 0.5
    if lunghezza < 0.0001:
        return unreal.Vector(0.0, 0.0, 0.0)
    return unreal.Vector(dx / lunghezza, dy / lunghezza, 0.0)


def _distanza_lungo_direzione(da, verso, direzione_normalizzata):
    dx = verso.x - da.x
    dy = verso.y - da.y
    return dx * direzione_normalizzata.x + dy * direzione_normalizzata.y


def _cattura_posizione_viewport():
    subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    location, rotation = subsystem.get_level_viewport_camera_info()
    return location, rotation


def _sposta_viewport(location_base, rotation_base, direzione_orizzontale, offset_avanti):
    subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    nuova_location = location_base + direzione_orizzontale * offset_avanti
    subsystem.set_level_viewport_camera_info(nuova_location, rotation_base)
    unreal.log(f"   [debug camera] offset applicato: {offset_avanti:.0f} unita'")


def _aspect_ratio_da_config():
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
    fov_o_rad = math.radians(FOV_ORIZZONTALE_GRADI)
    fov_v_rad = 2.0 * math.atan(math.tan(fov_o_rad / 2.0) / aspect_ratio)
    distanza_esatta = altezza_palo / (2.0 * math.tan(fov_v_rad / 2.0))
    return distanza_esatta * FATTORE_MARGINE_INQUADRATURA


def _tick(delta_time):
    s = _state
    fase = s["fase"]

    if fase == "avvio":
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
            _imposta_rotazione(attore, 0.0)

        unreal.log("=" * 60)
        unreal.log(f"PALI TROVATI ({len(trovati)}/{len(NOMI_PIVOT)}): {trovati}")
        if non_trovati:
            unreal.log_error(f"PALI NON TROVATI: {non_trovati}")
        unreal.log("=" * 60)

        if not s["attori"]:
            unreal.log_error("Nessun palo trovato! Controlla i nomi in NOMI_PIVOT.")
            s["fase"] = "fine"
            return

        # --- verifica che ogni pivot trovato abbia un'altezza in ALTEZZE_PALI ---
        mancanti_altezza = [nome for nome in trovati if nome not in ALTEZZE_PALI]
        if mancanti_altezza:
            unreal.log_error(
                f"ALTEZZE_PALI non ha una voce per: {mancanti_altezza} — "
                f"la distanza di inquadratura per questi pali non puo' essere calcolata "
                f"correttamente. Aggiungi la loro altezza al dizionario prima di procedere."
            )

        # --- calcola la distanza target SEPARATAMENTE per ogni pivot,
        # usando la sua altezza specifica (asset diversi = altezze diverse) ---
        if USA_DISTANZA_TARGET_ASSOLUTA:
            aspect_ratio = _aspect_ratio_da_config()
            for nome in trovati:
                altezza_palo = ALTEZZE_PALI.get(nome)
                if altezza_palo is None:
                    continue  # gia' segnalato sopra come errore
                distanza_target = _calcola_distanza_target(altezza_palo, aspect_ratio)
                s["distanza_target_base_per_palo"][nome] = distanza_target
                unreal.log(f"[distanza target] '{nome}': altezza={altezza_palo:.0f}, "
                          f"target={distanza_target:.0f} unita'")

        if VARIA_TEMPERATURA_LUCE:
            s["componente_luce"] = _trova_componente_luce_direzionale()
            if s["componente_luce"] is not None:
                s["temperatura_originale"] = s["componente_luce"].get_editor_property("temperature")
                s["use_temperature_originale"] = s["componente_luce"].get_editor_property("use_temperature")
                s["componente_luce"].set_editor_property("use_temperature", True)
                unreal.log(f"Variazione temperatura luce attiva. "
                          f"Valore originale salvato: {s['temperatura_originale']:.0f}K")
            else:
                unreal.log_warning(f"DirectionalLight '{NOME_DIRECTIONAL_LIGHT}' non trovato.")

        os.makedirs(CARTELLA_OUTPUT, exist_ok=True)
        s["cartella_saved"] = os.path.join(
            unreal.SystemLibrary.get_project_saved_directory(), "Screenshots"
        )

        if DISATTIVA_MOTION_BLUR:
            unreal.SystemLibrary.execute_console_command(
                unreal.EditorLevelLibrary.get_editor_world(), "r.MotionBlurQuality 0"
            )
            unreal.log("Motion blur disattivato per la durata del run.")

        if VARIA_ROTAZIONE_NUVOLE:
            s["attore_nuvole"] = _trova_attore_per_nome(NOME_VOLUMETRIC_CLOUD)
            if s["attore_nuvole"] is not None:
                s["rotazione_nuvole_originale"] = s["attore_nuvole"].get_actor_rotation()
                unreal.log(f"Rotazione nuvole casuale attiva.")
            else:
                unreal.log_warning(f"Attore nuvole '{NOME_VOLUMETRIC_CLOUD}' non trovato.")

        if MUOVI_CAMERA:
            if POSIZIONE_INIZIALE_FISSA is not None and ROTAZIONE_INIZIALE_FISSA is not None:
                subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
                subsystem.set_level_viewport_camera_info(
                    POSIZIONE_INIZIALE_FISSA, ROTAZIONE_INIZIALE_FISSA
                )
                s["camera_location_base"] = POSIZIONE_INIZIALE_FISSA
                s["camera_rotation_base"] = ROTAZIONE_INIZIALE_FISSA
                unreal.log("Viewport posizionato automaticamente alla posizione fissa configurata.")
            else:
                s["camera_location_base"], s["camera_rotation_base"] = _cattura_posizione_viewport()
                unreal.log("Nessuna posizione fissa configurata — uso la posizione ATTUALE del viewport.")

            s["direzione_binario"] = _direzione_da_due_punti(PUNTO_BINARIO_A, PUNTO_BINARIO_B)
            unreal.log(f"Direzione binario: {s['direzione_binario']}")

        unreal.log(f"=== Avvio generazione: {NUM_IMMAGINI} immagini, "
                  f"{len(s['attori'])} pali disponibili, "
                  f"|angolo| >= {ANGOLO_MODULO_MIN} gradi, asse={ASSE_ROTAZIONE} ===")
        s["fase"] = "imposta_rotazione"
        return

    if fase == "imposta_rotazione":
        if s["indice"] >= NUM_IMMAGINI:
            s["fase"] = "fine"
            return
        nome_scelto = random.choice(list(s["attori"].keys()))
        s["nome_corrente"] = nome_scelto
        s["attore_corrente"] = s["attori"][nome_scelto]

        angolo = _campiona_angolo_estremo()
        s["angolo_corrente"] = angolo
        _imposta_rotazione(s["attore_corrente"], angolo)

        if VARIA_TEMPERATURA_LUCE and s["componente_luce"] is not None:
            temperatura = round(random.uniform(TEMPERATURA_MIN, TEMPERATURA_MAX), 0)
            s["componente_luce"].set_editor_property("temperature", temperatura)
            s["temperatura_corrente"] = temperatura
        else:
            s["temperatura_corrente"] = None

        if VARIA_ROTAZIONE_NUVOLE and s["attore_nuvole"] is not None:
            yaw_casuale = random.uniform(0.0, 360.0)
            nuova_rotazione_nuvole = unreal.Rotator(
                roll=s["rotazione_nuvole_originale"].roll,
                pitch=s["rotazione_nuvole_originale"].pitch,
                yaw=yaw_casuale,
            )
            s["attore_nuvole"].set_actor_rotation(nuova_rotazione_nuvole, False)

        s["distanza_lungo_binario_corrente"] = None
        s["distanza_target_corrente"] = None
        s["frazione_corrente"] = None

        if MUOVI_CAMERA and s["camera_location_base"] is not None:
            posizione_palo = s["attore_corrente"].get_actor_location()
            distanza_lungo_binario = _distanza_lungo_direzione(
                s["camera_location_base"], posizione_palo, s["direzione_binario"]
            )
            s["distanza_lungo_binario_corrente"] = distanza_lungo_binario

            # <<< usa la distanza target SPECIFICA di questo pivot, non un
            # valore globale — perche' pali di asset diversi hanno altezze
            # (quindi distanze target) diverse tra loro
            distanza_target_base_palo = s["distanza_target_base_per_palo"].get(nome_scelto)

            if USA_DISTANZA_TARGET_ASSOLUTA and distanza_target_base_palo is not None:
                variazione = random.uniform(
                    1.0 - VARIAZIONE_DISTANZA_TARGET, 1.0 + VARIAZIONE_DISTANZA_TARGET
                )
                distanza_target = distanza_target_base_palo * variazione
                s["distanza_target_corrente"] = distanza_target

                if distanza_lungo_binario <= distanza_target:
                    frazione = 0.0
                    unreal.log(f"   [debug camera] '{nome_scelto}' gia' entro la distanza target "
                              f"({distanza_lungo_binario:.0f} <= {distanza_target:.0f}) — nessun avvicinamento")
                else:
                    frazione = 1.0 - (distanza_target / distanza_lungo_binario)
                    frazione = max(FRAZIONE_CAMERA_MIN, min(frazione, FRAZIONE_CAMERA_MAX_CAP))
                    unreal.log(f"   [debug camera] '{nome_scelto}': partenza={distanza_lungo_binario:.0f}, "
                              f"target={distanza_target:.0f}, frazione derivata={frazione:.3f}")
            else:
                frazione = random.uniform(FRAZIONE_CAMERA_MIN, FRAZIONE_CAMERA_MAX)

            s["frazione_corrente"] = frazione
            offset = distanza_lungo_binario * frazione
            offset = max(0.0, offset)

            _sposta_viewport(s["camera_location_base"], s["camera_rotation_base"],
                            s["direzione_binario"], offset)

        s["frame_contatore"] = 0
        s["fase"] = "attesa_render"
        return

    if fase == "attesa_render":
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
            angolo = -s["angolo_corrente"] if INVERTI_SEGNO_ETICHETTA else s["angolo_corrente"]
            nome_file = f"palo_{i:04d}_{s['nome_corrente']}_angolo_{angolo:+06.1f}.png"
            destinazione = os.path.join(CARTELLA_OUTPUT, nome_file)
            try:
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

                if SALVA_ETICHETTE_OGNI_N > 0 and len(s["etichette"]) % SALVA_ETICHETTE_OGNI_N == 0:
                    percorso_json_parziale = os.path.join(CARTELLA_OUTPUT, "etichette.json")
                    with open(percorso_json_parziale, "w") as f:
                        json.dump(s["etichette"], f, indent=2)
                    unreal.log(f"   [salvataggio incrementale] etichette.json aggiornato "
                              f"({len(s['etichette'])} record)")
            except Exception as e:
                unreal.log_warning(f"[{i+1}/{NUM_IMMAGINI}] Errore spostando il file: {e}")
                s["falliti"] += 1

            _imposta_rotazione(s["attore_corrente"], 0.0)

            s["indice"] += 1
            s["fase"] = "imposta_rotazione"

        elif s["frame_contatore"] >= FRAMES_ATTESA_FILE_MAX:
            unreal.log_warning(f"[{s['indice']+1}/{NUM_IMMAGINI}] Screenshot non trovato — salto.")
            s["falliti"] += 1
            s["indice"] += 1
            s["fase"] = "imposta_rotazione"
        return

    if fase == "fine":
        percorso_json = os.path.join(CARTELLA_OUTPUT, "etichette.json")
        with open(percorso_json, "w") as f:
            json.dump(s["etichette"], f, indent=2)

        for attore in s["attori"].values():
            _imposta_rotazione(attore, 0.0)

        if MUOVI_CAMERA and s["camera_location_base"] is not None:
            subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
            subsystem.set_level_viewport_camera_info(
                s["camera_location_base"], s["camera_rotation_base"]
            )
            unreal.log("Viewport riportato alla posizione di partenza.")

        if VARIA_TEMPERATURA_LUCE and s["componente_luce"] is not None:
            s["componente_luce"].set_editor_property("temperature", s["temperatura_originale"])
            s["componente_luce"].set_editor_property("use_temperature", s["use_temperature_originale"])
            unreal.log(f"Temperatura luce ripristinata.")

        if DISATTIVA_MOTION_BLUR:
            unreal.SystemLibrary.execute_console_command(
                unreal.EditorLevelLibrary.get_editor_world(),
                f"r.MotionBlurQuality {MOTION_BLUR_QUALITY_ORIGINALE}"
            )
            unreal.log(f"Motion blur ripristinato.")

        if VARIA_ROTAZIONE_NUVOLE and s["attore_nuvole"] is not None:
            s["attore_nuvole"].set_actor_rotation(s["rotazione_nuvole_originale"], False)
            unreal.log("Rotazione nuvole ripristinata.")

        unreal.log(f"=== FATTO: {len(s['etichette'])} immagini generate "
                  f"({s['falliti']} fallite) in {CARTELLA_OUTPUT} ===")
        unreal.log(f"Etichette salvate in: {percorso_json}")

        unreal.unregister_slate_post_tick_callback(s["handle"])
        s["fase"] = "terminato"
        return


_state["handle"] = unreal.register_slate_post_tick_callback(_tick)
unreal.log("Generazione avviata in background (un frame alla volta). "
          "Controlla l'Output Log per i progressi. Non toccare il viewport.")
