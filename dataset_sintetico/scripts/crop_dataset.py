# -*- coding: utf-8 -*-
"""
Post-processing: ritaglia le immagini gia' generate attorno al palo,
ricostruendo la geometria della camera dai metadati salvati in etichette.json
(nessun nuovo render necessario in Unreal).

PERCHE' SERVE:
Il modello CNN per l'inclinazione va affiancato a YOLO e DeepLab come modulo
indipendente nella pipeline reale. In produzione ricevera' un crop attorno al
palo, ritagliato dal bbox che DeepLab produce da _analyze_poles()
(connectedComponentsWithStats sulla classe CL_PALI), non il frame intero.
Le immagini generate in Unreal sono invece scene intere — serve un crop
coerente per non introdurre un mismatch training/inferenza.

COME FUNZIONA:
Per ogni immagine, dai metadati salvati (distanza_lungo_binario_iniziale,
frazione_applicata) si ricostruisce l'offset esatto applicato alla camera,
quindi la sua posizione 3D al momento dello scatto (la rotazione della
camera resta invece fissa per l'intero run, quindi e' nota a priori). Nota
la posizione 3D della base del palo (misurata a mano in Unreal) e la sua
altezza, si proietta un volume di sicurezza attorno al palo (che copre
l'intero range di inclinazione, non solo l'angolo del singolo scatto, per
margine di sicurezza) sullo schermo con un modello pinhole camera standard,
ottenendo un bounding box 2D. Le immagini vengono ritagliate su quel bbox
(con margine extra) e opzionalmente ridimensionate con letterboxing a una
dimensione fissa per il training.

USO:
    python crop_dataset.py --config original   # dataset principale (2000 img)
    python crop_dataset.py --config asset2      # dataset test (300 img)

Prima di lanciare, verifica che i valori in DATASET_CONFIGS corrispondano
esattamente a quelli usati negli script Unreal (posizione/rotazione camera,
punti binario, altezza palo, FOV, risoluzione) — sono duplicati qui perche'
questo script gira fuori da Unreal, in Python puro sul PC.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

from PIL import Image

# ==================== CONFIGURAZIONE PER DATASET ====================
# ATTENZIONE: questi valori devono coincidere esattamente con quelli usati
# negli script Unreal al momento della generazione. Se modifichi lo script
# Unreal e rigeneri un dataset, aggiorna anche qui.

DATASET_CONFIGS = {
    "original": dict(
        etichette_json=r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_originale\etichette.json",
        cartella_immagini=r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_originale",
        cartella_output=r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_originale_crop",
        posizione_iniziale=(-707.034477, 233027.226015, 51789.551552),
        rotazione_iniziale=dict(pitch=2.200100, yaw=-1530.598195, roll=-0.000001),
        # direzione binario = normalize(B - A), stessi punti usati nello script V3
        punto_binario_a=(0.0, 0.0, 0.0),
        punto_binario_b=(90.0, -3683600.0, 0.0),
        altezza_palo=92990.0,
        fov_orizzontale_gradi=90.0,
        risoluzione=(1920, 1080),
        angolo_max_gradi=22.0,          # ANGOLO_MAX dello script Unreal
        posizioni_pali={
            "PolePivot_Base2": (79140.00, -102300.00, 10940.00),
            "PolePivot_Base3": (-58640.00, -553770.00, 10720.00),
            "PolePivot_Base4": (79370.00, -1031230.00, 7730.00),
        },
    ),
    "asset2": dict(
        etichette_json=r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_asset2_test\etichette.json",
        cartella_immagini=r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_asset2_test",
        cartella_output=r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_asset2_test_crop",
        posizione_iniziale=(3367.633837, -5981442.977994, 81735.589591),
        rotazione_iniziale=dict(pitch=1.200100, yaw=-1708.798147, roll=-0.000001),
        # punti SCAMBIATI rispetto all'originale (stesso fix applicato nello
        # script Unreal per la direzione invertita in questa zona)
        punto_binario_a=(90.0, -3683600.0, 0.0),
        punto_binario_b=(0.0, 0.0, 0.0),
        altezza_palo=151800.0,
        fov_orizzontale_gradi=90.0,
        risoluzione=(1920, 1080),
        angolo_max_gradi=22.0,
        posizioni_pali={
            "PolePivot_Asset2_1": (69500.00, -5565440.00, 9390.00),
            "PolePivot_Asset2_2": (-67180.00, -5236290.00, 9390.00),
            "PolePivot_Asset2_3": (69510.00, -4945430.00, 9390.00),
        },
    ),
    "asset3": dict(
        etichette_json=r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_asset3\etichette.json",
        cartella_immagini=r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_asset3",
        cartella_output=r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_asset3_crop",
        posizione_iniziale=(102.712831, -1951596.096710, 60287.113353),
        rotazione_iniziale=dict(pitch=2.600100, yaw=-1170.998088, roll=-0.000001),
        # punti NON scambiati (stessa direzione dello script principale): la
        # camera dell'asset3 guarda nello stesso verso generale del dataset
        # originale (yaw quasi identico, differenza ~1°), verificato anche
        # empiricamente col batch di test (partenza sempre positiva, nessuna
        # inversione di segno necessaria per questo asset)
        punto_binario_a=(0.0, 0.0, 0.0),
        punto_binario_b=(90.0, -3683600.0, 0.0),
        altezza_palo=117040.0,
        fov_orizzontale_gradi=90.0,
        risoluzione=(1920, 1080),
        angolo_max_gradi=22.0,
        posizioni_pali={
            "Electric_Pole": (63230.00, -2253230.00, 8100.00),
            "Electric_Pole2": (63240.00, -2896910.00, 8100.00),
            "Electric_Pole3": (-81480.00, -2549770.00, 8100.00),
        },
    ),
    "estremi": dict(
        etichette_json=r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_estremi\etichette.json",
        cartella_immagini=r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_estremi",
        cartella_output=r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_estremi_crop",
        posizione_iniziale=(-479.380163, -3184983.712027, 52162.036756),
        rotazione_iniziale=dict(pitch=6.200100, yaw=-1170.398096, roll=-0.000001),
        # punti NON scambiati, stesso orientamento generale del dataset
        # originale (yaw quasi identico, differenza ~0.2°)
        punto_binario_a=(0.0, 0.0, 0.0),
        punto_binario_b=(90.0, -3683600.0, 0.0),
        # ALTEZZE PER PALO (non un valore unico): questo dataset mescola
        # duplicati dell'asset originale (92990) e dell'asset3 (117040)
        altezze_pali={
            "PolePivot_Base5": 92990.0,
            "PolePivot_Base6": 92990.0,
            "Electric_Pole4": 117040.0,
            "Electric_Pole5": 117040.0,
        },
        fov_orizzontale_gradi=90.0,
        risoluzione=(1920, 1080),
        angolo_max_gradi=22.0,
        posizioni_pali={
            "Electric_Pole4": (63250.00, -3414780.00, 8100.00),
            "Electric_Pole5": (-81470.00, -4478390.00, 8100.00),
            "PolePivot_Base5": (-58630.00, -3699880.00, 10720.00),
            "PolePivot_Base6": (79150.00, -4159240.00, 10940.00),
        },
    ),
}

# --- margine di sicurezza extra sul bbox proiettato ---
# Copre: braccio/catenaria o traversa che si estende oltre il solo montante,
# imprecisione realistica del detector DeepLab in produzione (che non
# isolera' mai il palo con bbox pixel-perfect), e un margine per il fatto
# che il volume 3D usato per la proiezione e' una scatola conservativa, non
# la mesh esatta.
MARGINE_ORIZZONTALE_FRAZIONE_ALTEZZA = 0.55   # margine orizzontale ai lati
                                                # del palo, come frazione
                                                # dell'altezza del palo
MARGINE_VERTICALE_FRAZIONE_ALTEZZA = 0.12     # margine extra sopra/sotto
PADDING_FINALE_PIXEL_FRAZIONE = 0.06          # padding finale sul bbox in
                                                # pixel, come frazione della
                                                # dimensione del bbox stesso

# --- dimensione di output dopo il crop (letterbox, per input di rete
# uniforme) ---
DIMENSIONE_OUTPUT = 224   # lato del quadrato finale (es. per ResNet18/34)


# ==================== GEOMETRIA ====================

@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def scale(self, s):
        return Vec3(self.x * s, self.y * s, self.z * s)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self):
        return math.sqrt(self.dot(self))

    def normalized(self):
        l = self.length()
        if l < 1e-9:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / l, self.y / l, self.z / l)


def direzione_binario(punto_a: Tuple[float, float, float], punto_b: Tuple[float, float, float]) -> Vec3:
    a = Vec3(*punto_a)
    b = Vec3(*punto_b)
    d = b - a
    d.z = 0.0  # solo orizzontale, come nello script Unreal
    return d.normalized()


def assi_camera(rotazione: Dict[str, float]) -> Tuple[Vec3, Vec3, Vec3]:
    """
    Ricostruisce forward/right/up dalla Rotator (pitch, yaw, roll) in gradi,
    con la convenzione di Unreal Engine (Z-up, X-forward, Y-right).

    Il roll viene ignorato (assunto ~0, coerente con tutti gli scatti del
    progetto: ROTAZIONE_INIZIALE_FISSA ha sempre roll=-0.000001): right e up
    si derivano da prodotti vettoriali con l'asse verticale del mondo, che
    garantiscono una base ortonormale corretta senza ambiguita' sull'ordine
    di composizione delle rotazioni (pitch/yaw/roll), a differenza di una
    formula a matrice completa che e' facile sbagliare nel segno.
    """
    p = math.radians(rotazione["pitch"])
    y = math.radians(rotazione["yaw"])

    forward = Vec3(math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p))
    world_up = Vec3(0.0, 0.0, 1.0)

    # right = normalize(world_up x forward) — verificato numericamente che
    # a yaw=0/pitch=0 da' (0,1,0), coerente con la convenzione UE (+Y = destra)
    right = Vec3(
        world_up.y * forward.z - world_up.z * forward.y,
        world_up.z * forward.x - world_up.x * forward.z,
        world_up.x * forward.y - world_up.y * forward.x,
    ).normalized()

    # up = forward x right, per completare una base ortonormale coerente
    up = Vec3(
        forward.y * right.z - forward.z * right.y,
        forward.z * right.x - forward.x * right.z,
        forward.x * right.y - forward.y * right.x,
    )

    return forward, right, up


def proietta_punto(punto: Vec3, camera_pos: Vec3, forward: Vec3, right: Vec3, up: Vec3,
                    fov_o_rad: float, fov_v_rad: float, larghezza: int, altezza: int):
    """
    Proietta un punto 3D sullo schermo (modello pinhole).
    Ritorna (pixel_x, pixel_y, dietro_camera: bool).
    """
    d = punto - camera_pos
    profondita = d.dot(forward)
    if profondita <= 1.0:
        return None, None, True  # dietro o troppo vicino alla camera

    comp_right = d.dot(right)
    comp_up = d.dot(up)

    mezza_larghezza = profondita * math.tan(fov_o_rad / 2.0)
    mezza_altezza = profondita * math.tan(fov_v_rad / 2.0)

    screen_x_norm = 0.5 + 0.5 * (comp_right / mezza_larghezza)
    screen_y_norm = 0.5 - 0.5 * (comp_up / mezza_altezza)

    px = screen_x_norm * larghezza
    py = screen_y_norm * altezza
    return px, py, False


def calcola_bbox_palo(cfg: dict, palo_nome: str, distanza_lungo_binario: float,
                       frazione_applicata: float) -> Tuple[int, int, int, int]:
    """
    Calcola il bbox 2D (x_min, y_min, x_max, y_max) in pixel del volume di
    sicurezza attorno al palo, per un dato scatto.
    """
    larghezza, altezza_img = cfg["risoluzione"]
    fov_o_rad = math.radians(cfg["fov_orizzontale_gradi"])
    aspect = larghezza / altezza_img
    fov_v_rad = 2.0 * math.atan(math.tan(fov_o_rad / 2.0) / aspect)

    direzione = direzione_binario(cfg["punto_binario_a"], cfg["punto_binario_b"])
    posizione_base_camera = Vec3(*cfg["posizione_iniziale"])
    offset = distanza_lungo_binario * frazione_applicata
    camera_pos = posizione_base_camera + direzione.scale(offset)

    forward, right, up = assi_camera(cfg["rotazione_iniziale"])

    base_palo = Vec3(*cfg["posizioni_pali"][palo_nome])
    # supporta sia un'altezza UNICA per tutti i pali del dataset (chiave
    # "altezza_palo", usata da original/asset2/asset3) sia altezze DIVERSE
    # per palo (chiave "altezze_pali", dizionario nome->altezza — necessario
    # per "estremi", che mescola pali di asset diversi con altezze diverse
    # nello stesso batch, stessa logica gia' usata nello script Unreal)
    if "altezze_pali" in cfg:
        altezza_palo = cfg["altezze_pali"][palo_nome]
    else:
        altezza_palo = cfg["altezza_palo"]
    top_palo = base_palo + Vec3(0.0, 0.0, altezza_palo)

    margine_h = altezza_palo * MARGINE_ORIZZONTALE_FRAZIONE_ALTEZZA
    margine_v = altezza_palo * MARGINE_VERTICALE_FRAZIONE_ALTEZZA

    # IMPORTANTE: il margine va espanso lungo gli assi RELATIVI ALLA CAMERA
    # (right/up), non lungo gli assi mondo X/Y. La direzione del binario (e
    # quindi la vista della camera) e' quasi allineata con l'asse mondo Y in
    # questo progetto: espandere lungo Y significherebbe espandere lungo la
    # PROFONDITA', non lateralmente sullo schermo — con punti che finiscono
    # troppo vicini alla camera e un bbox che esplode fuori scala.
    punti_3d = []
    for punto_centrale in (base_palo, top_palo):
        for dr in (-margine_h, margine_h):
            for du in (-margine_v, margine_v):
                punti_3d.append(punto_centrale + right.scale(dr) + up.scale(du))

    punti_x, punti_y = [], []
    for p in punti_3d:
        px, py, dietro = proietta_punto(p, camera_pos, forward, right, up,
                                         fov_o_rad, fov_v_rad, larghezza, altezza_img)
        if not dietro:
            punti_x.append(px)
            punti_y.append(py)

    if not punti_x:
        # tutti i punti sono risultati dietro la camera: caso anomalo,
        # non dovrebbe succedere con una geometria corretta
        return None

    x_min, x_max = min(punti_x), max(punti_x)
    y_min, y_max = min(punti_y), max(punti_y)

    # padding finale come frazione delle dimensioni del bbox
    pad_x = (x_max - x_min) * PADDING_FINALE_PIXEL_FRAZIONE
    pad_y = (y_max - y_min) * PADDING_FINALE_PIXEL_FRAZIONE
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    # clamp ai bordi dell'immagine
    x_min = max(0, int(round(x_min)))
    y_min = max(0, int(round(y_min)))
    x_max = min(larghezza, int(round(x_max)))
    y_max = min(altezza_img, int(round(y_max)))

    return (x_min, y_min, x_max, y_max)


# --- controllo gizmo assi Unreal ---
# Il viewport editor disegna sempre una piccola icona degli assi (rosso/
# verde/blu) nell'angolo in basso a sinistra dello schermo — un artefatto
# dell'editor che non esistera' mai nei frame video reali della pipeline.
# Se il bbox calcolato per un crop si estende abbastanza verso quell'angolo,
# rischia di includerlo. Questa soglia (in pixel, sulla risoluzione
# 1920x1080) segna come "a rischio" ogni crop il cui bordo sinistro/inferiore
# entra in quella zona, per un controllo sistematico invece che a campione.
ZONA_GIZMO_LARGHEZZA_PX = 90
ZONA_GIZMO_ALTEZZA_PX = 90


def bbox_rischia_gizmo(bbox: Tuple[int, int, int, int], altezza_img: int) -> bool:
    x_min, y_min, x_max, y_max = bbox
    return x_min < ZONA_GIZMO_LARGHEZZA_PX and y_max > (altezza_img - ZONA_GIZMO_ALTEZZA_PX)


# ==================== LETTERBOX RESIZE ====================

def letterbox(img: Image.Image, dimensione: int) -> Image.Image:
    """Ridimensiona mantenendo l'aspect ratio, con padding nero per riempire
    un quadrato dimensione x dimensione — cosi' il crop rettangolare
    variabile (dipende da distanza/angolo) diventa un input di rete
    uniforme senza distorcere le proporzioni del palo."""
    w, h = img.size
    scala = dimensione / max(w, h)
    nuova_w, nuova_h = int(round(w * scala)), int(round(h * scala))
    img_ridim = img.resize((nuova_w, nuova_h), Image.LANCZOS)

    risultato = Image.new("RGB", (dimensione, dimensione), (0, 0, 0))
    offset_x = (dimensione - nuova_w) // 2
    offset_y = (dimensione - nuova_h) // 2
    risultato.paste(img_ridim, (offset_x, offset_y))
    return risultato


# ==================== MAIN ====================

def main():
    ap = argparse.ArgumentParser(description="Ritaglia il dataset sintetico attorno al palo")
    ap.add_argument("--config", required=True, choices=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--no-letterbox", action="store_true",
                     help="Salva il crop rettangolare cosi' com'e', senza ridimensionarlo a quadrato")
    args = ap.parse_args()

    cfg = DATASET_CONFIGS[args.config]

    with open(cfg["etichette_json"], "r", encoding="utf-8") as f:
        etichette = json.load(f)

    os.makedirs(cfg["cartella_output"], exist_ok=True)

    nuove_etichette = []
    falliti = 0
    a_rischio_gizmo = []

    for i, rec in enumerate(etichette):
        percorso_originale = os.path.join(cfg["cartella_immagini"], rec["file"])
        if not os.path.exists(percorso_originale):
            print(f"[SKIP] File non trovato: {percorso_originale}")
            falliti += 1
            continue

        bbox = calcola_bbox_palo(
            cfg, rec["palo"], rec["distanza_lungo_binario_iniziale"], rec["frazione_applicata"]
        )
        if bbox is None:
            print(f"[SKIP] Bbox non calcolabile per {rec['file']} (geometria anomala)")
            falliti += 1
            continue

        if bbox_rischia_gizmo(bbox, cfg["risoluzione"][1]):
            a_rischio_gizmo.append(rec["file"])

        try:
            img = Image.open(percorso_originale).convert("RGB")
            crop = img.crop(bbox)
            if args.no_letterbox:
                finale = crop
            else:
                finale = letterbox(crop, DIMENSIONE_OUTPUT)

            percorso_output = os.path.join(cfg["cartella_output"], rec["file"])
            finale.save(percorso_output)

            nuovo_rec = dict(rec)
            nuovo_rec["bbox_crop"] = bbox
            nuove_etichette.append(nuovo_rec)
        except Exception as e:
            print(f"[SKIP] Errore su {rec['file']}: {e}")
            falliti += 1

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(etichette)}] elaborate...")

    percorso_json_out = os.path.join(cfg["cartella_output"], "etichette.json")
    with open(percorso_json_out, "w", encoding="utf-8") as f:
        json.dump(nuove_etichette, f, indent=2)

    print(f"\n=== FATTO: {len(nuove_etichette)} immagini ritagliate ({falliti} fallite) "
          f"in {cfg['cartella_output']} ===")
    print(f"Etichette (con bbox_crop aggiunto) salvate in: {percorso_json_out}")

    if a_rischio_gizmo:
        print(f"\n[ATTENZIONE] {len(a_rischio_gizmo)} crop potrebbero includere il gizmo "
              f"assi in basso a sinistra (bbox esteso in quella zona). Controlla a occhio "
              f"alcuni di questi file nella cartella di output:")
        for f in a_rischio_gizmo[:15]:
            print(f"   - {f}")
        if len(a_rischio_gizmo) > 15:
            print(f"   ... e altri {len(a_rischio_gizmo) - 15}")
    else:
        print("\nNessun crop risulta a rischio gizmo (nessun bbox si estende in quella zona).")


if __name__ == "__main__":
    main()