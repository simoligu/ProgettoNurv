# -*- coding: utf-8 -*-
"""
Validazione di PoleTiltAnalyzer su texture REALI (non render Unreal): prende
un palo vero e dritto da un'immagine reale, gli applica rotazioni 2D note
(5, 10, 15, 20 gradi...) e verifica se il CNN stima un angolo vicino a
quello imposto.

PERCHE' QUESTO TEST, E COSA NON PUO' DIRCI
Il test MAE=1.24 riportato nei pesi e' su render Unreal sintetici (fedeli
geometricamente, ma non fotorealistici al 100%). Questo test valida invece
la sensibilita' del modello su texture vere (luce, sporco, riflessi reali)
mai viste in quella combinazione esatta durante il training — un controllo
di generalizzazione complementare, non un sostituto del test originale.

METODOLOGIA (v3 — due correzioni successive a un primo tentativo fuorviante)
La v1 di questo script ruotava l'INTERA area di lavoro attorno al palo,
sfondo compreso (montagne, cielo, cavi) — producendo una scena in cui
sembrava che fosse la CAMERA a essere inclinata, non il solo palo. Nei
render Unreal usati per il training, verosimilmente la camera restava
livellata e solo il palo si inclinava nella scena 3D: lo sfondo attorno
restava normale. La v2 ha provato a limitare la rotazione a una patch piu'
piccola attorno al palo, ma includeva ancora parecchio contesto non-palo
(traversine, cavi, sfondo vicino) — risultato praticamente invariato
(MAE ~9.2° contro l'1.24° del test originale), segno che il problema non
era ancora risolto.

Questa versione (v3) ruota SOLO il contenuto del bbox stretto stesso (il
palo, cosi' come lo conosciamo dalle coordinate passate) — nessun contesto
extra. Calcola il canvas minimo necessario a contenere il rettangolo
ruotato senza tagli, e incolla quel piccolo risultato sull'immagine
originale non ruotata: tutto cio' che sta fuori dal bbox stretto (sfondo,
traversine, cavi) resta esattamente come nell'originale. Il bbox passato a
predict_angle coincide per costruzione con l'intera regione incollata,
senza calcoli separati.

Se anche questa versione mostrasse ancora una sensibilita' molto bassa,
diventerebbe piu' plausibile un problema di generalizzazione del modello a
texture reali (o un limite intrinseco della sola rotazione 2D rispetto a
una vera inclinazione 3D, vedi limite sotto), non piu' un artefatto di
metodologia del test.

LIMITE ONESTO RESIDUO, da riportare comunque in tesi: anche cosi', resta una
rotazione 2D nel piano immagine, non una vera inclinazione 3D del palo (che
cambierebbe leggermente anche la prospettiva). Il pivot alla base approssima
la parte piu' importante dell'effetto (il perno a terra), non la componente
prospettica.

COME FUNZIONA
1. Definisce una patch STRETTA attorno al bbox del palo (non l'intera area
   di lavoro) — abbastanza grande da contenere il palo anche ruotato al
   massimo angolo testato, ma senza includere sfondo lontano non necessario.
2. Per ogni angolo di test: ritaglia la patch dall'immagine ORIGINALE, la
   ruota attorno alla base del palo, e la incolla (sovrascrivendo) su una
   COPIA dell'immagine originale — il resto della scena resta invariato.
3. Ruota anche i 4 angoli del bbox originale con la stessa trasformazione e
   ne prende il rettangolo di inclusione, in coordinate dell'immagine
   originale.
4. Chiama predict_angle() esattamente come nella pipeline reale.
5. Salva un ritaglio annotato (zoomato sulla zona del palo, non l'intero
   frame) per il controllo visivo, e un CSV con angolo imposto vs stimato.

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python valida_pole_tilt_rotazione_2d.py --immagine frame_palo_dritto.jpg \
        --bbox 850,400,40,180 --pesi weights/pole_tilt_best.pt

--bbox: x,y,w,h del rettangolo STRETTO attorno al palo dritto nell'immagine
        sorgente.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from pole_tilt_analyzer import PoleTiltAnalyzer

ANGOLI_DEFAULT = [-20, -15, -10, -5, -3, 0, 3, 5, 10, 15, 20]


def ruota_punto(px, py, cx, cy, angolo_rad):
    """Ruota un punto (px,py) attorno al centro (cx,cy) di angolo_rad radianti."""
    dx, dy = px - cx, py - cy
    cos_a, sin_a = math.cos(angolo_rad), math.sin(angolo_rad)
    return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--immagine", required=True, help="Immagine sorgente col palo dritto")
    ap.add_argument("--bbox", required=True,
                     help="x,y,w,h del bbox STRETTO attorno al palo dritto, es. 850,400,40,180")
    ap.add_argument("--pesi", default="weights/pole_tilt_best.pt")
    ap.add_argument("--angolo_max", type=float, default=22.0)
    ap.add_argument("--dimensione_input", type=int, default=224)
    ap.add_argument("--margine_bbox_frazione", type=float, default=0.0,
                     help="Passato al costruttore di PoleTiltAnalyzer, ma di fatto un NO-OP in "
                          "questo script: il bbox finale passato a predict_angle coincide sempre "
                          "con l'intera immagine gia' pronta (costruita qui in stile training, "
                          "margine 0.55x l'altezza in orizzontale — vedi commenti nel ciclo "
                          "principale), quindi qualunque espansione automatica verrebbe comunque "
                          "troncata ai bordi. Lasciato configurabile solo per completezza.")
    ap.add_argument("--angoli", default=",".join(str(a) for a in ANGOLI_DEFAULT),
                     help="Lista di angoli da testare, separati da virgola (gradi)")
    ap.add_argument("--out", default="valida_pole_tilt_out")
    args = ap.parse_args()

    angoli_test = [float(a) for a in args.angoli.split(",")]
    x, y, w, h = (int(v) for v in args.bbox.split(","))

    img_originale = cv2.imread(args.immagine)
    if img_originale is None:
        print(f"[ERRORE] Impossibile aprire {args.immagine}")
        return
    img_h, img_w = img_originale.shape[:2]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = PoleTiltAnalyzer(
        weights_path=args.pesi,
        angolo_max=args.angolo_max,
        dimensione_input=args.dimensione_input,
        margine_bbox_frazione=args.margine_bbox_frazione,
    )

    # base del palo = pivot di rotazione (perno a terra, non centro del bbox),
    # in coordinate LOCALI al ritaglio stretto (non dell'immagine intera)
    pivot_loc_x, pivot_loc_y = w / 2.0, float(h)
    pivot_x_globale, pivot_y_globale = x + pivot_loc_x, y + pivot_loc_y

    crop_stretto = img_originale[y:y + h, x:x + w].copy()

    print(f"[INFO] Ruoto SOLO il contenuto del bbox ({w}x{h}px) — non una patch piu' ampia "
          f"con contesto attorno (traversine, cavi, sfondo)")
    print(f"[INFO] Testo {len(angoli_test)} angoli: {angoli_test}\n")

    risultati = []
    for angolo_imposto in angoli_test:
        # ruoto SOLO il ritaglio stretto, con un canvas di output dimensionato
        # esattamente per contenere il rettangolo ruotato senza tagli
        M = cv2.getRotationMatrix2D((pivot_loc_x, pivot_loc_y), angolo_imposto, 1.0)
        corners_locali = [(0, 0), (w, 0), (0, h), (w, h)]
        corners_ruotati = [ruota_punto(cx, cy, pivot_loc_x, pivot_loc_y, math.radians(angolo_imposto))
                            for cx, cy in corners_locali]
        xs = [c[0] for c in corners_ruotati]
        ys = [c[1] for c in corners_ruotati]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        out_w, out_h = int(math.ceil(max_x - min_x)), int(math.ceil(max_y - min_y))

        # sposto la matrice di rotazione perche' il contenuto ruotato (che puo'
        # avere coordinate negative rispetto al pivot originale) cada tutto
        # dentro il nuovo canvas [0, out_w] x [0, out_h]
        M[0, 2] -= min_x
        M[1, 2] -= min_y

        # IMPORTANTE: il rettangolo ruotato non riempie tutto il canvas di
        # output (che deve essere piu' grande per contenerlo senza tagli) —
        # i "vuoti" agli angoli vanno riempiti con lo SFONDO VERO (originale,
        # non ruotato), non con bordi replicati/stirati (BORDER_REPLICATE
        # produceva un artefatto a strisce diagonali, scambiabile per errore
        # con una struttura reticolare reale — bug scoperto empiricamente).
        # Soluzione: ruoto anche una maschera bianca della stessa forma del
        # rettangolo originale, e compongo pixel per pixel solo dove la
        # maschera indica "qui c'e' davvero contenuto ruotato".
        ruotato = cv2.warpAffine(crop_stretto, M, (out_w, out_h),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        maschera = np.full((h, w), 255, dtype=np.uint8)
        maschera_ruotata = cv2.warpAffine(maschera, M, (out_w, out_h),
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # il pivot, essendo il centro di rotazione, resta fermo durante la
        # rotazione pura — la sua posizione nel nuovo canvas e' semplicemente
        # la sua posizione originale traslata dello stesso spostamento
        pivot_out_x, pivot_out_y = pivot_loc_x - min_x, pivot_loc_y - min_y

        # incollo SOLO dove la maschera e' valida — il resto della scena
        # (sfondo, traversine, cavi, e ora anche i "vuoti" del canvas ruotato)
        # resta esattamente come nell'immagine originale
        paste_x = int(round(pivot_x_globale - pivot_out_x))
        paste_y = int(round(pivot_y_globale - pivot_out_y))

        frame_composito = img_originale.copy()
        dst_x0, dst_y0 = max(0, paste_x), max(0, paste_y)
        dst_x1, dst_y1 = min(img_w, paste_x + out_w), min(img_h, paste_y + out_h)
        src_x0, src_y0 = dst_x0 - paste_x, dst_y0 - paste_y
        src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)

        regione_dst = frame_composito[dst_y0:dst_y1, dst_x0:dst_x1]
        regione_src = ruotato[src_y0:src_y1, src_x0:src_x1]
        maschera_regione = maschera_ruotata[src_y0:src_y1, src_x0:src_x1]
        maschera_3ch = cv2.cvtColor(maschera_regione, cv2.COLOR_GRAY2BGR).astype(bool)
        regione_dst[maschera_3ch] = regione_src[maschera_3ch]
        frame_composito[dst_y0:dst_y1, dst_x0:dst_x1] = regione_dst

        # il bbox da passare a predict_angle e' ESATTAMENTE la regione appena
        # incollata — nessun calcolo separato necessario, coincide per costruzione
        nuovo_bbox = (dst_x0, dst_y0, dst_x1 - dst_x0, dst_y1 - dst_y0)
        nx_min, ny_min, nx_max, ny_max = dst_x0, dst_y0, dst_x1, dst_y1

        # COSTRUISCO IL CROP IN STILE TRAINING invece di affidarmi al margine
        # automatico interno di predict_angle (margine_bbox_frazione, simmetrico
        # sulle proprie dimensioni) — validato empiricamente come sbagliato
        # rispetto alla vera convenzione di crop_dataset.py (margine 0.55x
        # l'ALTEZZA, solo in orizzontale). Vedi costruisci_crop_stile_training.py
        # per lo stesso procedimento usato con successo su foto reali statiche.
        nb_x, nb_y, nb_w, nb_h = nuovo_bbox
        margine_x_training = int(nb_h * 0.55)
        crop_x_min = max(0, nb_x - margine_x_training)
        crop_x_max = min(img_w, nb_x + nb_w + margine_x_training)
        crop_y_min = max(0, nb_y)
        crop_y_max = min(img_h, nb_y + nb_h)
        crop_per_modello = frame_composito[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        if crop_per_modello.size == 0:
            angolo_predetto = None
        else:
            crop_h_m, crop_w_m = crop_per_modello.shape[:2]
            dim = args.dimensione_input
            scala = dim / max(crop_w_m, crop_h_m)
            nuova_w_m = max(1, int(round(crop_w_m * scala)))
            nuova_h_m = max(1, int(round(crop_h_m * scala)))
            ridim = cv2.resize(crop_per_modello, (nuova_w_m, nuova_h_m),
                                interpolation=cv2.INTER_LANCZOS4)
            input_finale = np.zeros((dim, dim, 3), dtype=np.uint8)
            off_x = (dim - nuova_w_m) // 2
            off_y = (dim - nuova_h_m) // 2
            input_finale[off_y:off_y + nuova_h_m, off_x:off_x + nuova_w_m] = ridim
            # margine_bbox_frazione=0 nel costruttore garantisce che predict_angle
            # non applichi ULTERIORE espansione sopra al crop gia' pronto
            angolo_predetto = analyzer.predict_angle(input_finale, (0, 0, dim, dim))

        # DEBUG: salvo anche esattamente cio' che il modello riceve dopo
        # espansione del margine e letterbox — per capire se un eventuale
        # problema e' nella geometria di rotazione (sopra) o nel preprocessing
        # interno di PoleTiltAnalyzer
        x_min_esp, y_min_esp, x_max_esp, y_max_esp = analyzer._espandi_bbox(
            nuovo_bbox, img_w, img_h)
        crop_espanso = frame_composito[y_min_esp:y_max_esp, x_min_esp:x_max_esp]
        if crop_espanso.size > 0:
            crop_rgb_per_letterbox = cv2.cvtColor(crop_espanso, cv2.COLOR_BGR2RGB)
            input_modello = analyzer._letterbox(crop_rgb_per_letterbox)
            input_modello_bgr = cv2.cvtColor(input_modello, cv2.COLOR_RGB2BGR)
            nome_debug = f"input_modello_{angolo_imposto:+05.1f}.jpg".replace("+", "p").replace("-", "m")
            cv2.imwrite(str(out_dir / nome_debug), input_modello_bgr)

        if angolo_predetto is None:
            print(f"  angolo imposto {angolo_imposto:+.1f}° -> CROP NON VALIDO (bbox degenere)")
            risultati.append((angolo_imposto, None, None))
        else:
            errore = angolo_predetto - angolo_imposto
            print(f"  angolo imposto {angolo_imposto:+6.1f}° -> CNN stima {angolo_predetto:+6.2f}° "
                  f"(errore {errore:+6.2f}°)")
            risultati.append((angolo_imposto, angolo_predetto, errore))

        # salva un ritaglio (zoomato sulla zona del palo, non l'intero frame)
        # con il bbox disegnato, per il controllo visivo
        margine_vista = int(math.hypot(w, h) * 0.6)
        vx_min = max(0, dst_x0 - margine_vista)
        vy_min = max(0, dst_y0 - margine_vista)
        vx_max = min(img_w, dst_x1 + margine_vista)
        vy_max = min(img_h, dst_y1 + margine_vista)
        annotata = frame_composito[vy_min:vy_max, vx_min:vx_max].copy()
        cv2.rectangle(annotata, (nx_min - vx_min, ny_min - vy_min),
                       (nx_max - vx_min, ny_max - vy_min), (0, 165, 255), 2)
        cv2.circle(annotata, (int(pivot_x_globale - vx_min), int(pivot_y_globale - vy_min)),
                   5, (0, 0, 255), -1)
        testo = f"imposto={angolo_imposto:+.1f} CNN={angolo_predetto:+.2f}" if angolo_predetto is not None \
            else f"imposto={angolo_imposto:+.1f} CNN=non valido"
        cv2.putText(annotata, testo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        nome_file = f"angolo_{angolo_imposto:+05.1f}.jpg".replace("+", "p").replace("-", "m")
        cv2.imwrite(str(out_dir / nome_file), annotata)

    # CSV riassuntivo
    csv_path = out_dir / "risultati.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angolo_imposto", "angolo_predetto", "errore"])
        writer.writerows(risultati)

    validi = [r for r in risultati if r[1] is not None]
    if validi:
        mae = sum(abs(r[2]) for r in validi) / len(validi)
        print(f"\n=== RISULTATO ===")
        print(f"MAE su {len(validi)}/{len(risultati)} angoli validi: {mae:.2f}°")
        print(f"(confronta con MAE=1.24° del test originale su render Unreal sintetici — "
              f"un valore sensibilmente piu' alto qui indicherebbe un problema di "
              f"generalizzazione a texture reali, non necessariamente un bug)")
    print(f"\nImmagini annotate e CSV salvati in: {out_dir}/")


if __name__ == "__main__":
    main()