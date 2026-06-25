import cv2
import numpy as np
from pathlib import Path

from sympy.codegen.ast import continue_
from tqdm import tqdm

#--CONFIGURAZIONE--
BASE = Path("data/rs19_val")
MASKS_DIR = BASE / "uint8" / "rs19_val"     #le maschere dense
IMAGES_DIR = BASE / "jpgs" / "rs19_val"     #le immagini (per verifica corrispondenza)
OUT_LABELS = BASE / "labels"                #qui scriviamo i .txt YOLO

#Mappatura: valore pixel in RailSem19 -> indice classe nel dataset YOLO
#17 (rail-raised) -> 0; 5 (pole) -> 1 ; 8 (vegetation) -> 2
CLASS_MAP = {
    17: 0,  #rotaie
    5 : 1,  #pali
    8 : 2,  #vegetazione
}

#Parametri di pulizia
MIN_AREA = 80      #ignora potenziali macchie piu piccole di TOT pixel (RUMORE)
APPROX_EPS_RATIO = 0.001    #semplificazione contorno (più alto = meno punti)

CLASSI_SOTTILI = {17}
DILATAZIONE_PX = 15

def mask_to_yolo_lines(mask: np.ndarray):
    """Converte una maschera densa in righe YOLO-seg (poligoni normalizzati)."""
    h, w = mask.shape
    lines = []

    for pixel_value, class_idx in CLASS_MAP.items():
    #isola i pixel di questa classe -> maschera binaria
        binary = (mask == pixel_value).astype(np.uint8) * 255
        if cv2.countNonZero(binary) == 0:
            continue    #questa classe non c'è in questa immagine

        if pixel_value in CLASSI_SOTTILI:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATAZIONE_PX, DILATAZIONE_PX))
            binary = cv2.dilate(binary, kernel, iterations=1)

        #trova i contorni degli oggetti di questa classe
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue    #scarta macchie troppo piccole

        #semplifica il contorno (riduce il numero di punti)
        eps = APPROX_EPS_RATIO * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)

        #YOLO-seg richiede almeno 3 punti (un poligono)
        if len(approx) < 3 or cv2.contourArea(approx)<area*0.5:
            approx = cnt    #ripiega sul contorno pieno, non semplificato

        if len(approx)<3:
            continue        #davvero degenere, saltalo

        #normalizza le coordinate tra 0 e 1 e costruisci la riga
        tokens = [str(class_idx)]
        for point in approx:
            x, y = point[0]
            tokens.append(f"{x / w:.6f}")
            tokens.append(f"{y / h:.6f}")

        line = " ".join(tokens)
        lines.append(line)

    return lines

def main():
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(MASKS_DIR.glob("*.png"))
    mask_files = mask_files[:20]   # PROVA: solo le prime 20, poi rimuovi questa riga
    if not mask_files:
        print(f"[Errore] Nessuna maschera trovata in {MASKS_DIR}")
        return
    print(f"[INFO] Trovate {len(mask_files)} maschere. Inizio conversione...")

    n_con_label = 0
    n_vuote = 0

    for mask_path in tqdm(mask_files, desc="Conversione"):
        mask=cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Maschera illegibile: {mask_path.name}")
            continue

        lines = mask_to_yolo_lines(mask)

        #scrivi il .txt con lo stesso nome dell'immagine
        out_file = OUT_LABELS / f"{mask_path.stem}.txt"
        out_file.write_text("\n".join(lines))

        if lines:
            n_con_label += 1
        else:
            n_vuote += 1

    print(f"\n[INFO] Conversione completata.")
    print(f"    Immagini con almeno una classe: {n_con_label}")
    print(f"    Immagini senza nessuna delle 3 classi (label vuota): {n_vuote}")
    print(f"    Label salvate in: {OUT_LABELS}")

if __name__ == "__main__":
    main()
