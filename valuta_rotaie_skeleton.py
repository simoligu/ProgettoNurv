# -*- coding: utf-8 -*-
"""
Valutazione SKELETON-BASED delle ROTAIE per NURV.

Perche' serve: le label delle rotaie sono state INGROSSATE in conversione (chiusura
morfologica + dilatazione, per collegare i frammenti sottili). Quindi:
  - l'IoU pixel contro il ground truth sottile e' ingiustamente BASSO (la predizione
    "sborda" ai lati della rotaia vera);
  - l'IoU contro le label ingrossate misurerebbe solo quanto il modello replica il
    nostro stesso artefatto.
La soluzione corretta per una struttura LINEARE e' valutare la LINEA CENTRALE (skeleton):
si riducono sia ground truth sia predizione alla loro mezzeria (spessore 1 px) e si
misura quanto le due linee coincidono, con una TOLLERANZA (banda di N pixel) che assorbe
gli sfasamenti di pochi pixel inevitabili.

Metriche (standard per strade/vasi/rotaie):
  - Completeness (recall):  frazione dello skeleton VERO che cade entro tolleranza
                            dallo skeleton PREDETTO  -> "quanta rotaia vera ho trovato".
  - Correctness (precision):frazione dello skeleton PREDETTO che cade entro tolleranza
                            dallo skeleton VERO      -> "quanto di cio' che predico e' rotaia".
  - F1: media armonica delle due.

Lo script prova PIU' TOLLERANZE in un colpo solo, cosi si vede l'andamento e si sceglie
con cognizione (il "ginocchio" della curva e' di norma la tolleranza giusta).

Dipendenze extra:  pip install scikit-image
Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python scripts\\valuta_rotaie_skeleton.py
    python scripts\\valuta_rotaie_skeleton.py --weights runs\\segment\\runs_nurv\\train_s_v2\\weights\\best.pt
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

try:
    from skimage.morphology import skeletonize
except ImportError:
    print("[ERRORE] Manca scikit-image. Installa con:  pip install scikit-image")
    raise SystemExit(1)

# ------------------ CONFIG ------------------
BASE = Path("data/rs19_val")
MASKS_DIR = BASE / "uint8" / "rs19_val"
VAL_IMAGES_DIR = BASE / "images" / "val"

# valori-pixel delle rotaie in RailSem19 (rail-raised + rail-embedded)
ROTAIE_PIXEL = (17, 18)
CLASSE_ROTAIE_YOLO = 0   # nel modello le rotaie sono la classe 0

# tolleranze (in pixel) da provare tutte
TOLLERANZE = [2, 3, 4, 5, 6, 8]

CONF = 0.25
IMGSZ = 640


def skeleton_da_binaria(binary_uint8: np.ndarray) -> np.ndarray:
    """Riduce una maschera binaria (0/255) alla sua linea centrale (skeleton, bool)."""
    return skeletonize(binary_uint8 > 0)


def pred_rotaie_binaria(result, h, w) -> np.ndarray:
    """Unisce tutte le istanze di classe 'rotaie' in un'unica maschera binaria 0/255."""
    out = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None:
        return out
    masks = result.masks.data.cpu().numpy()
    classi = result.boxes.cls.cpu().numpy().astype(int)
    for i in range(len(classi)):
        if classi[i] != CLASSE_ROTAIE_YOLO:
            continue
        m = masks[i]
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out[m > 0.5] = 255
    return out


def coverage(skel_a: np.ndarray, skel_b: np.ndarray, tol: int):
    """Frazione dei punti di skel_a che cadono entro 'tol' pixel da skel_b.
    Usa la distance transform di (NON skel_b): distanza di ogni pixel dallo skeleton b."""
    if skel_a.sum() == 0:
        return None  # niente skeleton A in questa immagine: non conteggiare
    if skel_b.sum() == 0:
        return 0.0   # c'e' A ma B e' vuoto: copertura nulla
    # distanza di ogni pixel dal punto piu vicino di skel_b
    inv_b = (~skel_b).astype(np.uint8)  # 0 dove c'e' skeleton, 1 altrove
    dist = cv2.distanceTransform(inv_b, cv2.DIST_L2, 3)
    d_su_a = dist[skel_a]
    entro = (d_su_a <= tol).sum()
    return entro / skel_a.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/segment/runs_nurv/train_s_v2/weights/best.pt")
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"[ERRORE] Pesi non trovati: {args.weights}  (usa --weights)")
        return

    val_imgs = sorted(VAL_IMAGES_DIR.glob("*.jpg"))
    if not val_imgs:
        print(f"[ERRORE] Nessuna immagine in {VAL_IMAGES_DIR}")
        return
    print(f"[INFO] Valuto ROTAIE (skeleton) su {len(val_imgs)} immagini.")
    print(f"[INFO] Modello: {args.weights} | conf={args.conf} | imgsz={args.imgsz}")
    print(f"[INFO] Tolleranze testate (px): {TOLLERANZE}\n")

    model = YOLO(args.weights)

    # accumulatori: per ogni tolleranza, somma delle coperture e conteggio immagini valide
    comp_sum = {t: 0.0 for t in TOLLERANZE}   # completeness (recall)
    comp_n   = {t: 0   for t in TOLLERANZE}
    corr_sum = {t: 0.0 for t in TOLLERANZE}   # correctness (precision)
    corr_n   = {t: 0   for t in TOLLERANZE}

    saltate = 0
    for idx, img_path in enumerate(val_imgs, 1):
        stem = img_path.stem
        mask_path = MASKS_DIR / f"{stem}.png"
        if not mask_path.exists():
            saltate += 1
            continue
        mask_densa = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_densa is None:
            saltate += 1
            continue
        if mask_densa.ndim == 3:
            mask_densa = mask_densa[:, :, 0]
        h, w = mask_densa.shape[:2]

        # ground truth rotaie -> skeleton (sul GT SOTTILE originale, non ingrossato)
        gt_bin = np.isin(mask_densa, ROTAIE_PIXEL).astype(np.uint8) * 255
        skel_gt = skeleton_da_binaria(gt_bin)

        # predizione rotaie -> skeleton
        res = model.predict(str(img_path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        pred_bin = pred_rotaie_binaria(res, h, w)
        skel_pred = skeleton_da_binaria(pred_bin)

        for t in TOLLERANZE:
            # completeness: quanta GT-line e' coperta dalla PRED-line
            c = coverage(skel_gt, skel_pred, t)
            if c is not None:
                comp_sum[t] += c; comp_n[t] += 1
            # correctness: quanta PRED-line e' coperta dalla GT-line
            k = coverage(skel_pred, skel_gt, t)
            if k is not None:
                corr_sum[t] += k; corr_n[t] += 1

        if idx % 100 == 0:
            print(f"  ...{idx}/{len(val_imgs)} immagini")

    print("\n" + "=" * 70)
    print(" RISULTATI — ROTAIE, valutazione SKELETON con tolleranza")
    print("=" * 70)
    print(f"{'Tol(px)':>8}{'Completeness':>15}{'Correctness':>14}{'F1':>9}")
    print("-" * 70)
    righe = []
    for t in TOLLERANZE:
        comp = comp_sum[t] / comp_n[t] if comp_n[t] else float('nan')
        corr = corr_sum[t] / corr_n[t] if corr_n[t] else float('nan')
        f1 = (2 * comp * corr / (comp + corr)) if (comp + corr) > 0 else float('nan')
        righe.append((t, comp, corr, f1))
        print(f"{t:>8}{comp:>15.3f}{corr:>14.3f}{f1:>9.3f}")
    print("=" * 70)
    if saltate:
        print(f"[nota] {saltate} immagini saltate (maschera vera non trovata).")
    print("\nCome leggere: Completeness = % della rotaia VERA trovata dal modello;")
    print("Correctness = % di cio' che il modello chiama rotaia che e' davvero rotaia.")
    print("Cerca il 'ginocchio': la tolleranza dove i valori smettono di salire molto")
    print("e' quella che assorbe il rumore di pochi pixel senza regalare punti.")


if __name__ == "__main__":
    main()
