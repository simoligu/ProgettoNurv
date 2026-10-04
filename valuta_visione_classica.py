# -*- coding: utf-8 -*-
"""
Valutazione della diagnostica a visione classica ereditata sul ground truth di
RailSem19, e confronto con DeepLabV3+ sulle stesse immagini (capitolo 10).

Il metodo classico non era mai stato misurato: sui video non esiste un ground
truth. Qui lo si esegue in isolamento, con i parametri del codice ereditato,
sulle 1700 immagini di validazione. Due metodi distinti, tenuti separati:

  A. la diagnostica ESEGUITA dal codice consegnato (commit a3f700e, corpo del
     metodo run della pipeline):
       - pali: Canny + HoughLinesP (soglia 50, lunghezza minima 100, gap 30)
         nelle due bande laterali del 30%, segmenti fra 80 e 100 gradi e alti
         piu' di un quarto dell'immagine; inclinazione theta, segnalazione ALTA
         per 3 <= theta < 8, CRITICA per theta >= 8;
       - vegetazione: soglia HSV [20,30,20]-[95,255,255] sull'intero fotogramma,
         segnalazione se i pixel verdi superano il 20% (CRITICA oltre il 40%);
  B. la logica di StructuralDetector.process_frame, presente nel repository ma
     mai invocata: segmenti di Hough classificati per angolo in rotaie e pali,
     larghezza fra rotaia piu' a sinistra e piu' a destra a meta' altezza.

Per DeepLab (runs_seg/deeplab_hires/best.pt, 896 px, su CPU circa 2 s per
immagine) si calcolano sulle stesse immagini: IoU della vegetazione (controllo di
coerenza con la tabella del capitolo 5), la stessa regola del 20% applicata alla
sua maschera, e completezza / correttezza dello skeleton delle rotaie.

Scrive il resoconto in UTF-8 in risultati_tesi/valutazione_visione_classica.txt.

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python valuta_visione_classica.py
    python valuta_visione_classica.py --limite 20 --senza_deeplab   # prova rapida
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

BASE = Path("data/rs19_val")
MASK_DIR = BASE / "uint8" / "rs19_val"
VAL_DIR = BASE / "images" / "val"

ROTAIE = (17, 18)
PALO = 5
VEG = 8
TOLLERANZE = [2, 3, 4, 5, 6, 8]
TOL = 4                    # px: un segmento "cade su" una classe se vi cade per almeno meta'
SOGLIA_VEG = 20.0          # % del fotogramma, come nel codice ereditato
SOGLIA_VEG_CRIT = 40.0


# ---------------------------------------------------------------- utilita'
def hough(edges, soglia, lung, gap):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, soglia, minLineLength=lung, maxLineGap=gap)
    # OpenCV 4 restituisce (N, 1, 4), OpenCV 5 (N, 4): con il secondo formato il
    # codice ereditato (x1, y1, x2, y2 = l[0]) si interrompe con TypeError.
    return [] if lines is None else [tuple(int(v) for v in l) for l in lines.reshape(-1, 4)]


def distanza_da(mask_bool):
    return cv2.distanceTransform((~mask_bool).astype(np.uint8), cv2.DIST_L2, 3)


def frazione_su(seg, dist, shape):
    x1, y1, x2, y2 = seg
    n = max(abs(x2 - x1), abs(y2 - y1)) + 1
    xs = np.clip(np.rint(np.linspace(x1, x2, n)).astype(int), 0, shape[1] - 1)
    ys = np.clip(np.rint(np.linspace(y1, y2, n)).astype(int), 0, shape[0] - 1)
    return float((dist[ys, xs] <= TOL).mean())


def raster(segmenti, shape):
    m = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in segmenti:
        cv2.line(m, (x1, y1), (x2, y2), 255, 1)
    return m > 0


def copertura(a, dist_b, tol):
    if a.sum() == 0:
        return None
    return float((dist_b[a] <= tol).sum()) / float(a.sum())


class Pixel:
    def __init__(self):
        self.i = self.u = self.p = self.g = 0

    def add(self, pred, gt):
        self.i += int((pred & gt).sum()); self.u += int((pred | gt).sum())
        self.p += int(pred.sum()); self.g += int(gt.sum())

    def riga(self, nome):
        f = lambda a, b: a / b if b else float("nan")
        return (f"   {nome:34s} IoU {f(self.i, self.u):.3f}   precision {f(self.i, self.p):.3f}"
                f"   recall {f(self.i, self.g):.3f}")


class Skel:
    def __init__(self):
        self.comp = {t: [] for t in TOLLERANZE}
        self.corr = {t: [] for t in TOLLERANZE}

    def add(self, pred_bool, skel_gt, d_skel):
        if skel_gt.sum() == 0:
            return
        sp = skeletonize(pred_bool) if pred_bool.any() else pred_bool
        d_p = distanza_da(sp) if sp.any() else None
        for t in TOLLERANZE:
            self.comp[t].append(0.0 if d_p is None else copertura(skel_gt, d_p, t))
            c = copertura(sp, d_skel, t)
            if c is not None:
                self.corr[t].append(c)

    def righe(self):
        out = []
        for t in TOLLERANZE:
            c = float(np.mean(self.comp[t])) if self.comp[t] else float("nan")
            r = float(np.mean(self.corr[t])) if self.corr[t] else float("nan")
            f1 = 2 * c * r / (c + r) if (c + r) > 0 else 0.0
            out.append(f"   tol {t} px: completezza {c:.3f}   correttezza {r:.3f}   F1 {f1:.3f}")
        return out


class Decisione:
    """Matrice di confusione della regola 'vegetazione > 20% del fotogramma',
    applicata a una maschera predetta contro la stessa regola sul ground truth."""
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def add(self, pred_si, gt_si):
        if pred_si and gt_si: self.tp += 1
        elif pred_si: self.fp += 1
        elif gt_si: self.fn += 1
        else: self.tn += 1

    def riga(self, nome):
        pr = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")
        rc = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")
        return (f"   {nome:10s} segnala {self.tp + self.fp:4d}   giuste {self.tp:4d}   in piu' {self.fp:4d}"
                f"   mancate {self.fn:4d}   precision {pr:.3f}   recall {rc:.3f}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--senza_deeplab", action="store_true")
    ap.add_argument("--deeplab", default="runs_seg/deeplab_hires/best.pt")
    ap.add_argument("--out", default="risultati_tesi/valutazione_visione_classica.txt")
    args = ap.parse_args()

    deeplab = None
    if not args.senza_deeplab:
        from deeplab_analyzer import DeepLabAnalyzer
        deeplab = DeepLabAnalyzer(weights_path=args.deeplab, imgsz=896)

    immagini = sorted(VAL_DIR.glob("*.jpg"))
    if args.limite:
        immagini = immagini[:args.limite]

    # A. metodo eseguito
    veg_hsv = {"intera": Pixel(), "roi33": Pixel()}
    dec_hsv, dec_dl = Decisione(), Decisione()
    n_gt_sopra, n_hsv_crit = 0, 0
    pali_cand = pali_cand_su_palo = 0
    alert_alta = alert_crit = alert_su_palo = 0
    img_con_alert = 0
    img_pali_bande = img_pali_bande_trovati = 0
    # B. StructuralDetector
    skel_hough = Skel()
    cat_su_rot = {"rotaia": 0, "palo": 0, "scartato": 0}
    cat_su_palo = {"rotaia": 0, "palo": 0, "scartato": 0}
    n_misure = n_estremi_ok = 0
    # DeepLab
    veg_dl = {"intera": Pixel(), "roi33": Pixel()}
    skel_dl = Skel()
    n_img = 0

    for k, p in enumerate(immagini, 1):
        frame = cv2.imread(str(p))
        g = cv2.imread(str(MASK_DIR / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if frame is None or g is None:
            continue
        if g.ndim == 3:
            g = g[..., 0]
        h, w = g.shape
        n_img += 1
        r0 = h - int(h * 33 / 100)

        gt_veg = g == VEG
        gt_rot = np.isin(g, ROTAIE)
        gt_palo = g == PALO
        d_rot = distanza_da(gt_rot) if gt_rot.any() else None
        d_palo = distanza_da(gt_palo) if gt_palo.any() else None
        skel_gt = skeletonize(gt_rot)
        d_skel = distanza_da(skel_gt) if skel_gt.any() else None
        gt_sopra = 100.0 * gt_veg.mean() > SOGLIA_VEG
        n_gt_sopra += int(gt_sopra)

        # ---- A. vegetazione eseguita
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        verde = cv2.inRange(hsv, np.array([20, 30, 20]), np.array([95, 255, 255])) > 0
        veg_hsv["intera"].add(verde, gt_veg)
        veg_hsv["roi33"].add(verde[r0:], gt_veg[r0:])
        perc = 100.0 * verde.mean()
        dec_hsv.add(perc > SOGLIA_VEG, gt_sopra)
        n_hsv_crit += int(perc > SOGLIA_VEG_CRIT)

        # ---- A. pali eseguiti
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = np.zeros((h, w), np.uint8)
        cv2.rectangle(roi, (0, 0), (int(w * 0.30), h), 255, -1)
        cv2.rectangle(roi, (int(w * 0.70), 0), (w, h), 255, -1)
        segs = hough(cv2.bitwise_and(cv2.Canny(gray, 50, 150), roi), 50, 100, 30)
        trovato, allarme = False, False
        for (x1, y1, x2, y2) in segs:
            dx, dy = x2 - x1, y2 - y1
            ang = abs(np.degrees(np.arctan2(dy, dx)))
            if not (80 < ang < 100) or abs(dy) <= h * 0.25:
                continue
            pali_cand += 1
            su_palo = d_palo is not None and frazione_su((x1, y1, x2, y2), d_palo, (h, w)) >= 0.5
            pali_cand_su_palo += int(su_palo)
            trovato |= su_palo
            theta = abs(np.degrees(np.arctan(dx / dy))) if dy != 0 else 0.0
            if theta >= 3:
                allarme = True
                alert_crit += int(theta >= 8)
                alert_alta += int(theta < 8)
                alert_su_palo += int(su_palo)
        img_con_alert += int(allarme)
        bande = np.zeros((h, w), bool); bande[:, :int(w * 0.30)] = True; bande[:, int(w * 0.70):] = True
        if (gt_palo & bande).any():
            img_pali_bande += 1
            img_pali_bande_trovati += int(trovato)

        # ---- B. StructuralDetector.process_frame (mai invocato)
        tutti = hough(cv2.Canny(gray, 50, 150, apertureSize=3), 100, 100, 10)
        rail = []
        for s in tutti:
            x1, y1, x2, y2 = s
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            cat = "palo" if 80 < ang < 100 else ("rotaia" if (ang < 40 or ang > 140) else "scartato")
            if cat == "rotaia":
                rail.append(s)
            if d_rot is not None and frazione_su(s, d_rot, (h, w)) >= 0.5:
                cat_su_rot[cat] += 1
            elif d_palo is not None and frazione_su(s, d_palo, (h, w)) >= 0.5:
                cat_su_palo[cat] += 1
        if d_skel is not None:
            skel_hough.add(raster(rail, (h, w)), skel_gt, d_skel)
        if len(rail) >= 2:
            n_misure += 1
            ls = sorted(rail, key=lambda l: (l[0] + l[2]) / 2)
            mid = h // 2
            ok = d_rot is not None
            for (x1, y1, x2, y2) in (ls[0], ls[-1]):
                x = (x1 + x2) / 2 if abs(y2 - y1) < 1 else x1 + (x2 - x1) * (mid - y1) / (y2 - y1)
                x = int(round(x))
                if not ok or not (0 <= x < w) or d_rot[mid, x] > TOL:
                    ok = False
            n_estremi_ok += int(ok)

        # ---- DeepLab
        if deeplab is not None:
            m = deeplab.segment(frame)
            pv = m == 3
            veg_dl["intera"].add(pv, gt_veg)
            veg_dl["roi33"].add(pv[r0:], gt_veg[r0:])
            dec_dl.add(100.0 * pv.mean() > SOGLIA_VEG, gt_sopra)
            if d_skel is not None:
                skel_dl.add(m == 1, skel_gt, d_skel)

        if k % 50 == 0:
            print(f"  {k}/{len(immagini)}", flush=True)

    L = [f"Diagnostica a visione classica contro DeepLabV3+ su RailSem19, {n_img} immagini di validazione", ""]
    L.append("A. METODO ESEGUITO DAL CODICE CONSEGNATO (a3f700e, corpo di run)")
    L.append("")
    L.append("A1. Vegetazione, maschera HSV [20,30,20]-[95,255,255] contro vegetation (8)")
    L.append(veg_hsv["intera"].riga("HSV, immagine intera"))
    L.append(veg_hsv["roi33"].riga("HSV, terzo inferiore"))
    if deeplab is not None:
        L.append(veg_dl["intera"].riga("DeepLab, immagine intera"))
        L.append(veg_dl["roi33"].riga("DeepLab, terzo inferiore"))
    L.append("")
    L.append(f"A2. Regola 'vegetazione > {SOGLIA_VEG:.0f}% del fotogramma' (quella del codice ereditato)")
    L.append(f"   immagini in cui la regola scatta sul ground truth: {n_gt_sopra}")
    L.append(dec_hsv.riga("HSV"))
    if deeplab is not None:
        L.append(dec_dl.riga("DeepLab"))
    L.append(f"   immagini con HSV oltre il {SOGLIA_VEG_CRIT:.0f}% (severita' CRITICA): {n_hsv_crit}")
    L.append("")
    L.append("A3. Pali: segmenti candidati (bande laterali 30%, 80-100 gradi, alti > h/4)")
    L.append(f"   candidati: {pali_cand}; di cui su pali veri: {pali_cand_su_palo}"
             f" ({100 * pali_cand_su_palo / max(pali_cand, 1):.1f}%)")
    L.append(f"   segnalazioni (theta >= 3): {alert_alta + alert_crit}  (ALTA {alert_alta}, CRITICA {alert_crit});"
             f" su pali veri: {alert_su_palo}")
    L.append(f"   immagini con almeno una segnalazione: {img_con_alert}")
    L.append(f"   immagini con pali veri nelle bande: {img_pali_bande}; con almeno un candidato su un palo vero:"
             f" {img_pali_bande_trovati}")
    L.append("")
    L.append("B. LOGICA DI StructuralDetector.process_frame (presente, mai invocata)")
    L.append("")
    tr, tp = sum(cat_su_rot.values()), sum(cat_su_palo.values())
    L.append(f"B1. Classificazione per angolo dei segmenti di Hough")
    L.append(f"   segmenti su rotaie vere: {tr}")
    for c in ("rotaia", "palo", "scartato"):
        L.append(f"      classificati '{c}': {cat_su_rot[c]} ({100 * cat_su_rot[c] / max(tr, 1):.1f}%)")
    L.append(f"   segmenti su pali veri: {tp}")
    for c in ("rotaia", "palo", "scartato"):
        L.append(f"      classificati '{c}': {cat_su_palo[c]} ({100 * cat_su_palo[c] / max(tp, 1):.1f}%)")
    L.append("")
    L.append("B2. Rotaie: skeleton dei segmenti 'rotaia' contro skeleton del ground truth")
    L += skel_hough.righe()
    if deeplab is not None:
        L.append("   stessa misura per DeepLab (classe rotaie):")
        L += skel_dl.righe()
    L.append("")
    L.append("B3. Larghezza fra rotaia piu' a sinistra e piu' a destra, a meta' altezza")
    L.append(f"   immagini con misura: {n_misure}; con entrambi gli estremi entro {TOL} px da una rotaia vera: {n_estremi_ok}")

    testo = "\n".join(L)
    print("\n" + testo)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(testo + "\n", encoding="utf-8")
    print(f"\n[FATTO] {out}")


if __name__ == "__main__":
    main()
