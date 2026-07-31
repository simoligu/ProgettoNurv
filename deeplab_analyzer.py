# -*- coding: utf-8 -*-
"""
DeepLabAnalyzer — Analisi infrastruttura ferroviaria basata su DeepLabV3+.

Sostituisce la visione classica (Hough per rotaie, gradienti per pali,
HSV per vegetazione) con segmentazione semantica + post-processing geometrico.

Classe unica che:
  1. Carica il modello DeepLabV3+ addestrato (best.pt)
  2. Segmenta un frame del video -> mappa di classi (sfondo/rotaie/pali/vegetazione)
  3. Analizza la mappa per rilevare anomalie:
     - Scartamento fuori tolleranza (distanza tra le rotaie)
     - Pali inclinati (angolo dalla verticale)
     - Vegetazione invasiva (prossimita' ai binari)

Uso nella pipeline:
    analyzer = DeepLabAnalyzer("runs_seg/deeplab_hires/best.pt")
    anomalies = analyzer.analyze(frame_bgr)
    for a in anomalies:
        print(a["label"], a["severity"], a["details"])
"""

import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
from typing import List, Dict, Any, Optional, Tuple

# --- COSTANTI MODELLO ---
NUM_CLASSI = 4
CL_SFONDO = 0
CL_ROTAIE = 1
CL_PALI = 2
CL_VEGETAZIONE = 3

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DeepLabAnalyzer:
    """Analizzatore infrastruttura ferroviaria basato su DeepLabV3+."""

    def __init__(self,
                 weights_path: str,
                 backbone: str = "resnet34",
                 imgsz: int = 896,
                 device: Optional[str] = None,
                 # --- parametri anomalie ---
                 expected_gauge_px: int = 150,
                 gauge_tolerance: float = 0.15,
                 max_tilt_deg: float = 8.0,
                 veg_proximity_px: int = 80,
                 veg_density_threshold: float = 0.03):
        """
        Args:
            weights_path: percorso al best.pt del modello DeepLabV3+
            backbone: encoder usato in training (resnet34)
            imgsz: risoluzione di inferenza (deve corrispondere al training)
            device: "cuda", "cpu", o None per auto-detect
            expected_gauge_px: larghezza attesa tra le rotaie in pixel (da calibrare per il video)
            gauge_tolerance: tolleranza percentuale sullo scartamento (0.15 = 15%)
            max_tilt_deg: inclinazione massima accettabile per un palo (gradi)
            veg_proximity_px: distanza in pixel dai binari entro cui la vegetazione e' "invasiva"
            veg_density_threshold: densita' minima di vegetazione nella zona critica per generare alert
        """
        # device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.imgsz = imgsz
        self.expected_gauge_px = expected_gauge_px
        self.gauge_tolerance = gauge_tolerance
        self.max_tilt_deg = max_tilt_deg
        self.veg_proximity_px = veg_proximity_px
        self.veg_density_threshold = veg_density_threshold

        # carica modello
        self.model = self._load_model(weights_path, backbone)
        print(f"[DeepLabAnalyzer] Modello caricato da {weights_path} | device={self.device}")

    def _load_model(self, weights_path: str, backbone: str) -> torch.nn.Module:
        """Carica il DeepLabV3+ dal checkpoint (formato di train_deeplab.py)."""
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSI,
        )
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()

        epoca = ckpt.get("epoca", "?")
        miou = ckpt.get("best_miou", "?")
        if isinstance(miou, float):
            miou = f"{miou:.3f}"
        print(f"[DeepLabAnalyzer] Epoca {epoca}, mIoU {miou}")
        return model

    # ------------------------------------------------------------------
    # SEGMENTAZIONE
    # ------------------------------------------------------------------

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Segmenta un frame BGR -> mappa di classi (H, W) con valori 0-3.

        La predizione avviene alla risoluzione imgsz (quadrata), poi viene
        ridimensionata alla risoluzione originale del frame.
        """
        h, w = frame_bgr.shape[:2]

        # preprocess: BGR -> RGB, resize, normalizza, tensore
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # inferenza
        with torch.no_grad():
            logits = self.model(tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # riporta alla dimensione originale
        if pred.shape[0] != h or pred.shape[1] != w:
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        return pred

    # ------------------------------------------------------------------
    # ANALISI COMPLETA
    # ------------------------------------------------------------------

    def analyze(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Esegue segmentazione + analisi anomalie su un frame.
        Ritorna lista di anomalie, ognuna con: label, severity, conf, details, bbox.
        """
        class_map = self.segment(frame_bgr)
        h, w = class_map.shape

        anomalies = []
        anomalies.extend(self._analyze_gauge(class_map, h, w))
        anomalies.extend(self._analyze_poles(class_map, h, w))
        anomalies.extend(self._analyze_vegetation(class_map, h, w))
        return anomalies

    # ------------------------------------------------------------------
    # 1. SCARTAMENTO (distanza tra le rotaie)
    # ------------------------------------------------------------------

    def _analyze_gauge(self, class_map: np.ndarray, h: int, w: int) -> List[Dict[str, Any]]:
        """
        Misura lo scartamento: distanza in pixel tra le due rotaie.

        Strategia: nella fascia inferiore dell'immagine (dove le rotaie sono grandi
        e ben segmentate), scansiona righe orizzontali e trova i "run" di pixel-rotaia.
        Se ci sono almeno due run separati, misura la distanza tra le loro bordi interni.
        La mediana su piu' righe da' una stima robusta.
        """
        anomalies = []
        rail_mask = (class_map == CL_ROTAIE).astype(np.uint8)

        # lavora nella fascia inferiore (60%-90% dell'altezza)
        y_start = int(h * 0.60)
        y_end = int(h * 0.90)

        # campiona alcune righe in questa fascia
        rows_to_check = np.linspace(y_start, y_end, num=20, dtype=int)
        distances = []

        for y in rows_to_check:
            row = rail_mask[y, :]
            runs = self._find_runs(row)
            if len(runs) >= 2:
                # prendi le due run piu' larghe (le due rotaie principali)
                runs_sorted = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)
                r1, r2 = sorted(runs_sorted[:2], key=lambda r: r[0])  # ordina per posizione x
                # distanza tra bordo destro della rotaia sinistra e bordo sinistro della destra
                gap = r2[0] - r1[1]
                if gap > 10:  # filtro minimo per evitare run adiacenti
                    distances.append(gap)

        if not distances:
            return anomalies

        gauge = float(np.median(distances))
        deviation = abs(gauge - self.expected_gauge_px)
        tolerance_px = self.expected_gauge_px * self.gauge_tolerance

        if deviation > tolerance_px:
            severity = "CRITICA" if deviation > tolerance_px * 2 else "ALTA"
            anomalies.append({
                "label": "SCARTAMENTO_ANOMALO",
                "severity": severity,
                "conf": min(deviation / (self.expected_gauge_px * 0.5), 1.0),
                "details": f"Scartamento: {gauge:.0f}px (atteso: {self.expected_gauge_px}px, "
                           f"deviazione: {deviation:.0f}px, tolleranza: {tolerance_px:.0f}px)",
                "bbox": (0, y_start, w, y_end - y_start),
            })

        return anomalies

    @staticmethod
    def _find_runs(binary_row: np.ndarray) -> List[Tuple[int, int]]:
        """
        Trova le "run" consecutive di 1 in un vettore binario.
        Ritorna lista di (start_x, end_x) per ogni run.
        """
        runs = []
        in_run = False
        start = 0
        for i, val in enumerate(binary_row):
            if val and not in_run:
                start = i
                in_run = True
            elif not val and in_run:
                runs.append((start, i))
                in_run = False
        if in_run:
            runs.append((start, len(binary_row)))
        return runs

    # ------------------------------------------------------------------
    # 2. PALI INCLINATI
    # ------------------------------------------------------------------

    def _analyze_poles(self, class_map: np.ndarray, h: int, w: int) -> List[Dict[str, Any]]:
        """
        Rileva pali inclinati analizzando le componenti connesse della classe 'pali'.

        Per ogni componente abbastanza grande e verticale, calcola l'angolo rispetto
        alla verticale usando minAreaRect di OpenCV. Se l'inclinazione supera la soglia,
        genera un alert.
        """
        anomalies = []
        pole_mask = (class_map == CL_PALI).astype(np.uint8) * 255

        # pulizia morfologica: chiudi piccoli gap
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pole_mask = cv2.morphologyEx(pole_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # trova componenti connesse
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            pole_mask, connectivity=8
        )

        for i in range(1, num_labels):  # salta lo sfondo (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            # filtro: deve essere abbastanza grande e piu' alto che largo (verticale)
            if area < 500 or bh < h * 0.15 or bh < bw * 2:
                continue

            # estrai i pixel della componente e calcola il rettangolo minimo ruotato
            points = np.column_stack(np.where(labels == i))  # (y, x) pairs
            points_xy = points[:, ::-1].astype(np.float32)  # converti a (x, y)

            if len(points_xy) < 5:
                continue

            rect = cv2.minAreaRect(points_xy)
            # rect: (center, (width, height), angle)
            rect_w, rect_h = rect[1]
            angle = rect[2]

            # minAreaRect restituisce angoli tra -90 e 0:
            # un oggetto verticale ha width < height e angle vicino a 0 o -90
            # calcoliamo l'inclinazione dalla verticale
            if rect_w > rect_h:
                # l'oggetto e' stato misurato "coricato", correggiamo
                rect_w, rect_h = rect_h, rect_w
                angle = angle + 90

            tilt = abs(angle) if abs(angle) <= 45 else abs(90 - abs(angle))

            if tilt > self.max_tilt_deg:
                bx = stats[i, cv2.CC_STAT_LEFT]
                by = stats[i, cv2.CC_STAT_TOP]

                if tilt >= self.max_tilt_deg * 2:
                    severity = "CRITICA"
                elif tilt >= self.max_tilt_deg:
                    severity = "ALTA"
                else:
                    severity = "MEDIO"

                anomalies.append({
                    "label": "PALO_INCLINATO",
                    "severity": severity,
                    "conf": min(tilt / (self.max_tilt_deg * 3), 1.0),
                    "details": f"Inclinazione: {tilt:.1f} gradi (soglia: {self.max_tilt_deg} gradi)",
                    "bbox": (bx, by, bw, bh),
                })

        return anomalies

    # ------------------------------------------------------------------
    # 3. VEGETAZIONE INVASIVA
    # ------------------------------------------------------------------

    def _analyze_vegetation(self, class_map: np.ndarray, h: int, w: int) -> List[Dict[str, Any]]:
        """
        Rileva vegetazione troppo vicina ai binari.

        Strategia: usa la distance transform dalla maschera delle rotaie per calcolare
        la distanza di ogni pixel dai binari. Poi conta quanti pixel di vegetazione
        cadono entro la soglia di prossimita'. Se la densita' supera la soglia -> alert.

        Molto piu' robusto dell'approccio HSV perche':
        - DeepLab sa cos'e' davvero vegetazione (non qualsiasi cosa verde)
        - la prossimita' e' misurata rispetto ai binari segmentati, non a linee di Hough
        """
        anomalies = []
        rail_mask = (class_map == CL_ROTAIE).astype(np.uint8)
        veg_mask = (class_map == CL_VEGETAZIONE).astype(np.uint8)

        # se non ci sono rotaie o vegetazione, niente da analizzare
        if rail_mask.sum() == 0 or veg_mask.sum() == 0:
            return anomalies

        # distance transform: per ogni pixel, distanza dal piu' vicino pixel-rotaia
        # (invertiamo: distanceTransform misura la distanza dai pixel a 0)
        rail_inv = (1 - rail_mask) * 255
        dist = cv2.distanceTransform(rail_inv, cv2.DIST_L2, 5)

        # vegetazione nella zona critica (entro veg_proximity_px dai binari)
        veg_near = (veg_mask > 0) & (dist < self.veg_proximity_px)
        n_veg_near = int(veg_near.sum())

        # densita': rapporto tra vegetazione vicina e area della fascia di prossimita'
        proximity_zone = (dist < self.veg_proximity_px)
        zone_area = int(proximity_zone.sum())
        if zone_area == 0:
            return anomalies

        density = n_veg_near / zone_area

        if density > self.veg_density_threshold:
            # calcola la copertura percentuale per il messaggio
            coverage_pct = density * 100

            if density > self.veg_density_threshold * 3:
                severity = "CRITICA"
            elif density > self.veg_density_threshold * 1.5:
                severity = "ALTA"
            else:
                severity = "MEDIO"

            # bbox: area dove c'e' vegetazione vicina ai binari
            ys, xs = np.where(veg_near)
            if len(xs) > 0:
                bx, by = int(xs.min()), int(ys.min())
                bw_v = int(xs.max() - xs.min())
                bh_v = int(ys.max() - ys.min())
            else:
                bx, by, bw_v, bh_v = 0, 0, w, h

            anomalies.append({
                "label": "VEGETAZIONE_INVASIVA",
                "severity": severity,
                "conf": min(density / (self.veg_density_threshold * 5), 1.0),
                "details": f"Vegetazione entro {self.veg_proximity_px}px dai binari: "
                           f"{coverage_pct:.1f}% della zona critica "
                           f"(soglia: {self.veg_density_threshold * 100:.1f}%)",
                "bbox": (bx, by, bw_v, bh_v),
            })

        return anomalies

    # ------------------------------------------------------------------
    # AUTO-CALIBRAZIONE SCARTAMENTO
    # ------------------------------------------------------------------

    def measure_gauge(self, class_map: np.ndarray) -> Optional[float]:
        """
        Misura lo scartamento in pixel dalla mappa di classi.
        Ritorna il valore mediano della distanza tra le rotaie, o None se non rilevabile.
        Usato per l'auto-calibrazione: si segmenta un frame del reference video,
        si misura lo scartamento, e quel valore diventa il baseline.
        """
        h, w = class_map.shape
        rail_mask = (class_map == CL_ROTAIE).astype(np.uint8)

        y_start = int(h * 0.60)
        y_end = int(h * 0.90)
        rows_to_check = np.linspace(y_start, y_end, num=20, dtype=int)
        distances = []

        for y in rows_to_check:
            row = rail_mask[y, :]
            runs = self._find_runs(row)
            if len(runs) >= 2:
                runs_sorted = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)
                r1, r2 = sorted(runs_sorted[:2], key=lambda r: r[0])
                gap = r2[0] - r1[1]
                if gap > 10:
                    distances.append(gap)

        if distances:
            return float(np.median(distances))
        return None

    # ------------------------------------------------------------------
    # UTILITA' (per debug e tesi)
    # ------------------------------------------------------------------

    def get_colored_overlay(self, frame_bgr: np.ndarray,
                            class_map: np.ndarray,
                            alpha: float = 0.5) -> np.ndarray:
        """
        Genera un'immagine con la segmentazione sovrapposta.
        Utile per debug e per le figure della tesi.
        Colori: rotaie=rosso, pali=blu, vegetazione=verde.
        """
        COLORI = {CL_ROTAIE: (0, 0, 255), CL_PALI: (255, 0, 0), CL_VEGETAZIONE: (0, 200, 0)}
        out = frame_bgr.copy()
        layer = np.zeros_like(frame_bgr)
        for c, col in COLORI.items():
            layer[class_map == c] = col
        mask_any = (class_map > 0)
        out[mask_any] = cv2.addWeighted(frame_bgr, 1 - alpha, layer, alpha, 0)[mask_any]
        return out


# ------------------------------------------------------------------
# TEST STANDALONE
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test DeepLabAnalyzer su un'immagine singola")
    ap.add_argument("--weights", default="runs_seg/deeplab_hires/best.pt")
    ap.add_argument("--image", required=True, help="Percorso a un'immagine di test")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--gauge", type=int, default=150, help="Scartamento atteso in pixel")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERRORE] Immagine non trovata: {args.image}")
        exit(1)

    analyzer = DeepLabAnalyzer(
        weights_path=args.weights,
        imgsz=args.imgsz,
        expected_gauge_px=args.gauge,
    )

    print(f"\n[TEST] Segmentazione di {args.image} ({img.shape[1]}x{img.shape[0]})...")
    class_map = analyzer.segment(img)

    # stampa distribuzione classi
    nomi = {0: "sfondo", 1: "rotaie", 2: "pali", 3: "vegetazione"}
    total = class_map.size
    for c in range(NUM_CLASSI):
        n = int((class_map == c).sum())
        print(f"  {nomi[c]:<12}: {n:>8} pixel ({100*n/total:.1f}%)")

    print(f"\n[TEST] Analisi anomalie...")
    anomalies = analyzer.analyze(img)
    if anomalies:
        for a in anomalies:
            print(f"  [{a['severity']}] {a['label']}: {a['details']}")
    else:
        print("  Nessuna anomalia rilevata.")

    # salva overlay
    overlay = analyzer.get_colored_overlay(img, class_map)
    out_path = "test_deeplab_overlay.jpg"
    cv2.imwrite(out_path, overlay)
    print(f"\n[TEST] Overlay salvato in: {out_path}")