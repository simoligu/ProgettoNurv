import numpy as np
import cv2
from typing import List, Optional, Tuple

BBox = Tuple[int, int, int, int, int]

class BackgroundModel:
    @staticmethod
    def build_median_background(frames: List[np.ndarray],
                                sample_n: int = 50,
                                resize_to: Optional[Tuple[int, int]] = None,
                                use_luminance: bool = False) -> np.ndarray:
        """
        Costruisce un'immagine di background usando la mediana per pixel.
        - frames: lista di frame BGR (HxWx3, dtype=uint8)
        - sample_n: numero massimo di frame da usare
        - resize_to: (width, height) opzionale
        - use_luminance: True per calcolare la mediana sul canale Y
        """
        if not frames:
            raise ValueError("No frames provided")

        n = min(len(frames), sample_n)
        idx = np.linspace(0, len(frames) - 1, n, dtype=int)
        sel = []
        for i in idx:
            frame = frames[i]
            if resize_to is not None:
                frame = cv2.resize(frame, resize_to)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            sel.append(frame.astype(np.uint8))

        arr = np.stack(sel, axis=0)  # (N, H, W, 3)

        if use_luminance:
            ycrcb = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb) for f in arr])
            median_ycrcb = np.median(ycrcb, axis=0).astype(np.uint8)
            median_bgr = cv2.cvtColor(median_ycrcb, cv2.COLOR_YCrCb2BGR)
            return median_bgr
        else:
            return np.median(arr, axis=0).astype(np.uint8)

    @staticmethod
    def estimate_luminance(frame: np.ndarray) -> float:
        """Restituisce la luminanza media (canale Y) del frame"""
        if frame is None:
            return 0.0
        y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[..., 0] if frame.ndim == 3 else frame
        return float(np.mean(y))


class AdaptiveBackground:
    """
    Background subtractor adattivo wrapper (MOG2/KNN) con warm-up e update
    """

    def __init__(self,
                 method: str = 'MOG2',
                 history: int = 500,
                 var_threshold: float = 16.0,
                 detect_shadows: bool = True,
                 resize_to: Optional[Tuple[int, int]] = None):
        self.resize_to = resize_to
        self.method = method.upper()
        self.detect_shadows = detect_shadows

        if self.method == 'MOG2':
            self.sub = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold, detectShadows=self.detect_shadows)
        elif self.method == 'KNN':
            self.sub = cv2.createBackgroundSubtractorKNN(
                history=history, dist2Threshold=var_threshold, detectShadows=self.detect_shadows)
        else:
            raise ValueError("method must be 'MOG2' or 'KNN'")

        self._bg_img: Optional[np.ndarray] = None

    def initialize(self, frames: List[np.ndarray], sample_n: int = 50, learning_rate: float = 0.5):
        """Warm-up del modello usando alcuni frame"""
        if not frames:
            return None
        n = min(len(frames), sample_n)
        idx = np.linspace(0, len(frames) - 1, n, dtype=int)
        for i in idx:
            frame = frames[i]
            if self.resize_to:
                frame = cv2.resize(frame, self.resize_to)
            self.sub.apply(frame, learningRate=learning_rate)
        return self.get_background()

    def apply(self, frame: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        """Applica il subtractor e restituisce la mask binaria 0/255"""
        if frame is None:
            return np.zeros((0, 0), dtype=np.uint8)

        f = cv2.resize(frame, self.resize_to) if self.resize_to else frame
        lr = -1 if learning_rate is None else float(learning_rate)
        fg = self.sub.apply(f, learningRate=lr)
        if self.detect_shadows:
            fg[fg == 127] = 0
        return (fg > 0).astype(np.uint8) * 255

    def get_background(self) -> Optional[np.ndarray]:
        """Restituisce l'immagine di background corrente"""
        return self.sub.getBackgroundImage()

    def update_with_weighted(self, frame: np.ndarray, alpha: float = 0.01):
        """Aggiorna background usando blending esponenziale"""
        if self._bg_img is None:
            self._bg_img = frame.copy().astype(np.float32)
            return
        self._bg_img = cv2.addWeighted(frame.astype(np.float32), alpha, self._bg_img, 1 - alpha, 0)

    def get_weighted_background(self) -> Optional[np.ndarray]:
        if self._bg_img is not None:
            return np.clip(self._bg_img, 0, 255).astype(np.uint8)
        return None


def mask_to_boxes(mask: np.ndarray, diff_thresh: int = 40, min_area: int = 500,
                  min_compattezza: float = None
                  ) -> Tuple[List[Tuple[int, int, int, int, int]], np.ndarray]:
    """
    Estrae bounding boxes da una mask (binaria o diff image)
    Ritorna sempre (boxes, mask) con mask np.ndarray

    min_compattezza: se fornito (0-1), filtra anche sulla COMPATTEZZA del blob
    (area_contorno / area_bounding_box) — un rumore da disallineamento
    omografico tende a formare linee sottili e allungate lungo i contorni
    (bordi di edifici, binari), con compattezza bassa (bounding box molto
    piu' grande dell'area reale occupata); un'anomalia strutturale vera
    (frana, cedimento, accumulo) tende a essere piu' compatta/tondeggiante.
    None (default) disattiva il filtro — comportamento invariato per chi
    chiama questa funzione senza il nuovo parametro.
    """
    boxes: List[Tuple[int, int, int, int, int]] = []

    if mask is None or mask.size == 0:
        th = np.zeros((0, 0), dtype=np.uint8)
        return boxes, th

    # converte in grayscale se necessario
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.ndim == 3 else mask

    # threshold
    if mask_gray.max() <= 1:
        th = (mask_gray * 255).astype(np.uint8)
    else:
        _, th = cv2.threshold(mask_gray, diff_thresh, 255, cv2.THRESH_BINARY)

    # morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # estrae contorni
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if min_compattezza is not None:
            area_bbox = w * h
            compattezza = (area / area_bbox) if area_bbox > 0 else 0.0
            if compattezza < min_compattezza:
                continue
        boxes.append((x, y, w, h, int(area)))

    return boxes, th