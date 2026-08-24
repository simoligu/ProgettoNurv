# -*- coding: utf-8 -*-
"""
PoleTiltAnalyzer — Stima fine dell'inclinazione di un palo da un crop RGB,
usando il regressore ResNet34 addestrato su dataset sintetico Unreal Engine
(vedi train_pole_tilt.py e NURV_pali_inclinati_diario.md per la storia
completa dell'addestramento — test MAE 1.24° +/- 0.21° su asset mai visto
in training).

RUOLO NELLA PIPELINE: modulo indipendente che AFFIANCA il rilevamento
geometrico gia' presente in DeepLabAnalyzer._analyze_poles() (minAreaRect
sulla maschera di segmentazione). Il flusso e':

    1. DeepLabAnalyzer.analyze() riceve questo modulo come tilt_estimator
       (il metodo predict_angle, passato come funzione — nessun import
       incrociato tra i due moduli)
    2. Per OGNI componente "a forma di palo" individuato nella maschera
       (stesso filtro geometrico di area/verticalita' di sempre, che serve
       comunque a trovare DOVE guardare), DeepLabAnalyzer ritaglia il bbox
       dal frame BGR e chiama questo predict_angle()
    3. L'angolo della CNN (se il crop e' valido) sostituisce quello di
       minAreaRect per decidere se generare l'alert e con quale severity —
       minAreaRect resta solo un fallback se il crop CNN non e' valido

Questo significa che il modulo interviene su TUTTI i candidati, non solo
su quelli che minAreaRect avrebbe gia' segnalato oltre soglia — altrimenti
un falso negativo del metodo geometrico (sottostima dell'angolo, comune su
maschere di segmentazione rumorose) impedirebbe alla stima piu' accurata di
intervenire proprio dove servirebbe di piu'.

Il modulo NON fa detection (non trova pali da solo): riceve sempre un bbox
gia' individuato dal filtro geometrico di DeepLabAnalyzer.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet34
from typing import Tuple, Optional

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PoleTiltAnalyzer:
    """Stima l'inclinazione di un palo da un crop RGB con un regressore CNN."""

    def __init__(self,
                 weights_path: str,
                 angolo_max: float = 22.0,
                 dimensione_input: int = 224,
                 margine_bbox_frazione: float = 0.25,
                 device: Optional[str] = None):
        """
        Args:
            weights_path: percorso al checkpoint .pt (formato di
                          train_pole_tilt.py: dict con chiave "model").
            angolo_max: range di normalizzazione usato in training (deve
                          coincidere con ANGOLO_MAX di train_pole_tilt.py,
                          default 22.0 — coerente con ANGOLO_MIN/MAX degli
                          script Unreal usati per generare il dataset).
            dimensione_input: lato del quadrato di input della rete (deve
                          coincidere con DIMENSIONE_OUTPUT di
                          crop_dataset.py, default 224).
            margine_bbox_frazione: il bbox prodotto da DeepLab e' stretto
                          attorno alla sola maschera di segmentazione del
                          palo, mentre i crop usati in training avevano un
                          margine di sicurezza generoso attorno al palo
                          (~0.55x l'altezza in orizzontale, vedi
                          crop_dataset.py). Per ridurre il mismatch tra
                          training e inferenza, il bbox viene espanso di
                          questa frazione (in pixel, sui 4 lati) prima del
                          crop. 0.25 e' un punto di partenza ragionevole,
                          da validare/aggiustare su frame video reali.
            device: "cuda", "cpu", o None per auto-detect.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.angolo_max = angolo_max
        self.dimensione_input = dimensione_input
        self.margine_bbox_frazione = margine_bbox_frazione

        self.model = self._load_model(weights_path)
        print(f"[PoleTiltAnalyzer] Modello caricato da {weights_path} | device={self.device}")

    def _load_model(self, weights_path: str) -> torch.nn.Module:
        """Ricostruisce l'ARCHITETTURA ESATTA usata in train_pole_tilt.py
        (costruisci_modello): ResNet34 + testa di regressione a 1 output.
        Deve restare in sincrono con quella funzione — se l'architettura di
        training cambia, aggiornare anche qui."""
        model = resnet34(weights=None)
        n_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()

        epoca = ckpt.get("epoca", "?")
        mae = ckpt.get("migliore_val_mae", "?")
        if isinstance(mae, float):
            mae = f"{mae:.2f}°"
        print(f"[PoleTiltAnalyzer] Epoca {epoca}, val_MAE {mae}")
        return model

    # ------------------------------------------------------------------
    # PREPROCESSING (deve rispecchiare esattamente train_pole_tilt.py)
    # ------------------------------------------------------------------

    def _espandi_bbox(self, bbox: Tuple[int, int, int, int],
                       frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Espande il bbox del margine configurato, clampato ai bordi del
        frame. Ritorna (x_min, y_min, x_max, y_max)."""
        x, y, w, h = bbox
        margine_x = int(w * self.margine_bbox_frazione)
        margine_y = int(h * self.margine_bbox_frazione)

        x_min = max(0, x - margine_x)
        y_min = max(0, y - margine_y)
        x_max = min(frame_w, x + w + margine_x)
        y_max = min(frame_h, y + h + margine_y)
        return x_min, y_min, x_max, y_max

    def _letterbox(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Ridimensiona mantenendo l'aspect ratio con padding nero, stessa
        tecnica usata in crop_dataset.py (letterbox) per il training —
        preprocessing DIVERSO tra training e inferenza e' una causa comune
        di degrado prestazionale silenzioso, quindi replicata identica."""
        h, w = crop_rgb.shape[:2]
        dim = self.dimensione_input
        scala = dim / max(w, h)
        nuova_w, nuova_h = max(1, int(round(w * scala))), max(1, int(round(h * scala)))
        ridim = cv2.resize(crop_rgb, (nuova_w, nuova_h), interpolation=cv2.INTER_LANCZOS4)

        risultato = np.zeros((dim, dim, 3), dtype=np.uint8)
        offset_x = (dim - nuova_w) // 2
        offset_y = (dim - nuova_h) // 2
        risultato[offset_y:offset_y + nuova_h, offset_x:offset_x + nuova_w] = ridim
        return risultato

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_rgb = self._letterbox(crop_rgb)
        img = crop_rgb.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor

    # ------------------------------------------------------------------
    # INFERENZA
    # ------------------------------------------------------------------

    def predict_angle(self, frame_bgr: np.ndarray,
                       bbox: Tuple[int, int, int, int]) -> Optional[float]:
        """
        Stima l'angolo di inclinazione (gradi, positivo/negativo) di un
        palo, dato il frame BGR completo e il bbox (x, y, w, h) prodotto da
        DeepLabAnalyzer._analyze_poles(). Ritorna None se il crop risulta
        vuoto/non valido (bbox degenere).
        """
        h_frame, w_frame = frame_bgr.shape[:2]
        x_min, y_min, x_max, y_max = self._espandi_bbox(bbox, w_frame, h_frame)

        if x_max <= x_min or y_max <= y_min:
            return None

        crop_bgr = frame_bgr[y_min:y_max, x_min:x_max]
        if crop_bgr.size == 0:
            return None

        tensor = self._preprocess(crop_bgr)
        with torch.no_grad():
            pred_normalizzata = self.model(tensor)

        angolo = float(pred_normalizzata.item()) * self.angolo_max
        return angolo