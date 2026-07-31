import cv2
import numpy as np
from typing import List, Tuple
from background import mask_to_boxes, BBox


class ChangeDetector:
    @staticmethod
    def detect_changes(
            ref_bgr: np.ndarray,
            query_bgr: np.ndarray,
            diff_thresh: int = 40,
            min_area: int = 500
    ) -> Tuple[List[BBox], np.ndarray]:
        """
        Rileva cambiamenti tra ref_bgr e query_bgr usando differenza assoluta sulla scala di grigi.
        (Versione originale: converte da BGR, nessuna normalizzazione di illuminazione.)
        """
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        q_gray = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(ref_gray, q_gray)
        boxes, cleaned_mask = mask_to_boxes(diff, diff_thresh=diff_thresh, min_area=min_area)
        return boxes, cleaned_mask

    @staticmethod
    def detect_changes_gray(
            ref_gray: np.ndarray,
            query_gray: np.ndarray,
            diff_thresh: int = 40,
            min_area: int = 500
    ) -> Tuple[List[BBox], np.ndarray]:
        """
        Come detect_changes, ma accetta direttamente immagini in scala di grigi
        gia' pronte (es. gia' equalizzate con CLAHE per normalizzare l'illuminazione).

        Usare questa variante quando reference e query sono girati in condizioni di
        luce diverse: passare ref_gray e query_gray gia' processati con CLAHE evita
        che la differenza assoluta rilevi falsi cambiamenti dovuti solo alla luce.
        """
        diff = cv2.absdiff(ref_gray, query_gray)
        boxes, cleaned_mask = mask_to_boxes(diff, diff_thresh=diff_thresh, min_area=min_area)
        return boxes, cleaned_mask