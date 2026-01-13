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
        """
        # Converti in scala di grigi
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        q_gray = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)

        # Calcola la differenza assoluta
        diff = cv2.absdiff(ref_gray, q_gray)

        # Usa mask_to_boxes per sogliare, pulire e trovare i contorni
        boxes, cleaned_mask = mask_to_boxes(
            diff, diff_thresh=diff_thresh, min_area=min_area
        )

        return boxes, cleaned_mask