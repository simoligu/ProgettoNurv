import cv2
import numpy as np
from typing import Optional, Tuple

class FrameAligner:
    def __init__(self, method='ORB', max_features=2000, ransac_seed=42):
        self.method = method.upper()
        self.max_features = max_features
        # Fissa il generatore di numeri casuali interno di OpenCV, usato da
        # cv2.findHomography(..., cv2.RANSAC, ...) per scegliere i sottoinsiemi
        # di punti su cui stimare la trasformazione. Senza questo, il risultato
        # di compute_homography() puo' variare leggermente da un'esecuzione
        # all'altra anche a parita' di input, perche' RANSAC e' un algoritmo
        # stocastico — con il seed fisso, stesso input -> stesso output, sempre.
        cv2.setRNGSeed(ransac_seed)

    def compute_homography(self, src_gray: np.ndarray, dst_gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Calcola l'omografia (H) tra due immagini in scala di grigi (src -> dst).
        """
        if self.method == 'SIFT':
            sift = cv2.SIFT_create()  # type: ignore
            kp1, des1 = sift.detectAndCompute(src_gray, None)
            kp2, des2 = sift.detectAndCompute(dst_gray, None)
        else:
            orb = cv2.ORB_create(self.max_features)  # type: ignore
            kp1, des1 = orb.detectAndCompute(src_gray, None)
            kp2, des2 = orb.detectAndCompute(dst_gray, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if self.method != 'SIFT' else cv2.BFMatcher()
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 10:
            return None

        good_matches = matches[:int(len(matches) * 0.6)]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Calcolo dell'omografia (H)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return H

    @staticmethod
    def warp_frame(frame: np.ndarray, homography_matrix: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Applica la matrice di omografia al frame per la warping."""
        if homography_matrix is None:
            return cv2.resize(frame, size)

        aligned_frame = cv2.warpPerspective(frame, homography_matrix, size)
        return aligned_frame