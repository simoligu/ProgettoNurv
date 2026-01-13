import cv2
import os
import numpy as np
from typing import List, Tuple, Union

class VideoIO:
    @staticmethod
    def ensure_dir(path: str):
        """Assicura che la directory esista."""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"[INFO] Creata directory: {path}")

    @staticmethod
    def sample_frames(video_path: str, max_frames: int = 300, step: int = 1) -> List[Tuple[int, np.ndarray]]:
        """
        Campiona frame dal video.
        Restituisce una lista di tuple (frame_idx, frame).
        """
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video non trovato: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Impossibile aprire il video: {video_path}")

        frames: List[Tuple[int, np.ndarray]] = []
        i = 0
        while True:
            ok, f = cap.read()
            if not ok or f is None:
                break

            # Ritorna la tupla (indice, frame) per la compatibilità con pipeline.py
            if i % step == 0:
                frames.append((i, f))
                if len(frames) >= max_frames:
                    break
            i += 1

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Nessun frame letto da {video_path}")

        print(f"[INFO] Campionati {len(frames)} frame da {video_path}")
        return frames