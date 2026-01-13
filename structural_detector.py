import cv2
import numpy as np
from typing import List, Dict, Tuple, Any

class StructuralDetector:
    def __init__(self, expected_rail_width_px: int = 150):
        """
        Inizializza il rilevatore di anomalie strutturali.

        Args:
            expected_rail_width_px: Larghezza attesa dei binari in pixel
            (IMPORTANTE: Calibra questo valore per il tuo video).
        """
        self.expected_rail_width_px = expected_rail_width_px
        self.max_tilt_deg = 8.0  # Soglia massima di inclinazione accettabile per i pali (8 gradi)

        # Soglie per il filtro HSV della vegetazione
        self.lower_green = np.array([25, 30, 30])
        self.upper_green = np.array([95, 255, 255])

        self.proximity_threshold = 50  # Distanza massima in pixel dal binario
        self.min_pixels_for_alert = 0.02  # Densità minima (2%) di pixel verdi

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Esegue l'analisi strutturale completa (dilatazione, inclinazione, vegetazione)
        su un singolo frame allineato.
        """
        anomalies: List[Dict[str, Any]] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Rilevamento Bordi e Linee (Pre-elaborazione)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        rail_lines = []
        pole_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad) % 180

                # Filtro Linee
                if 80 < angle_deg < 100:
                    pole_lines.append(line[0])
                elif (0 <= angle_deg < 40) or (140 < angle_deg <= 180):
                    rail_lines.append(line[0])

        # 2. Analisi Dilatazione Ferrovia
        anomalies.extend(self._analyze_dilatation(rail_lines, frame.shape))

        # 3. Analisi Palo Inclinato
        anomalies.extend(self._analyze_pole_tilt(pole_lines))

        # 4. Analisi Vegetazione
        anomalies.extend(self._analyze_vegetation_proximity(frame, rail_lines))

        return anomalies

    def _analyze_dilatation(self, rail_lines: List[np.ndarray], frame_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Misura la distanza tra i binari (Dilatazione)."""
        anomalies = []
        if len(rail_lines) < 2:
            return anomalies

        lines_sorted = sorted(rail_lines, key=lambda l: (l[0] + l[2]) / 2)
        left_rail = lines_sorted[0]
        right_rail = lines_sorted[-1]
        mid_y = frame_shape[0] // 2

        try:
            def get_x_at_y(line, y_target):
                x1, y1, x2, y2 = line
                if abs(y2 - y1) < 1:
                    return (x1 + x2) / 2
                return x1 + (x2 - x1) * (y_target - y1) / (y2 - y1)

            x_left = get_x_at_y(left_rail, mid_y)
            x_right = get_x_at_y(right_rail, mid_y)
            current_width = abs(x_right - x_left)

            tolerance = 0.1  # 10% di tolleranza
            if abs(current_width - self.expected_rail_width_px) > self.expected_rail_width_px * tolerance:
                anomalies.append({
                    "label": "Dilatazione Ferrovia",
                    "severity": "CRITICA",
                    "conf": min(abs(current_width - self.expected_rail_width_px) / (self.expected_rail_width_px * 2), 1.0),
                    "details": f"Larghezza misurata: {current_width:.2f}px (Attesa: {self.expected_rail_width_px}px)"
                })
        except Exception:
            pass

        return anomalies

    def _analyze_pole_tilt(self, pole_lines: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Misura l'inclinazione dei pali."""
        anomalies = []
        for line in pole_lines:
            x1, y1, x2, y2 = line
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)
            tilt = abs(90 - (angle_deg % 180))

            if tilt > self.max_tilt_deg:
                anomalies.append({
                    "label": "Palo Inclinato",
                    "severity": "MEDIO",
                    "conf": min(tilt / (self.max_tilt_deg * 2), 1.0),
                    "details": f"Inclinazione: {tilt:.2f} gradi (Max: {self.max_tilt_deg} gradi)"
                })
        return anomalies

    def _analyze_vegetation_proximity(self, frame: np.ndarray, rail_lines: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Rileva la vegetazione eccessivamente vicina usando l'analisi del colore HSV."""
        anomalies = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)

        if len(rail_lines) < 2:
            return []

        lines_sorted = sorted(rail_lines, key=lambda l: (l[0] + l[2]) / 2)
        left_rail_x = min(lines_sorted[0][0], lines_sorted[0][2])
        right_rail_x = max(lines_sorted[-1][0], lines_sorted[-1][2])

        def check_area(x_start, x_end, side_label):
            x_start_clamped = max(0, int(x_start))
            x_end_clamped = min(frame.shape[1], int(x_end))

            critical_area = mask_green[:, x_start_clamped:x_end_clamped]
            if critical_area.size > 0:
                density = np.sum(critical_area > 0) / critical_area.size
                if density > self.min_pixels_for_alert:
                    anomalies.append({
                        "label": f"Vegetazione Vicina ({side_label})",
                        "severity": "BASSO",
                        "conf": float(density),
                        "details": f"Densità vegetazione: {density * 100:.2f}% nell'area critica."
                    })

        check_area(left_rail_x - self.proximity_threshold, left_rail_x, "Sinistra")
        check_area(right_rail_x, right_rail_x + self.proximity_threshold, "Destra")

        return anomalies