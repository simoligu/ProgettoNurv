import os
import shutil
import cv2
from tqdm import tqdm
from video_io import VideoIO
from alignment import FrameAligner
from background import BackgroundModel
from detection import ChangeDetector
from structural_detector import StructuralDetector
from PIL import Image
import csv
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import requests
import json

class AnomalyDetectionPipeline:
    def __init__(self, reference_video: str, query_video: str, out_dir: str = 'out',
                 sample_step: int = 10, use_classifier: bool = False, classifier: Optional[object] = None,
                 project_name: str = "ProgettoNurv", alert_endpoint: str = "http://localhost:8080/api/alerts",tratta_id:Optional[int] = None):
        self.reference_video = reference_video
        self.query_video = query_video
        self.out_dir = out_dir
        self.sample_step = sample_step
        self.use_classifier = use_classifier
        self.classifier = classifier
        self.project_name = project_name
        self.alert_endpoint = alert_endpoint
        self.structural_detector = StructuralDetector(expected_rail_width_px=150)
        self.tratta_id = tratta_id

        VideoIO.ensure_dir(out_dir)

        if os.path.exists(out_dir):
            print(f"🧹 Pulizia profonda della cartella {out_dir}...")
            for filename in os.listdir(out_dir):
                file_path = os.path.join(out_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"[WARN] Errore pulizia {file_path}: {e}")

    def send_alert(self, alert: dict):
        try:
            requests.post(self.alert_endpoint, json=alert, timeout=2)
        except Exception:
            pass

    def run(self):
        print('🎬 Sampling reference frames...')
        ref_frames_tuple = VideoIO.sample_frames(self.reference_video, max_frames=200, step=self.sample_step)
        ref_anchor = ref_frames_tuple[len(ref_frames_tuple) // 2][1]
        h, w = ref_anchor.shape[:2]

        print('🧱 Building median background...')
        only_frames = [f[1] for f in ref_frames_tuple]
        ref_median = BackgroundModel.build_median_background(only_frames, sample_n=50, resize_to=(w, h), use_luminance=True)

        gray_median = cv2.cvtColor(ref_median, cv2.COLOR_BGR2GRAY)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        ref_median_gray = clahe_obj.apply(gray_median)

        cap = cv2.VideoCapture(self.query_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(os.path.join(self.out_dir, 'annotated_video.mp4'), fourcc, fps, (w, h))

        csv_f = open(os.path.join(self.out_dir, 'detections.csv'), 'w', newline='')
        csvw = csv.writer(csv_f)
        csvw.writerow(['frame_idx', 'time_s', 'x', 'y', 'w', 'h', 'class', 'conf'])

        aligner = FrameAligner()
        idx = 0

        # --- ❗ MEMORIA PER NON MANDARE TROPPI ALERT ---
        last_alert_frame = {}

        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_f, desc='Processing')

        while True:
            ok, qf = cap.read()
            if not ok: break
            t_sec = idx / fps
            qf_gray = cv2.cvtColor(qf, cv2.COLOR_BGR2GRAY)

            try:
                H = aligner.compute_homography(ref_median_gray, qf_gray)
                if H is not None and np.abs(np.linalg.det(H)) > 0.1:
                    aligned = cv2.warpPerspective(qf, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
                else:
                    aligned = cv2.resize(qf, (w, h))
            except Exception:
                aligned = cv2.resize(qf, (w, h))

            raw_pixel_boxes, mask = ChangeDetector.detect_changes(ref_median, aligned, diff_thresh=90, min_area=8000)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidate_detections = []

            # 1. INTEGRAZIONE YOLO (AI) CON SEVERITÀ DINAMICA
            if self.use_classifier and self.classifier is not None:
                results = self.classifier(qf, verbose=False)
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        class_name = self.classifier.names[int(box.cls[0])].upper()
                        if class_name != 'TRAIN' and conf > 0.45:
                            # Logica Tesi: Persone/Animali = CRITICA, altro = MEDIO
                            sev_ia = "CRITICA" if class_name in ["PERSON", "DOG", "CAT", "BICYCLE"] else "MEDIO"

                            b = box.xyxy[0].cpu().numpy()
                            bx, by, bw_b, bh_b = int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])

                            # Invio Alert ogni 60 frame (2 secondi) per non intasare
                            if idx - last_alert_frame.get(class_name, -100) > 60:
                                alert = self._create_alert_dict(idx, t_sec, bx, by, bw_b, bh_b, bw_b*bh_b, class_name, conf, sev_ia, f"Rilevato {class_name} tramite YOLO", self.query_video)
                                self.send_alert(alert)
                                last_alert_frame[class_name] = idx

                            candidate_detections.append({'bbox': (bx, by, bw_b, bh_b), 'label': class_name, 'conf': conf})

            # 2. ANOMALIE GEOMETRICHE
            for cnt in contours:
                cx, cy, cww, chh = cv2.boundingRect(cnt)
                if cy < (h * 0.25) or cww < 50 or chh < 50: continue
                if cww > (w * 0.5) or chh > (h * 0.5): continue
                is_covered = False
                for det in candidate_detections:
                    dx, dy, dw, dh = det['bbox']
                    if not (cx + cww < dx or cx > dx + dw or cy + chh < dy or cy > dy + dh):
                        is_covered = True
                        break
                if not is_covered:
                    label_geo = "ANOMALIA_STRUTTURALE"
                    candidate_detections.append({'bbox': (int(cx), int(cy), int(cww), int(chh)), 'label': label_geo, 'conf': 0.85})
                    # Invio alert geometrico (ogni 90 frame)
                    if idx - last_alert_frame.get(label_geo, -100) > 90:
                        alert = self._create_alert_dict(idx, t_sec, cx, cy, cww, chh, cww*chh, label_geo, 0.85, "ALTA", "Rilevato cambiamento geometrico binari", self.query_video)
                        self.send_alert(alert)
                        last_alert_frame[label_geo] = idx

            annotated = aligned.copy()
            for det in candidate_detections:
                dx, dy, dww, dhh = det['bbox']
                cv2.rectangle(annotated, (dx, dy), (dx + dww, dy + dhh), (0, 0, 255), 2)
                csvw.writerow([idx, round(t_sec, 2), dx, dy, dww, dhh, det['label'], round(det['conf'], 2)])

            # 3. DIAGNOSTICA E ANALISI STRUTTURALE (PALI/VEGETAZIONE)
            if idx % 150 == 0:
                diag_dir = os.path.join(self.out_dir, f"diagnostics_frame_{idx}")
                os.makedirs(diag_dir, exist_ok=True)

                # --- ANALISI PALI ---
                pole_vis = aligned.copy()
                roi_pali = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(roi_pali, (0, 0), (int(w*0.30), h), 255, -1)
                cv2.rectangle(roi_pali, (int(w*0.70), 0), (w, h), 255, -1)
                edges = cv2.Canny(cv2.cvtColor(qf, cv2.COLOR_BGR2GRAY), 50, 150)
                masked_pali = cv2.bitwise_and(edges, roi_pali)
                lines = cv2.HoughLinesP(masked_pali, 1, np.pi/180, 50, minLineLength=100, maxLineGap=30)

                if lines is not None:
                    for l in lines:
                        x1, y1, x2, y2 = l[0]
                        dx, dy = x2 - x1, y2 - y1
                        angle = np.abs(np.degrees(np.arctan2(dy, dx)))
                        if 80 < angle < 100:
                            if abs(y2 - y1) > (h * 0.25):
                                theta = np.abs(np.arctan(dx / dy) * (180 / np.pi)) if dy != 0 else 0.0

                                # Logica Severità Tesi
                                if theta >= 8:
                                    sev, lab_p = "CRITICA", "PALO_INCLINATO"
                                elif 3 <= theta < 8:
                                    sev, lab_p = "ALTA", "PALO_INCLINATO"
                                else:
                                    sev, lab_p = "OK", "PALO_OK"

                                if sev != "OK" and idx - last_alert_frame.get("PALO", -100) > 120:
                                    alert = self._create_alert_dict(idx, t_sec, x1, y1, abs(x2-x1), abs(y2-y1), 0, lab_p, 1.0, sev, f"Inclinazione: {theta:.1f} gradi", self.query_video)
                                    self.send_alert(alert)
                                    last_alert_frame["PALO"] = idx

                # --- ANALISI VEGETAZIONE ---
                hsv = cv2.cvtColor(aligned, cv2.COLOR_BGR2HSV)
                mask_v = cv2.inRange(hsv, np.array([20, 30, 20]), np.array([95, 255, 255]))
                perc = (cv2.countNonZero(mask_v) / (w * h)) * 100
                if perc > 20 and idx - last_alert_frame.get("VEG", -100) > 300:
                    sev_v = "CRITICA" if perc > 40 else "MEDIO"
                    alert = self._create_alert_dict(idx, t_sec, 0, 0, w, h, 0, "VEGETAZIONE", 0.7, sev_v, f"Copertura binari: {perc:.1f}%", self.query_video)
                    self.send_alert(alert)
                    last_alert_frame["VEG"] = idx

            writer.write(annotated)
            idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()
        csv_f.close()
        print(f"✅ Analisi terminata. Risultati in '{self.out_dir}'")

    def _create_alert_dict(self, idx, t, x, y, ww, hh, area, label, conf, severity, details, query_video) -> Dict[str, Any]:
        return {
            "project": self.project_name, "frame_idx": int(idx), "time_s": round(t, 2),
            "bbox": {"x": int(x), "y": int(y), "w": int(ww), "h": int(hh)},
            "area": int(area), "label": label, "conf": float(conf), "severity": severity,
            "details": details, "source_video": os.path.basename(query_video),
            "trattaId": self.tratta_id
        }