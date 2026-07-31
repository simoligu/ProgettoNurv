import os
import shutil
import cv2
from tqdm import tqdm
from video_io import VideoIO
from alignment import FrameAligner
from background import BackgroundModel
from detection import ChangeDetector
from PIL import Image
import csv
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import requests
import json


class AnomalyDetectionPipeline:
    def __init__(self, reference_video: str, query_video: str, out_dir: str = 'out',
                 sample_step: int = 10, use_classifier: bool = False, classifier: Optional[object] = None,
                 project_name: str = "ProgettoNurv", alert_endpoint: str = "http://localhost:8080/api/alerts",
                 tratta_id: Optional[int] = None,
                 # --- NUOVO: parametri DeepLab ---
                 deeplab_weights: Optional[str] = None,
                 deeplab_imgsz: int = 896,
                 seg_step: int = 30):
        """
        Args (nuovi rispetto alla versione precedente):
            deeplab_weights: percorso al best.pt del modello DeepLabV3+.
                             Se None, l'analisi strutturale non viene eseguita.
            deeplab_imgsz: risoluzione di inferenza DeepLab (deve corrispondere al training).
            seg_step: ogni quanti frame eseguire l'analisi strutturale DeepLab.
                      Default 30 (~1 secondo a 30fps). De Paolis usava 150.
        """
        self.reference_video = reference_video
        self.query_video = query_video
        self.out_dir = out_dir
        self.sample_step = sample_step
        self.use_classifier = use_classifier
        self.classifier = classifier
        self.project_name = project_name
        self.alert_endpoint = alert_endpoint
        self.tratta_id = tratta_id
        self.seg_step = seg_step

        # --- DeepLab analyzer (sostituisce StructuralDetector) ---
        self.analyzer = None
        if deeplab_weights and os.path.exists(deeplab_weights):
            from deeplab_analyzer import DeepLabAnalyzer
            self.analyzer = DeepLabAnalyzer(
                weights_path=deeplab_weights,
                imgsz=deeplab_imgsz,
            )
        else:
            if deeplab_weights:
                print(f"[WARN] Pesi DeepLab non trovati: {deeplab_weights}")
            print("[INFO] Analisi strutturale DeepLab disattivata.")

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

        # --- AUTO-CALIBRAZIONE SCARTAMENTO DAL REFERENCE ---
        if self.analyzer is not None:
            print('📐 Auto-calibrazione scartamento dal reference video...')
            ref_class_map = self.analyzer.segment(ref_median)
            baseline_gauge = self.analyzer.measure_gauge(ref_class_map)
            if baseline_gauge is not None:
                self.analyzer.expected_gauge_px = baseline_gauge
                print(f'   Scartamento reference: {baseline_gauge:.0f}px')
            else:
                print('   [WARN] Rotaie non rilevate nel reference — scartamento non calibrato.')

        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # --- FIX OMOGRAFIA: usa un frame NITIDO (ref_anchor) per il matching ORB,
        #     non il background mediano (che e' sfocato e povero di keypoint).
        #     Il background mediano resta usato solo per il confronto/diff finale. ---
        ref_anchor_gray = cv2.cvtColor(ref_anchor, cv2.COLOR_BGR2GRAY)
        ref_anchor_gray = clahe_obj.apply(ref_anchor_gray)

        # versione equalizzata del background mediano, usata SOLO per il diff
        # (normalizza l'illuminazione cosi' il confronto non risente di differenze
        #  sistematiche di luce tra reference e query)
        ref_median_gray_eq = clahe_obj.apply(cv2.cvtColor(ref_median, cv2.COLOR_BGR2GRAY))

        cap = cv2.VideoCapture(self.query_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(os.path.join(self.out_dir, 'annotated_video.mp4'), fourcc, fps, (w, h))

        csv_f = open(os.path.join(self.out_dir, 'detections.csv'), 'w', newline='')
        csvw = csv.writer(csv_f)
        csvw.writerow(['frame_idx', 'time_s', 'x', 'y', 'w', 'h', 'class', 'conf'])

        aligner = FrameAligner()
        idx = 0

        # --- MEMORIA PER NON MANDARE TROPPI ALERT ---
        last_alert_frame = {}

        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_f, desc='Processing')

        while True:
            ok, qf = cap.read()
            if not ok:
                break
            t_sec = idx / fps
            qf_gray = cv2.cvtColor(qf, cv2.COLOR_BGR2GRAY)

            try:
                # matching ORB sul frame NITIDO di reference (ref_anchor), non sul
                # background mediano sfocato -> keypoint molto piu' affidabili
                qf_gray_eq = clahe_obj.apply(qf_gray)
                H = aligner.compute_homography(ref_anchor_gray, qf_gray_eq)
                if H is not None and np.abs(np.linalg.det(H)) > 0.1:
                    aligned = cv2.warpPerspective(qf, H, (w, h), flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                else:
                    aligned = cv2.resize(qf, (w, h))
            except Exception:
                aligned = cv2.resize(qf, (w, h))

            # normalizza l'illuminazione del frame allineato prima del confronto,
            # cosi' una differenza di luce sistematica reference/query non genera
            # falsi positivi diffusi su tutto il frame
            aligned_gray_eq = clahe_obj.apply(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
            raw_pixel_boxes, mask = ChangeDetector.detect_changes_gray(
                ref_median_gray_eq, aligned_gray_eq, diff_thresh=90, min_area=8000
            )
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidate_detections = []

            # 1. INTEGRAZIONE YOLO (AI) CON SEVERITA' DINAMICA
            if self.use_classifier and self.classifier is not None:
                results = self.classifier(qf, verbose=False)
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        class_name = self.classifier.names[int(box.cls[0])].upper()
                        if class_name != 'TRAIN' and conf > 0.45:
                            sev_ia = "CRITICA" if class_name in ["PERSON", "DOG", "CAT", "BICYCLE"] else "MEDIO"

                            b = box.xyxy[0].cpu().numpy()
                            bx, by, bw_b, bh_b = int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])

                            if idx - last_alert_frame.get(class_name, -100) > 60:
                                alert = self._create_alert_dict(idx, t_sec, bx, by, bw_b, bh_b,
                                                                bw_b * bh_b, class_name, conf, sev_ia,
                                                                f"Rilevato {class_name} tramite YOLO",
                                                                self.query_video)
                                self.send_alert(alert)
                                last_alert_frame[class_name] = idx

                            candidate_detections.append(
                                {'bbox': (bx, by, bw_b, bh_b), 'label': class_name, 'conf': conf})

            # 2. ANOMALIE GEOMETRICHE (background subtraction)
            for cnt in contours:
                cx, cy, cww, chh = cv2.boundingRect(cnt)
                if cy < (h * 0.25) or cww < 50 or chh < 50:
                    continue
                if cww > (w * 0.5) or chh > (h * 0.5):
                    continue
                is_covered = False
                for det in candidate_detections:
                    dx, dy, dw, dh = det['bbox']
                    if not (cx + cww < dx or cx > dx + dw or cy + chh < dy or cy > dy + dh):
                        is_covered = True
                        break
                if not is_covered:
                    label_geo = "ANOMALIA_STRUTTURALE"
                    candidate_detections.append(
                        {'bbox': (int(cx), int(cy), int(cww), int(chh)), 'label': label_geo, 'conf': 0.85})
                    if idx - last_alert_frame.get(label_geo, -100) > 90:
                        alert = self._create_alert_dict(idx, t_sec, cx, cy, cww, chh, cww * chh,
                                                        label_geo, 0.85, "ALTA",
                                                        "Rilevato cambiamento geometrico binari",
                                                        self.query_video)
                        self.send_alert(alert)
                        last_alert_frame[label_geo] = idx

            annotated = aligned.copy()
            for det in candidate_detections:
                dx, dy, dww, dhh = det['bbox']
                cv2.rectangle(annotated, (dx, dy), (dx + dww, dy + dhh), (0, 0, 255), 2)
                csvw.writerow([idx, round(t_sec, 2), dx, dy, dww, dhh, det['label'], round(det['conf'], 2)])

            # ============================================================
            # 3. ANALISI INFRASTRUTTURA CON DeepLabV3+
            #    Sostituisce la visione classica di De Paolis:
            #    - Hough per rotaie     -> segmentazione semantica + misura scartamento
            #    - Gradienti per pali   -> segmentazione + angolo componenti connesse
            #    - HSV per vegetazione  -> segmentazione + distance transform prossimita'
            # ============================================================
            if self.analyzer is not None and idx % self.seg_step == 0:
                dl_anomalies = self.analyzer.analyze(aligned)

                for a in dl_anomalies:
                    label = a["label"]
                    bbox = a.get("bbox", (0, 0, w, h))

                    # throttling: non mandare lo stesso tipo di alert troppo spesso
                    if idx - last_alert_frame.get(label, -100) > self.seg_step * 2:
                        alert = self._create_alert_dict(
                            idx, t_sec,
                            bbox[0], bbox[1], bbox[2], bbox[3],
                            bbox[2] * bbox[3],
                            label, a["conf"], a["severity"],
                            a["details"], self.query_video
                        )
                        self.send_alert(alert)
                        last_alert_frame[label] = idx

                    # aggiungi anche alla lista detection per annotare il frame
                    candidate_detections.append({
                        'bbox': bbox,
                        'label': label,
                        'conf': a["conf"]
                    })

                    # disegna il rettangolo dell'anomalia sul frame annotato
                    bx_a, by_a, bw_a, bh_a = bbox
                    # colori diversi per tipo di anomalia
                    if "SCARTAMENTO" in label:
                        color = (0, 165, 255)   # arancione
                    elif "PALO" in label:
                        color = (255, 0, 0)     # blu
                    elif "VEGETAZIONE" in label:
                        color = (0, 200, 0)     # verde
                    else:
                        color = (0, 0, 255)     # rosso
                    cv2.rectangle(annotated, (bx_a, by_a), (bx_a + bw_a, by_a + bh_a), color, 2)
                    cv2.putText(annotated, f"{label} ({a['severity']})",
                                (bx_a, max(by_a - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    csvw.writerow([idx, round(t_sec, 2), bbox[0], bbox[1], bbox[2], bbox[3],
                                   label, round(a['conf'], 2)])

            writer.write(annotated)
            idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()
        csv_f.close()
        print(f"✅ Analisi terminata. Risultati in '{self.out_dir}'")

    def _create_alert_dict(self, idx, t, x, y, ww, hh, area, label, conf, severity, details,
                           query_video) -> Dict[str, Any]:
        return {
            "project": self.project_name, "frame_idx": int(idx), "time_s": round(t, 2),
            "bbox": {"x": int(x), "y": int(y), "w": int(ww), "h": int(hh)},
            "area": int(area), "label": label, "conf": float(conf), "severity": severity,
            "details": details, "source_video": os.path.basename(query_video),
            "trattaId": self.tratta_id
        }