import os
import shutil
import cv2
from tqdm import tqdm
from video_io import VideoIO
from alignment import FrameAligner
from background import BackgroundModel
from detection import ChangeDetector
import csv
import base64
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import requests
import json

# OCR opzionale per leggere le coordinate sovraimpresse dal collega sul video.
# Se pytesseract non e' installato, l'estrazione coordinate viene semplicemente
# disattivata (nessun crash): il campo "location" negli alert restera' None finche'
# non vi mettete d'accordo sul formato e installi la dipendenza.
try:
    import pytesseract
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False

# NOTA: l'invio Telegram NON viene fatto da qui. Il routing multi-canale (un
# canale Telegram diverso per ogni tratta) e' gia' gestito lato dashboard Java,
# che riceve l'alert su /api/alerts e decide dove inoltrarlo in base a trattaId.
# La pipeline si limita ad allegare l'immagine del frame (base64) all'alert,
# cosi' il backend puo' usarla per la notifica su qualsiasi canale scelga.


class AnomalyDetectionPipeline:
    def __init__(self, reference_video: str, query_video: str, out_dir: str = 'out',
                 sample_step: int = 10, use_classifier: bool = False, classifier: Optional[object] = None,
                 project_name: str = "ProgettoNurv", alert_endpoint: str = "http://localhost:8080/api/alerts",
                 tratta_id: Optional[int] = None,
                 # --- DeepLab (modulo indipendente, non dipende dal reference per l'analisi) ---
                 deeplab_weights: Optional[str] = None,
                 deeplab_imgsz: int = 896,
                 seg_step: int = 30,
                 default_gauge_px: Optional[int] = None,
                 gauge_tolerance: float = 0.15,
                 # --- PoleTilt (modulo indipendente, raffina l'angolo dei pali gia'
                 # rilevati da DeepLab — vedi pole_tilt_analyzer.py) ---
                 pole_tilt_weights: Optional[str] = None,
                 pole_tilt_angolo_max: float = 22.0,
                 # --- Mappa di sincronizzazione (vedi sync_videos_dtw.py) ---
                 sync_map_csv: Optional[str] = None,
                 # --- Coordinate GPS sovraimpresse (opzionale, da definire col collega) ---
                 coord_region: Optional[Tuple[int, int, int, int]] = None):
        """
        Args (nuovi rispetto alla versione precedente):
            deeplab_weights: percorso al best.pt del modello DeepLabV3+.
                             Se None, l'analisi strutturale DeepLab non viene eseguita.
            deeplab_imgsz: risoluzione di inferenza DeepLab (deve corrispondere al training).
            seg_step: ogni quanti frame eseguire l'analisi strutturale DeepLab.
            default_gauge_px: scartamento di riferimento in pixel, usato se l'auto-
                             calibrazione dal reference fallisce (rotaie non rilevate).
                             Se None, la calibrazione automatica e' l'unica fonte.
            gauge_tolerance: tolleranza percentuale sullo scartamento (default 0.15 = 15%).
                             Esposta qui per test di sanita': abbassandola temporaneamente
                             (es. 0.05) si verifica che il meccanismo di alert sia ancora
                             sensibile e non "spento" dal filtro di robustezza.
            pole_tilt_weights: percorso al checkpoint .pt del regressore CNN
                             per l'inclinazione dei pali (vedi train_pole_tilt.py /
                             NURV_pali_inclinati_diario.md). Se None, gli alert
                             PALO_INCLINATO restano basati solo sulla stima
                             geometrica di DeepLab (minAreaRect), invariata.
                             Se fornito, l'angolo del CNN SOSTITUISCE quello
                             geometrico per severity/details di ogni alert
                             PALO_INCLINATO (il rilevamento — SE un dato bbox
                             genera un'anomalia — resta comunque governato da
                             DeepLab: il CNN raffina, non aggiunge rilevamenti).
            pole_tilt_angolo_max: deve coincidere con ANGOLO_MAX usato in
                             train_pole_tilt.py (default 22.0).
            coord_region: (x, y, w, h) del rettangolo dove il collega sovraimprime le
                             coordinate GPS sul frame. Se None, l'estrazione coordinate
                             e' disattivata. Formato atteso nel testo: "LAT:.. LON:..".
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
        self.coord_region = coord_region

        # --- DeepLab analyzer: sostituisce la visione classica (Hough/HSV) di De Paolis.
        #     E' un modulo INDIPENDENTE dal reference video: analizza ogni frame del
        #     query per quello che mostra (scartamento, pali, vegetazione), senza bisogno
        #     di un confronto prima/dopo. L'unico uso del reference e' la calibrazione
        #     iniziale dello scartamento (opzionale, con fallback su default_gauge_px). ---
        self.analyzer = None
        if deeplab_weights and os.path.exists(deeplab_weights):
            from deeplab_analyzer import DeepLabAnalyzer
            self.analyzer = DeepLabAnalyzer(
                weights_path=deeplab_weights,
                imgsz=deeplab_imgsz,
                gauge_tolerance=gauge_tolerance,
            )
            if default_gauge_px is not None:
                self.analyzer.expected_gauge_px = default_gauge_px
        else:
            if deeplab_weights:
                print(f"[WARN] Pesi DeepLab non trovati: {deeplab_weights}")
            print("[INFO] Analisi strutturale DeepLab disattivata.")

        # --- PoleTilt analyzer: modulo indipendente che raffina l'angolo dei
        #     pali GIA' rilevati da DeepLab. Non fa detection propria — riceve
        #     sempre un bbox gia' individuato da _analyze_poles(). ---
        self.pole_tilt_analyzer = None
        if pole_tilt_weights and os.path.exists(pole_tilt_weights):
            from pole_tilt_analyzer import PoleTiltAnalyzer
            self.pole_tilt_analyzer = PoleTiltAnalyzer(
                weights_path=pole_tilt_weights,
                angolo_max=pole_tilt_angolo_max,
            )
        else:
            if pole_tilt_weights:
                print(f"[WARN] Pesi PoleTilt non trovati: {pole_tilt_weights}")
            print("[INFO] Raffinamento CNN dell'inclinazione pali disattivato "
                  "(alert PALO_INCLINATO basati solo sulla stima geometrica DeepLab).")

        # --- Mappa di sincronizzazione: sostituisce il background mediato fisso
        #     con il frame reference CORRETTO per ogni frame query, e permette di
        #     saltare il modulo di background-subtraction sui frame senza vera
        #     corrispondenza (invece di confrontarli con un frame reference sbagliato,
        #     la causa nota dei falsi positivi su video che coprono l'intero tragitto). ---
        self.sync_map = None
        self._ref_cap_lookup = None
        self._ultimo_indice_ref_letto = None
        self._ultimo_frame_ref_letto = None
        if sync_map_csv and os.path.exists(sync_map_csv):
            from sync_map import MappaSincronizzazione
            self.sync_map = MappaSincronizzazione(sync_map_csv)
            self._ref_cap_lookup = cv2.VideoCapture(reference_video)
        else:
            if sync_map_csv:
                print(f"[WARN] Mappa di sincronizzazione non trovata: {sync_map_csv}")
            print("[INFO] Mappa di sincronizzazione non fornita — il modulo di "
                  "background-subtraction usa il background mediato fisso (comportamento invariato).")

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

    @staticmethod
    def _encode_frame_jpeg(frame: np.ndarray, max_width: int = 960, quality: int = 80) -> Optional[str]:
        """
        Comprime un frame in JPEG e lo codifica in base64, pronto per essere
        allegato a un alert JSON (campo 'frame_b64'). Ridimensiona per contenere
        il payload (non serve piena risoluzione per il contesto visivo di un'allerta).
        Il backend puo' usare questo campo per salvare l'immagine a DB e/o inoltrarla
        sul canale Telegram associato alla tratta.
        """
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            frame = cv2.resize(frame, (max_width, int(h * scale)))
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode('utf-8')

    def dispatch_alert(self, alert: dict, frame: Optional[np.ndarray] = None):
        """
        Punto unico di invio per ogni alert generato dalla pipeline (YOLO,
        background subtraction, DeepLab). Se viene passato un frame, lo allega
        come JPEG base64 (campo 'frame_b64') prima di inviare all'endpoint Java.
        Il routing verso Telegram (canale specifico per trattaId) e' interamente
        responsabilita' del backend, non della pipeline.
        """
        if frame is not None:
            b64 = self._encode_frame_jpeg(frame)
            if b64:
                alert["frame_b64"] = b64

        self.send_alert(alert)

    def _leggi_frame_reference_per_indice(self, indice: int) -> Optional[np.ndarray]:
        """
        Legge il frame del video REFERENCE a un indice arbitrario (random access
        via seek), usato dal modulo di sincronizzazione. Cache minimale: se
        l'indice richiesto coincide con l'ultimo letto (capita spesso, dato che
        indici query vicini spesso si arrotondano allo stesso indice reference),
        evita una seek/read ridondante.

        NOTA SULLE PRESTAZIONI: la seek (cv2.CAP_PROP_POS_FRAMES) puo' essere
        lenta su alcuni codec/formati, specialmente per salti ampi o all'indietro.
        Se questo risulta un collo di bottiglia su video lunghi, una possibile
        ottimizzazione futura e' leggere il reference in sequenza invece che a
        salti (dato che la mappa e' monotona non decrescente sull'indice
        reference, si potrebbe avanzare con .read() invece di seek() quando il
        prossimo indice richiesto e' vicino al precedente).
        """
        if indice == self._ultimo_indice_ref_letto:
            return self._ultimo_frame_ref_letto

        self._ref_cap_lookup.set(cv2.CAP_PROP_POS_FRAMES, indice)
        ok, frame = self._ref_cap_lookup.read()
        if not ok:
            return None

        self._ultimo_indice_ref_letto = indice
        self._ultimo_frame_ref_letto = frame
        return frame

    @staticmethod
    def _frame_with_box(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                        color: Tuple[int, int, int], label: Optional[str] = None) -> np.ndarray:
        """
        Ritorna una COPIA del frame con il rettangolo (e opzionalmente l'etichetta)
        disegnati sopra. Usato per l'immagine allegata agli alert (dashboard,
        Telegram): senza questo, l'immagine mostrata sarebbe il frame grezzo senza
        alcuna indicazione visiva di dove si trovi l'anomalia.
        """
        vis = frame.copy()
        bx, by, bw_v, bh_v = bbox
        cv2.rectangle(vis, (bx, by), (bx + bw_v, by + bh_v), color, 3)
        if label:
            cv2.putText(vis, label, (bx, max(by - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return vis

    # ------------------------------------------------------------------
    # ESTRAZIONE COORDINATE (aggancio per la scheda di localizzazione del collega)
    # ------------------------------------------------------------------

    def _extract_coordinates(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Legge le coordinate GPS sovraimpresse su una zona fissa del frame (OCR).
        Ritorna {"lat": .., "lon": ..} oppure None se non disponibile/non leggibile.

        Attivo solo se:
          - coord_region e' stato impostato (rettangolo dove il testo compare)
          - pytesseract e' installato (pip install pytesseract; richiede anche il
            binario Tesseract OCR installato sul sistema)

        Formato testo atteso (da concordare col collega): "LAT:41.8234 LON:12.4567"
        Se il formato cambia, basta adattare il parsing qui sotto.
        """
        if self.coord_region is None or not _HAS_OCR:
            return None

        x, y, w_r, h_r = self.coord_region
        crop = frame[y:y + h_r, x:x + w_r]
        if crop.size == 0:
            return None

        # OCR va meglio su testo ad alto contrasto: binarizza il crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        try:
            text = pytesseract.image_to_string(binary, config='--psm 7')
        except Exception:
            return None

        return self._parse_coordinates(text)

    @staticmethod
    def _parse_coordinates(text: str) -> Optional[Dict[str, float]]:
        """Estrae lat/lon da una stringa tipo 'LAT:41.8234 LON:12.4567'."""
        import re
        m = re.search(r'LAT[:\s]*(-?\d+\.?\d*)\s*LON[:\s]*(-?\d+\.?\d*)', text.upper())
        if not m:
            return None
        try:
            return {"lat": float(m.group(1)), "lon": float(m.group(2))}
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------

    def run(self):
        print('🎬 Sampling reference frames...')
        ref_frames_tuple = VideoIO.sample_frames(self.reference_video, max_frames=200, step=self.sample_step)
        ref_anchor = ref_frames_tuple[len(ref_frames_tuple) // 2][1]
        h, w = ref_anchor.shape[:2]

        print('🧱 Building median background...')
        only_frames = [f[1] for f in ref_frames_tuple]
        ref_median = BackgroundModel.build_median_background(
            only_frames, sample_n=50, resize_to=(w, h), use_luminance=True
        )

        # --- AUTO-CALIBRAZIONE SCARTAMENTO (modulo DeepLab, indipendente) ---
        # Usa ref_anchor (frame nitido), non il background mediano sfocato: la
        # segmentazione, addestrata su immagini reali nitide, ne risente altrimenti.
        # Se fallisce, resta il default_gauge_px passato al costruttore (se fornito),
        # altrimenti il default hardcoded dell'analyzer.
        if self.analyzer is not None:
            print('📐 Auto-calibrazione scartamento dal reference video (frame nitido)...')
            ref_class_map = self.analyzer.segment(ref_anchor)
            baseline_gauge = self.analyzer.measure_gauge(ref_class_map)
            if baseline_gauge is not None:
                self.analyzer.expected_gauge_px = baseline_gauge
                print(f'   Scartamento reference: {baseline_gauge:.0f}px')
            else:
                print(f'   [WARN] Rotaie non rilevate nel reference — scartamento non calibrato, '
                      f'uso il valore corrente ({self.analyzer.expected_gauge_px}px).')

        # --- Setup per il modulo di De Paolis (YOLO ostacoli + background subtraction) ---
        # NOTA METODOLOGICA (limite noto, da documentare in tesi): questo modulo
        # confronta ogni frame query con un UNICO background mediato sul reference.
        # Funziona bene con camera fissa o tragitti brevi. Su un reference che copre
        # un intero viaggio (come nel nostro caso), il background mediato perde
        # dettaglio e l'omografia fatica a trovare corrispondenze -> percentuale di
        # match affidabili bassa. Il modulo DeepLab non soffre di questo problema
        # perche' analizza ogni frame query in modo indipendente.
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ref_anchor_gray = clahe_obj.apply(cv2.cvtColor(ref_anchor, cv2.COLOR_BGR2GRAY))
        ref_median_gray_eq = clahe_obj.apply(cv2.cvtColor(ref_median, cv2.COLOR_BGR2GRAY))

        cap = cv2.VideoCapture(self.query_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(os.path.join(self.out_dir, 'annotated_video.mp4'), fourcc, fps, (w, h))

        csv_f = open(os.path.join(self.out_dir, 'detections.csv'), 'w', newline='')
        csvw = csv.writer(csv_f)
        csvw.writerow(['frame_idx', 'time_s', 'x', 'y', 'w', 'h', 'class', 'conf', 'lat', 'lon'])

        aligner = FrameAligner()
        idx = 0
        last_alert_frame = {}

        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_f, desc='Processing')

        while True:
            ok, qf = cap.read()
            if not ok:
                break
            t_sec = idx / fps
            qf_gray = cv2.cvtColor(qf, cv2.COLOR_BGR2GRAY)

            # coordinate correnti (None se non configurate o non lette)
            coords = self._extract_coordinates(qf)

            # --- allineamento per il modulo De Paolis ---
            # Se e' disponibile una mappa di sincronizzazione, si usa il frame
            # REFERENCE CORRETTO per questo specifico frame query (letto a indice
            # arbitrario dal video reference) invece dell'unico background mediato
            # fisso — questo risolve il limite gia' documentato del modulo (falsi
            # positivi diffusi su video che coprono l'intero tragitto). Per i frame
            # query SENZA corrispondenza valida nella mappa, il modulo viene
            # SALTATO per quel frame (nessuna detection ANOMALIA_STRUTTURALE), per
            # non confrontarli con un frame reference sbagliato — YOLO e DeepLab
            # restano comunque attivi su ogni frame, non dipendono dal reference.
            saltato_per_mancanza_corrispondenza = False

            if self.sync_map is not None:
                indice_ref = self.sync_map.frame_reference_per(idx)
                frame_reference_locale = (
                    self._leggi_frame_reference_per_indice(indice_ref)
                    if indice_ref is not None else None
                )
            else:
                frame_reference_locale = None  # ramo non usato quando sync_map assente

            if self.sync_map is not None and frame_reference_locale is None:
                # nessuna corrispondenza valida per questo frame (o lettura fallita):
                # salta il modulo di background-subtraction per questo frame
                saltato_per_mancanza_corrispondenza = True
                aligned = cv2.resize(qf, (w, h))  # placeholder, serve solo per il video annotato
                raw_pixel_boxes, mask = [], np.zeros((h, w), dtype=np.uint8)
                contours = []
            else:
                if self.sync_map is not None:
                    # confronto contro il frame reference CORRETTO per questo istante
                    frame_reference_resized = cv2.resize(frame_reference_locale, (w, h))
                    riferimento_gray_eq = clahe_obj.apply(
                        cv2.cvtColor(frame_reference_resized, cv2.COLOR_BGR2GRAY)
                    )
                else:
                    # comportamento ORIGINALE invariato: anchor fisso + background mediato
                    riferimento_gray_eq = ref_anchor_gray

                try:
                    qf_gray_eq = clahe_obj.apply(qf_gray)
                    H = aligner.compute_homography(riferimento_gray_eq, qf_gray_eq)
                    if H is not None and np.abs(np.linalg.det(H)) > 0.1:
                        aligned = cv2.warpPerspective(qf, H, (w, h), flags=cv2.INTER_LINEAR,
                                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                    else:
                        aligned = cv2.resize(qf, (w, h))
                except Exception:
                    aligned = cv2.resize(qf, (w, h))

                aligned_gray_eq = clahe_obj.apply(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))

                # il TARGET del confronto pixel-per-pixel e' diverso a seconda del
                # ramo: col reference locale si confronta contro QUEL frame preciso
                # (non c'e' un 'background mediato' per un singolo frame — e' un
                # confronto diretto, piu' sensibile a rumore/luce momentanei di un
                # singolo frame rispetto alla mediana, ma confronta il posto giusto
                # invece di uno sbagliato, che e' il problema piu' grave da risolvere)
                target_confronto = riferimento_gray_eq if self.sync_map is not None else ref_median_gray_eq
                raw_pixel_boxes, mask = ChangeDetector.detect_changes_gray(
                    target_confronto, aligned_gray_eq, diff_thresh=90, min_area=8000
                )
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidate_detections = []

            # 1. YOLO ostacoli (De Paolis, invariato)
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
                                                                self.query_video, coords)
                                frame_annotato = self._frame_with_box(
                                    qf, (bx, by, bw_b, bh_b), (0, 0, 255), class_name
                                )
                                self.dispatch_alert(alert, frame_annotato)
                                last_alert_frame[class_name] = idx

                            candidate_detections.append(
                                {'bbox': (bx, by, bw_b, bh_b), 'label': class_name, 'conf': conf})

            # 2. Anomalie geometriche via background subtraction (De Paolis, invariato
            #    salvo i due fix di allineamento/illuminazione — limite noto sopra)
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
                                                        self.query_video, coords)
                        # NOTA: nessuna immagine allegata qui (dispatch_alert(alert, None)
                        # equivale a send_alert semplice). Questo modulo scatta su quasi
                        # ogni frame (rumore noto, vedi limite documentato del background
                        # subtraction su reference non sincronizzato) — allegare un'immagine
                        # da ~600KB a ciascuno intaserebbe DB e rete inutilmente.
                        self.dispatch_alert(alert, None)
                        last_alert_frame[label_geo] = idx

            annotated = aligned.copy()
            for det in candidate_detections:
                dx, dy, dww, dhh = det['bbox']
                cv2.rectangle(annotated, (dx, dy), (dx + dww, dy + dhh), (0, 0, 255), 2)
                lat = coords["lat"] if coords else ""
                lon = coords["lon"] if coords else ""
                csvw.writerow([idx, round(t_sec, 2), dx, dy, dww, dhh, det['label'],
                               round(det['conf'], 2), lat, lon])

            # ============================================================
            # 3. ANALISI INFRASTRUTTURA CON DeepLabV3+ — MODULO INDIPENDENTE
            #    Analizza il frame query originale (non l'aligned del modulo De Paolis,
            #    che dipende da un'omografia riuscita): scartamento, pali, vegetazione.
            # ============================================================
            if self.analyzer is not None and idx % self.seg_step == 0:
                qf_for_analysis = cv2.resize(qf, (w, h))

                # se il modulo CNN e' caricato, la sua funzione predict_angle
                # viene passata come tilt_estimator: DeepLabAnalyzer la userà
                # per TUTTI i componenti "a forma di palo" individuati nella
                # maschera, non solo per quelli che minAreaRect avrebbe gia'
                # segnalato oltre soglia — altrimenti un falso negativo del
                # metodo geometrico (sottostima dell'angolo su una maschera di
                # segmentazione rumorosa) impedirebbe alla CNN di intervenire
                # proprio nei casi dove servirebbe di piu'.
                tilt_estimator = (self.pole_tilt_analyzer.predict_angle
                                  if self.pole_tilt_analyzer is not None else None)
                dl_anomalies = self.analyzer.analyze(qf_for_analysis, tilt_estimator=tilt_estimator)

                for a in dl_anomalies:
                    label = a["label"]
                    bbox = a.get("bbox", (0, 0, w, h))

                    # colore per tipo di anomalia (spostato qui, prima serviva
                    # solo per il video annotato locale, ora serve anche per
                    # l'immagine allegata all'alert)
                    if "SCARTAMENTO" in label:
                        color = (0, 165, 255)
                    elif "PALO" in label:
                        color = (255, 0, 0)
                    elif "VEGETAZIONE" in label:
                        color = (0, 200, 0)
                    else:
                        color = (0, 0, 255)

                    if idx - last_alert_frame.get(label, -100) > self.seg_step * 2:
                        alert = self._create_alert_dict(
                            idx, t_sec, bbox[0], bbox[1], bbox[2], bbox[3],
                            bbox[2] * bbox[3], label, a["conf"], a["severity"],
                            a["details"], self.query_video, coords
                        )
                        frame_annotato = self._frame_with_box(
                            qf_for_analysis, bbox, color, f"{label} ({a['severity']})"
                        )
                        self.dispatch_alert(alert, frame_annotato)
                        last_alert_frame[label] = idx

                    candidate_detections.append({'bbox': bbox, 'label': label, 'conf': a["conf"]})

                    bx_a, by_a, bw_a, bh_a = bbox
                    cv2.rectangle(annotated, (bx_a, by_a), (bx_a + bw_a, by_a + bh_a), color, 2)
                    cv2.putText(annotated, f"{label} ({a['severity']})",
                                (bx_a, max(by_a - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    lat = coords["lat"] if coords else ""
                    lon = coords["lon"] if coords else ""
                    csvw.writerow([idx, round(t_sec, 2), bbox[0], bbox[1], bbox[2], bbox[3],
                                   label, round(a['conf'], 2), lat, lon])

            # overlay informativo con le coordinate correnti, se disponibili
            if coords:
                coord_txt = f"LAT:{coords['lat']:.5f} LON:{coords['lon']:.5f}"
                cv2.putText(annotated, coord_txt, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            # overlay informativo: segnala quando il modulo di background-
            # subtraction e' stato saltato per mancanza di corrispondenza nella
            # mappa di sincronizzazione — utile per capire a colpo d'occhio, dal
            # video annotato, perche' certi tratti non producono mai alert
            # ANOMALIA_STRUTTURALE (non e' un problema, e' voluto)
            if saltato_per_mancanza_corrispondenza:
                cv2.putText(annotated, "[no sync: modulo strutturale saltato]",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            writer.write(annotated)
            idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()
        csv_f.close()
        print(f"✅ Analisi terminata. Risultati in '{self.out_dir}'")

    def _create_alert_dict(self, idx, t, x, y, ww, hh, area, label, conf, severity, details,
                           query_video, coords: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        alert = {
            "project": self.project_name, "frame_idx": int(idx), "time_s": round(t, 2),
            "bbox": {"x": int(x), "y": int(y), "w": int(ww), "h": int(hh)},
            "area": int(area), "label": label, "conf": float(conf), "severity": severity,
            "details": details, "source_video": os.path.basename(query_video),
            "trattaId": self.tratta_id
        }
        # campo location presente solo se le coordinate sono state lette con successo
        if coords:
            alert["location"] = coords
        return alert