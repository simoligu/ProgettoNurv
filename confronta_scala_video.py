# -*- coding: utf-8 -*-
"""
Diagnostica rapida: misura lo scartamento (in pixel) su un frame di reference
e uno di query con measure_gauge(), senza lanciare l'intera pipeline.

Risponde alla domanda: "i due video condividono la stessa scala pixel/metro
per le rotaie, o uno dei due e' ripreso con zoom/focale diversa?" — se i due
valori sono vicini, la scala e' compatibile e il problema va cercato altrove;
se sono chiaramente diversi (piu' di qualche punto percentuale), i due video
non sono comparabili in pixel cosi' come sono, indipendentemente dalla
risoluzione (identica, gia' verificato con ffprobe).

Lancio (dalla radice di ProgettoNurv, con .venv attivo):
    python confronta_scala_video.py --reference data/videos/reference.mp4 \
        --query data/videos/query.mp4 --deeplab runs_seg/deeplab_hires/best.pt \
        --frame_idx 100
"""

import argparse
import cv2

from deeplab_analyzer import DeepLabAnalyzer


def frame_pulito(percorso_video: str, frame_idx: int):
    cap = cv2.VideoCapture(percorso_video)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire {percorso_video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Impossibile leggere il frame {frame_idx} da {percorso_video}")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--deeplab", default="runs_seg/deeplab_hires/best.pt")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--frame_idx", type=int, default=100,
                     help="Indice del frame da estrarre da ENTRAMBI i video per il confronto "
                          "(default 100: abbastanza avanti da evitare eventuali frame neri/di "
                          "transizione a inizio video, ma modificabile)")
    args = ap.parse_args()

    analyzer = DeepLabAnalyzer(weights_path=args.deeplab, imgsz=args.imgsz)

    print(f"[INFO] Estraggo frame {args.frame_idx} da reference e query...")
    frame_ref = frame_pulito(args.reference, args.frame_idx)
    frame_query = frame_pulito(args.query, args.frame_idx)

    print("[INFO] Segmento entrambi i frame...")
    class_map_ref = analyzer.segment(frame_ref)
    class_map_query = analyzer.segment(frame_query)

    gauge_ref = analyzer.measure_gauge(class_map_ref)
    gauge_query = analyzer.measure_gauge(class_map_query)

    print(f"\n=== RISULTATO ===")
    print(f"Scartamento misurato su REFERENCE (frame {args.frame_idx}): "
          f"{gauge_ref:.1f}px" if gauge_ref is not None else "REFERENCE: non misurabile")
    print(f"Scartamento misurato su QUERY     (frame {args.frame_idx}): "
          f"{gauge_query:.1f}px" if gauge_query is not None else "QUERY: non misurabile")

    if gauge_ref is not None and gauge_query is not None:
        diff_pct = abs(gauge_ref - gauge_query) / gauge_ref * 100
        print(f"\nDifferenza: {diff_pct:.1f}%")
        if diff_pct > 10:
            print("[ATTENZIONE] Differenza superiore al 10%: i due video molto probabilmente "
                  "NON condividono la stessa scala pixel/metro (zoom/focale/distanza dai binari "
                  "diversi) — calibrare su uno e misurare sull'altro non e' valido cosi' com'e'.")
        else:
            print("[OK] Differenza contenuta: la scala sembra compatibile tra i due video. "
                  "La causa degli alert va cercata altrove.")


if __name__ == "__main__":
    main()
