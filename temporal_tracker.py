# -*- coding: utf-8 -*-
"""
TemporalTracker — filtro di persistenza temporale per gli alert.

Implementazione LEGGERA (nessun training, nessun modello) dell'idea di
"ragionamento temporale" proposta da De Paolis (Video Vision Transformer per
distinguere un evento sostenuto da un disturbo momentaneo). Qui il principio e'
lo stesso ma realizzato con un'euristica classica: un'anomalia viene CONFERMATA
solo se ricompare in una posizione simile per almeno N controlli, invece di
scattare al primo frame isolato in cui viene rilevata.

Perche' funziona per tutti e tre i contesti della pipeline:
  - DeepLab (scartamento/pali/vegetazione): un'anomalia strutturale vera resta
    li' (la rotaia allargata, il palo storto, il cespuglio) per molti secondi
    consecutivi, non e' un evento istantaneo.
  - ANOMALIA_STRUTTURALE (background subtraction, De Paolis): un vero cambiamento
    nella scena persiste nella stessa zona per diversi frame; il rumore da
    disallineamento omografico invece tende a comparire in posizioni diverse e
    incoerenti frame per frame (l'errore di matching non e' spazialmente stabile).

Nessun training, nessuna GPU: e' tracking per sovrapposizione spaziale (IoU) tra
detection consecutive dello stesso tipo, con una finestra temporale scorrevole.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Intersection over Union tra due bounding box (x, y, w, h)."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


@dataclass
class _Track:
    """Una traccia: la storia di un'anomalia dello stesso tipo vista nel tempo."""
    label: str
    bbox: Tuple[int, int, int, int]
    last_frame_idx: int
    frame_idx_creazione: int = 0
    hit_count: int = 1
    frames_visti: List[int] = field(default_factory=list)
    frame_ultimo_alert: Optional[int] = None  # None finche' non ha mai alertato

    def __post_init__(self):
        if not self.frames_visti:
            self.frames_visti = [self.last_frame_idx]
        if self.frame_idx_creazione == 0:
            self.frame_idx_creazione = self.last_frame_idx


class TemporalTracker:
    """
    Tiene traccia delle anomalie rilevate nel tempo, per tipo (label), e decide
    se una nuova detection e' la prosecuzione di una traccia esistente (stessa
    zona, stesso tipo, non troppo distante nel tempo) o un evento nuovo/isolato.

    Uso tipico nel loop della pipeline:
        persistente, occorrenze, puo_alertare = tracker.aggiorna(label, bbox, frame_idx)
        if puo_alertare:
            # invia l'alert — accade UNA sola volta per traccia, nel momento
            # esatto in cui viene confermata persistente
    """

    def __init__(self,
                 iou_threshold: float = 0.25,
                 min_occorrenze: int = 2,
                 finestra_frame: int = 450,
                 cooldown_re_alert: Optional[int] = None):
        """
        Args:
            iou_threshold: sovrapposizione minima (IoU) tra bbox consecutivi
                          perche' siano considerati "la stessa anomalia" vista
                          di nuovo. Valori piu' bassi tollerano piu' movimento
                          tra un controllo e l'altro (utile con seg_step alto).
            min_occorrenze: quante volte l'anomalia deve ricomparire (in
                          posizione simile, entro la finestra) prima di essere
                          considerata "confermata" invece che rumore isolato.
            finestra_frame: quanti frame indietro nel tempo si cerca ancora una
                          corrispondenza. Oltre questa finestra la traccia e'
                          considerata "scaduta" e viene rimossa. Va tarata in
                          base a quanto spesso il modulo chiamante analizza i
                          frame (es. seg_step per DeepLab, ogni frame per
                          ANOMALIA_STRUTTURALE — la finestra deve coprire
                          almeno 2-3 controlli consecutivi attesi).
            cooldown_re_alert: se None (default), una traccia genera UN SOLO
                          alert per tutta la sua vita, non importa quanto resti
                          attiva. Se impostato (es. 300 frame), una traccia
                          molto longeva puo' generare un nuovo alert ogni
                          cooldown_re_alert frame invece di restare silenziosa
                          per sempre dopo il primo. IMPORTANTE con camera in
                          movimento: una traccia che resta "sempre attiva" per
                          gran parte del video e' piu' probabilmente una zona
                          dello schermo che intercetta POSIZIONI FISICHE DIVERSE
                          nel mondo reale (per la geometria della scena, es.
                          sempre vicino al centro-basso dove i binari
                          convergono), non la stessa anomalia fisica vista a
                          lungo — un fisico non si muove sullo schermo restando
                          fermo mentre la camera avanza. Il cooldown evita di
                          sopprimere silenziosamente eventi realmente distinti.
        """
        self.iou_threshold = iou_threshold
        self.min_occorrenze = min_occorrenze
        self.finestra_frame = finestra_frame
        self.cooldown_re_alert = cooldown_re_alert
        self._tracce: Dict[str, List[_Track]] = {}  # label -> lista di tracce attive

    def aggiorna(self, label: str, bbox: Tuple[int, int, int, int],
                 frame_idx: int) -> Tuple[bool, int, bool]:
        """
        Registra una nuova detection e ritorna una tripla:
          (e' persistente?, numero di occorrenze finora, puo' generare un alert ORA?)

        "Persistente" = True se questa traccia ha raggiunto min_occorrenze.
        "Puo' generare un alert ORA" = True SOLO nel momento esatto in cui la
        traccia diventa persistente per la prima volta (transizione da non
        confermata a confermata) — non ad ogni controllo successivo, anche se
        la traccia resta persistente e continua a essere rilevata. Questo
        sostituisce un throttling globale "un alert ogni N frame per etichetta"
        (cieco alla posizione: bloccherebbe anche anomalie in una zona
        completamente diversa) con un throttling PER TRACCIA: ogni anomalia
        spazialmente distinta genera esattamente un alert quando confermata,
        indipendentemente da quante altre tracce della stessa etichetta sono
        attive altrove nel frame.

        Se la traccia scade (nessun aggiornamento entro finestra_frame) e la
        stessa zona viene rilevata di nuovo in seguito, parte una traccia NUOVA
        che puo' generare un nuovo alert — comportamento corretto: e' passato
        abbastanza tempo da giustificare una nuova conferma.
        """
        tracce_label = self._tracce.setdefault(label, [])

        # rimuovi le tracce scadute (troppo tempo senza aggiornamenti)
        tracce_label[:] = [t for t in tracce_label
                           if frame_idx - t.last_frame_idx <= self.finestra_frame]

        # cerca una traccia esistente compatibile (stessa zona, IoU sufficiente)
        migliore_traccia = None
        migliore_iou = 0.0
        for t in tracce_label:
            val_iou = iou(t.bbox, bbox)
            if val_iou >= self.iou_threshold and val_iou > migliore_iou:
                migliore_traccia = t
                migliore_iou = val_iou

        if migliore_traccia is not None:
            migliore_traccia.bbox = bbox  # aggiorna posizione (puo' spostarsi leggermente)
            migliore_traccia.last_frame_idx = frame_idx
            migliore_traccia.hit_count += 1
            migliore_traccia.frames_visti.append(frame_idx)
            traccia = migliore_traccia
        else:
            traccia = _Track(label=label, bbox=bbox, last_frame_idx=frame_idx,
                             frame_idx_creazione=frame_idx)
            tracce_label.append(traccia)

        occorrenze = traccia.hit_count
        persistente = occorrenze >= self.min_occorrenze

        if not persistente:
            puo_alertare = False
        elif traccia.frame_ultimo_alert is None:
            # prima conferma di questa traccia: alerta sempre
            puo_alertare = True
        elif self.cooldown_re_alert is not None:
            # traccia gia' alertata in passato: ri-alerta solo se e' trascorso
            # il cooldown (vedi nota nel costruttore sul perche' serve, con
            # camera in movimento, per non sopprimere eventi realmente distinti
            # che coincidono nella stessa zona dello schermo)
            puo_alertare = (frame_idx - traccia.frame_ultimo_alert) >= self.cooldown_re_alert
        else:
            # cooldown_re_alert=None (default): un solo alert per tutta la vita
            # della traccia
            puo_alertare = False

        if puo_alertare:
            traccia.frame_ultimo_alert = frame_idx

        return persistente, occorrenze, puo_alertare

    def stato_tracce(self, label: str) -> List[dict]:
        """Utile per debug/log: ritorna lo stato delle tracce attive per un'etichetta."""
        return [
            {"bbox": t.bbox, "hit_count": t.hit_count,
             "ultimo_frame": t.last_frame_idx, "frames": t.frames_visti}
            for t in self._tracce.get(label, [])
        ]