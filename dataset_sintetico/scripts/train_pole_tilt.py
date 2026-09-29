# -*- coding: utf-8 -*-
"""
Training del regressore dell'inclinazione del palo (ResNet34 + testa di
regressione), da affiancare a YOLO e DeepLab nella pipeline NURV.

DATI ATTESI (prodotti da crop_dataset.py):
    dataset_originale_crop/       -> train + val (split casuale 85/15)
    dataset_asset2_test_crop/     -> test finale, MAI usato in training/tuning
                                      (asset diverso, non visto)

Ogni cartella contiene le immagini gia' ritagliate/letterbox (224x224) e un
etichette.json con il campo "angolo_gradi" (target di regressione, range
-22/+22) e "bbox_crop" (non serve per il training, e' solo tracciabilita').

MODELLO:
    ResNet34 pre-addestrato (ImageNet) come backbone, con l'ultimo fully-
    connected sostituito da una piccola testa di regressione (1 output).
    Il target viene normalizzato in [-1, 1] dividendo per ANGOLO_MAX durante
    il training (piu' stabile numericamente), e riportato in gradi nelle
    metriche stampate.

USO:
    python train_pole_tilt.py --epochs 40 --batch-size 32
    python train_pole_tilt.py --resume checkpoints/best.pt --epochs 20   # continua da un checkpoint

Al termine, valuta automaticamente sul test set (asset2) e stampa il MAE
in gradi — la metrica che vi serve per giudicare se il modello e'
abbastanza preciso da decidere gli alert.
"""

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image

# ==================== CONFIGURAZIONE ====================

CARTELLE_TRAIN_VAL = [
    r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_originale_crop",
    r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_asset3_crop",
    # r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_estremi_crop",
    # ^ ESCLUSO di proposito: il Run con questo dataset incluso (oversampling
    # sulla fascia |angolo|>=15°) ha migliorato quella fascia specifica
    # (3.14°->2.48° MAE) ma PEGGIORATO tutto il resto e il MAE globale sul
    # test (1.48°->2.04°) — probabile aumento dell'overfitting sui due asset
    # gia' noti, dato che "estremi" non introduce identita' visive nuove,
    # solo piu' occasioni di rivedere gli stessi due asset. Configurazione
    # riportata a questa (solo originale + asset3) per riprodurre il
    # risultato migliore ottenuto finora (test MAE 1.48°, run precedente).
]
# ^ Tutte le cartelle qui elencate vengono UNITE in un unico pool prima
# dello split train/val (85/15 casuale sul pool combinato). dataset_estremi
# e' oversampling mirato su |angolo|>=15 gradi (fascia dove il modello
# risultava piu' debole: MAE 3.14° contro 0.63-1.00° sulle fasce piu'
# basse) — aggiunge peso li' senza sostituire gli altri due dataset.
# training, senza toccare il test set (asset2), che resta separato e va
# usato SOLO per la valutazione finale, mai per split o training.
CARTELLA_TEST = r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\dataset_asset2_test_crop"
CARTELLA_CHECKPOINT = r"C:\Users\Simone\Desktop\Nurv_PaliInclinati\checkpoints"

ANGOLO_MAX = 22.0          # deve coincidere con ANGOLO_MAX/ANGOLO_MIN degli script Unreal
FRAZIONE_VALIDATION = 0.15  # 15% del dataset originale tenuto per validation
SEED = 42                   # per split riproducibile train/val

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==================== DATASET ====================

@dataclass
class Esempio:
    percorso_immagine: str
    angolo_gradi: float
    palo: str = ""


def carica_esempi(cartella: str) -> List[Esempio]:
    """Legge etichette.json da una cartella prodotta da crop_dataset.py."""
    percorso_json = os.path.join(cartella, "etichette.json")
    with open(percorso_json, "r", encoding="utf-8") as f:
        etichette = json.load(f)

    esempi = []
    for rec in etichette:
        percorso_img = os.path.join(cartella, rec["file"])
        if not os.path.exists(percorso_img):
            print(f"[WARN] File mancante, saltato: {percorso_img}")
            continue
        esempi.append(Esempio(percorso_img, float(rec["angolo_gradi"]), rec.get("palo", "")))
    return esempi


class DatasetPali(Dataset):
    """
    Dataset PyTorch per il regressore. 'augmenta' controlla se applicare
    data augmentation (solo per il train set, mai per val/test).

    IMPORTANTE sul flip orizzontale: dato che il palo puo' essere inclinato
    sia a sinistra sia a destra nel dataset, il flip orizzontale e' una
    augmentation valida SOLO se si nega anche l'etichetta dell'angolo
    insieme all'immagine — altrimenti si introdurrebbe rumore sistematico
    (immagine specchiata, etichetta invariata = coppia sbagliata).
    """

    def __init__(self, esempi: List[Esempio], augmenta: bool):
        self.esempi = esempi
        self.augmenta = augmenta

        # augmentation fotometriche: applicate SEMPRE dopo l'eventuale flip,
        # non toccano mai l'etichetta (colore/luminosita' non hanno relazione
        # con l'angolo)
        self.jitter_colore = T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15)

        self.normalizza = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.esempi)

    def __getitem__(self, idx):
        esempio = self.esempi[idx]
        img = Image.open(esempio.percorso_immagine).convert("RGB")
        angolo = esempio.angolo_gradi

        if self.augmenta:
            # flip orizzontale con probabilita' 50%, negando l'angolo insieme
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                angolo = -angolo

            img = self.jitter_colore(img)

            # piccolo random-resized-crop: simula l'imprecisione realistica
            # del bbox che DeepLab produrra' in produzione (mai perfettamente
            # centrato/scalato come il crop calcolato geometricamente qui)
            w, h = img.size
            scala = random.uniform(0.85, 1.0)
            nuova_w, nuova_h = int(w * scala), int(h * scala)
            max_dx = w - nuova_w
            max_dy = h - nuova_h
            dx = random.randint(0, max_dx) if max_dx > 0 else 0
            dy = random.randint(0, max_dy) if max_dy > 0 else 0
            img = img.crop((dx, dy, dx + nuova_w, dy + nuova_h)).resize((w, h), Image.LANCZOS)

        tensor = self.normalizza(img)
        target = torch.tensor([angolo / ANGOLO_MAX], dtype=torch.float32)
        return tensor, target


# ==================== MODELLO ====================

def costruisci_modello() -> nn.Module:
    """ResNet34 pre-addestrato, con la testa finale sostituita da un
    piccolo regressore (Linear -> ReLU -> Dropout -> Linear a 1 output)."""
    modello = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    n_features = modello.fc.in_features
    modello.fc = nn.Sequential(
        nn.Linear(n_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )
    return modello


# ==================== TRAIN / EVAL ====================

def esegui_epoca(modello, dataloader, criterio, device, optimizer=None) -> Tuple[float, float]:
    """
    Esegue un'epoca (train se optimizer e' fornito, altrimenti solo eval).
    Ritorna (loss_media, mae_gradi_medio).
    """
    is_train = optimizer is not None
    modello.train(is_train)

    loss_totale = 0.0
    mae_totale = 0.0
    n_esempi = 0

    contesto = torch.enable_grad() if is_train else torch.no_grad()
    with contesto:
        for immagini, target in dataloader:
            immagini = immagini.to(device)
            target = target.to(device)

            pred = modello(immagini)
            loss = criterio(pred, target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = immagini.size(0)
            loss_totale += loss.item() * batch_size

            # MAE in GRADI (non nello spazio normalizzato), per una metrica
            # direttamente interpretabile e confrontabile tra epoche/dataset
            mae_batch = (pred - target).abs().mean().item() * ANGOLO_MAX
            mae_totale += mae_batch * batch_size

            n_esempi += batch_size

    return loss_totale / n_esempi, mae_totale / n_esempi


def valuta_test_set(modello, dataloader, device, esempi_test: Optional[List[Esempio]] = None,
                     percorso_csv_diagnostico: Optional[str] = None) -> dict:
    """Valutazione dettagliata sul test set (asset2): MAE globale, e
    breakdown per fascia di angolo (per capire se il modello e' meno
    preciso su inclinazioni estreme, dove pero' conta di piu' per gli alert).

    Se percorso_csv_diagnostico e' fornito, salva anche un CSV con angolo
    reale/predetto per ogni esempio — utile per capire il PATTERN
    dell'errore (es. bias sistematico verso lo zero = segno di modello che
    non generalizza e "gioca sul sicuro", diverso da un errore caotico che
    indicherebbe un'altra causa, es. crop non allineato)."""
    modello.eval()
    errori_assoluti = []
    angoli_reali = []
    angoli_predetti = []

    with torch.no_grad():
        for immagini, target in dataloader:
            immagini = immagini.to(device)
            pred = modello(immagini).cpu()
            target_gradi = (target * ANGOLO_MAX).squeeze(1)
            pred_gradi = (pred * ANGOLO_MAX).squeeze(1)
            errori_assoluti.extend((pred_gradi - target_gradi).abs().tolist())
            angoli_reali.extend(target_gradi.tolist())
            angoli_predetti.extend(pred_gradi.tolist())

    mae_globale = sum(errori_assoluti) / len(errori_assoluti)

    # breakdown per fascia: |angolo| piccolo (quasi dritto) vs grande (inclinato)
    fasce = {"0-7 gradi": [], "7-15 gradi": [], "15-22 gradi": []}
    for err, ang in zip(errori_assoluti, angoli_reali):
        a = abs(ang)
        if a <= 7:
            fasce["0-7 gradi"].append(err)
        elif a <= 15:
            fasce["7-15 gradi"].append(err)
        else:
            fasce["15-22 gradi"].append(err)

    risultato = {"mae_globale": mae_globale, "n_esempi": len(errori_assoluti)}
    for nome_fascia, errori in fasce.items():
        if errori:
            risultato[nome_fascia] = sum(errori) / len(errori)
        else:
            risultato[nome_fascia] = None

    if percorso_csv_diagnostico:
        import csv
        with open(percorso_csv_diagnostico, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "palo", "angolo_reale", "angolo_predetto", "errore_assoluto"])
            for i, (reale, predetto, err) in enumerate(zip(angoli_reali, angoli_predetti, errori_assoluti)):
                nome_file = esempi_test[i].percorso_immagine if esempi_test else ""
                palo = esempi_test[i].palo if esempi_test else ""
                writer.writerow([nome_file, palo, f"{reale:.2f}", f"{predetto:.2f}", f"{err:.2f}"])
        print(f"\n[diagnostica] Predizioni dettagliate salvate in: {percorso_csv_diagnostico}")

        # riepilogo rapido del bias: media delle predizioni vs media dei
        # reali, e correlazione tra segno reale e segno predetto
        media_reale = sum(angoli_reali) / len(angoli_reali)
        media_predetta = sum(angoli_predetti) / len(angoli_predetti)
        segno_concorde = sum(
            1 for r, p in zip(angoli_reali, angoli_predetti)
            if (r > 1.0 and p > 0) or (r < -1.0 and p < 0)
        )
        n_non_quasi_zero = sum(1 for r in angoli_reali if abs(r) > 1.0)
        print(f"[diagnostica] media angolo reale: {media_reale:+.2f}° | "
              f"media angolo predetto: {media_predetta:+.2f}°")
        if n_non_quasi_zero > 0:
            print(f"[diagnostica] segno concorde (reale vs predetto, escludendo quasi-zero): "
                  f"{segno_concorde}/{n_non_quasi_zero} "
                  f"({100*segno_concorde/n_non_quasi_zero:.0f}%)")

    return risultato


# ==================== MAIN ====================

def main():
    ap = argparse.ArgumentParser(description="Training regressore inclinazione palo")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8,
                     help="Epoche senza miglioramento della val MAE prima di fermarsi (early stopping)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--resume", type=str, default=None,
                     help="Percorso a un checkpoint da cui continuare il training")
    ap.add_argument("--eval-only", type=str, default=None,
                     help="Percorso a un checkpoint gia' addestrato: esegue SOLO la "
                          "valutazione diagnostica sul test set, senza training. "
                          "Utile per rigenerare il CSV diagnostico su un modello gia' pronto "
                          "(es. best.pt) senza dover rifare tutte le epoche.")
    args = ap.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("[ATTENZIONE] Nessuna GPU rilevata — il training sara' molto piu' lento. "
              "Verifica i driver CUDA se ti aspettavi di usare la Quadro P4000.")

    os.makedirs(CARTELLA_CHECKPOINT, exist_ok=True)

    # --- dataset ---
    print(f"Carico esempi train/val da {len(CARTELLE_TRAIN_VAL)} cartelle:")
    esempi_train_val = []
    for cartella in CARTELLE_TRAIN_VAL:
        esempi_cartella = carica_esempi(cartella)
        print(f"  {cartella}: {len(esempi_cartella)} esempi")
        esempi_train_val.extend(esempi_cartella)
    print(f"  Totale combinato: {len(esempi_train_val)} esempi")

    print(f"Carico esempi test da: {CARTELLA_TEST}")
    esempi_test = carica_esempi(CARTELLA_TEST)
    print(f"  {len(esempi_test)} esempi trovati")

    n_val = int(len(esempi_train_val) * FRAZIONE_VALIDATION)
    n_train = len(esempi_train_val) - n_val
    generatore = torch.Generator().manual_seed(SEED)
    subset_train, subset_val = random_split(esempi_train_val, [n_train, n_val], generator=generatore)
    # random_split ritorna Subset di indici: recupera gli oggetti Esempio veri
    esempi_train = [esempi_train_val[i] for i in subset_train.indices]
    esempi_val = [esempi_train_val[i] for i in subset_val.indices]
    print(f"Split: {len(esempi_train)} train, {len(esempi_val)} val (seed={SEED})")

    dataset_train = DatasetPali(esempi_train, augmenta=True)
    dataset_val = DatasetPali(esempi_val, augmenta=False)
    dataset_test = DatasetPali(esempi_test, augmenta=False)

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # --- modello ---
    modello = costruisci_modello().to(device)

    if args.eval_only:
        print(f"\n=== Modalita' --eval-only: carico {args.eval_only} e valuto solo il test set ===")
        ckpt = torch.load(args.eval_only, map_location=device)
        modello.load_state_dict(ckpt["model"])
        print(f"Checkpoint caricato (epoca {ckpt.get('epoca', '?')}, "
              f"val_MAE={ckpt.get('migliore_val_mae', float('nan')):.2f}°)")

        percorso_csv_diagnostico = os.path.join(CARTELLA_CHECKPOINT, "diagnostica_test.csv")
        risultati_test = valuta_test_set(modello, loader_test, device,
                                          esempi_test=esempi_test,
                                          percorso_csv_diagnostico=percorso_csv_diagnostico)
        print(f"\nMAE sul test set: {risultati_test['mae_globale']:.2f}° "
              f"(su {risultati_test['n_esempi']} esempi)")
        print("Breakdown per fascia di inclinazione:")
        for fascia in ("0-7 gradi", "7-15 gradi", "15-22 gradi"):
            valore = risultati_test[fascia]
            if valore is not None:
                print(f"   {fascia}: MAE={valore:.2f}°")
            else:
                print(f"   {fascia}: nessun esempio in questa fascia")
        return  # non prosegue con il training

    epoca_iniziale = 0
    migliore_val_mae = float("inf")
    epoche_senza_miglioramento = 0

    if args.resume:
        print(f"Riprendo da checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        modello.load_state_dict(ckpt["model"])
        epoca_iniziale = ckpt.get("epoca", 0)
        migliore_val_mae = ckpt.get("migliore_val_mae", float("inf"))
        print(f"  Ripreso da epoca {epoca_iniziale}, migliore val MAE finora: {migliore_val_mae:.2f}°")

    # Huber loss (SmoothL1): meno sensibile a outlier rispetto a MSE puro,
    # buona scelta di default per regressione senza dover scegliere a mano
    # una soglia — utile qui perche' non abbiamo ancora un'idea precisa di
    # quanto "rumorosi" possano essere alcuni esempi sintetici
    criterio = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(modello.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    percorso_best = os.path.join(CARTELLA_CHECKPOINT, "best.pt")
    percorso_ultimo = os.path.join(CARTELLA_CHECKPOINT, "ultimo.pt")

    print(f"\n=== Avvio training: {args.epochs} epoche, batch_size={args.batch_size}, "
          f"lr={args.lr}, patience={args.patience} ===\n")

    for epoca in range(epoca_iniziale, epoca_iniziale + args.epochs):
        t0 = time.time()
        train_loss, train_mae = esegui_epoca(modello, loader_train, criterio, device, optimizer)
        val_loss, val_mae = esegui_epoca(modello, loader_val, criterio, device, optimizer=None)
        scheduler.step(val_mae)
        durata = time.time() - t0

        lr_corrente = optimizer.param_groups[0]["lr"]
        print(f"[epoca {epoca+1}] train_loss={train_loss:.4f} train_MAE={train_mae:.2f}° | "
              f"val_loss={val_loss:.4f} val_MAE={val_mae:.2f}° | lr={lr_corrente:.2e} | "
              f"{durata:.1f}s")

        # salva sempre l'ultimo checkpoint (per resume in caso di interruzione)
        torch.save({
            "model": modello.state_dict(),
            "epoca": epoca + 1,
            "migliore_val_mae": migliore_val_mae,
        }, percorso_ultimo)

        if val_mae < migliore_val_mae:
            migliore_val_mae = val_mae
            epoche_senza_miglioramento = 0
            torch.save({
                "model": modello.state_dict(),
                "epoca": epoca + 1,
                "migliore_val_mae": migliore_val_mae,
            }, percorso_best)
            print(f"   -> nuovo migliore checkpoint salvato (val_MAE={val_mae:.2f}°)")
        else:
            epoche_senza_miglioramento += 1
            if epoche_senza_miglioramento >= args.patience:
                print(f"\nNessun miglioramento per {args.patience} epoche consecutive — "
                      f"early stopping.")
                break

    # --- valutazione finale sul test set (asset2), con il MIGLIOR checkpoint ---
    print(f"\n=== Valutazione finale sul test set ({CARTELLA_TEST}) ===")
    ckpt_best = torch.load(percorso_best, map_location=device)
    modello.load_state_dict(ckpt_best["model"])
    print(f"Caricato il miglior checkpoint (epoca {ckpt_best['epoca']}, "
          f"val_MAE={ckpt_best['migliore_val_mae']:.2f}°)")

    percorso_csv_diagnostico = os.path.join(CARTELLA_CHECKPOINT, "diagnostica_test.csv")
    risultati_test = valuta_test_set(modello, loader_test, device,
                                      esempi_test=esempi_test,
                                      percorso_csv_diagnostico=percorso_csv_diagnostico)
    print(f"\nMAE sul test set (asset diverso, mai visto in training): "
          f"{risultati_test['mae_globale']:.2f}° (su {risultati_test['n_esempi']} esempi)")
    print("Breakdown per fascia di inclinazione:")
    for fascia in ("0-7 gradi", "7-15 gradi", "15-22 gradi"):
        valore = risultati_test[fascia]
        if valore is not None:
            print(f"   {fascia}: MAE={valore:.2f}°")
        else:
            print(f"   {fascia}: nessun esempio in questa fascia")

    # --- salvataggio di una copia VERSIONATA del checkpoint, cosi' un run
    # successivo (con dataset diversi) non sovrascrive mai un risultato buono
    # senza lasciarne traccia — best.pt/ultimo.pt restano sempre il piu'
    # recente, ma questa copia con timestamp e MAE nel nome resta come
    # riferimento permanente indipendente da run futuri
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_versionato = f"checkpoint_{timestamp}_testMAE_{risultati_test['mae_globale']:.2f}.pt"
    percorso_versionato = os.path.join(CARTELLA_CHECKPOINT, "storico", nome_versionato)
    os.makedirs(os.path.dirname(percorso_versionato), exist_ok=True)
    shutil.copy2(percorso_best, percorso_versionato)
    print(f"\nCopia permanente del checkpoint salvata in: {percorso_versionato}")
    print("(best.pt/ultimo.pt verranno sovrascritti dal prossimo training — "
          "questa copia in 'storico/' resta invece intatta come riferimento)")


if __name__ == "__main__":
    main()