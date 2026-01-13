import cv2
from pathlib import Path
import shutil
from tqdm import tqdm
from typing import Optional, List

# === CONFIGURAZIONE ===
BASE = Path("data/gen")
CLASSES = ["animals", "motos", "persons", "rocks", "vehicles"]
MODE = "copy"  # "move" oppure "copy"
THRESH = 127   # soglia binarizzazione

def find_mask_for_image(image_path: Path, masks_folder: Path) -> Optional[Path]:
    """Cerca la maschera corrispondente a un'immagine."""
    candidates = [
        masks_folder / f"{image_path.stem}_mask.jpg",
        masks_folder / f"{image_path.stem}_mask.png",
        masks_folder / f"{image_path.stem}.png",
        masks_folder / f"{image_path.stem}.jpg",
        masks_folder / f"{image_path.stem}_mask.jpeg",
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def mask_to_yolo(mask_path: Path, class_index: int) -> List[str]:
    """Converte una maschera binaria in formato YOLO."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[WARN] Maschera non trovata o corrotta: {mask_path}")
        return []

    _, binary = cv2.threshold(mask, THRESH, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape

    lines = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < 50:  # filtra oggetti troppo piccoli
            continue
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        lines.append(f"{class_index} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines

def main():
    for cls_idx, cls_name in enumerate(CLASSES):
        images_folder = BASE / cls_name / "imgs"
        masks_folder = BASE / cls_name / "masks"
        labels_folder = BASE / cls_name / "labels"
        labels_folder.mkdir(parents=True, exist_ok=True)

        for img_file in tqdm(list(images_folder.glob("*.*")), desc=f"Converto {cls_name}"):
            mask_path = find_mask_for_image(img_file, masks_folder)
            label_file = labels_folder / f"{img_file.stem}.txt"

            if mask_path:
                lines = mask_to_yolo(mask_path, cls_idx)
                label_file.write_text("\n".join(lines))
            else:
                label_file.write_text("")  # etichetta vuota


    print("[INFO] Conversione maschere → YOLO completata ✅")

if __name__ == "__main__":
    main()
