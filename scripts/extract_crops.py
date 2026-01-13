import csv, cv2, os

# Percorsi
video_path = 'data/query.mp4'
csv_path = 'out/detections.csv'
out_dir = 'dataset/raw_crops'
os.makedirs(out_dir, exist_ok=True)

# Apri video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Impossibile aprire il video {video_path}")

count = 0
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame_idx = int(float(row['frame_idx']))
        x, y, w, h = map(lambda v: int(float(v)), (row['x'], row['y'], row['w'], row['h']))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        fname = os.path.join(out_dir, f'crop_{count:06d}.jpg')
        cv2.imwrite(fname, crop)
        count += 1

cap.release()
print(f"✅ {count} crops estratti e salvati in {out_dir}")
