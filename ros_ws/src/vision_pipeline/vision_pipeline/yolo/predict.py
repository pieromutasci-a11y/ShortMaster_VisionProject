#!/usr/bin/env python3
"""Inferenza con un modello YOLOv8 pre-addestrato (COCO) su una cartella di immagini."""
from ultralytics import YOLO

# Modello pre-addestrato (nano = leggero/veloce). Verrà scaricato automaticamente
# la prima volta se non presente in locale.
MODEL_NAME = 'yolov8x.pt'

# Cartella con le immagini su cui fare inferenza.
SOURCE_DIR = '/home/user/ros_workspace/src/vision_pipeline/data/raw_captures/original'
RUNS_DIR = '/home/user/ros_workspace/src/vision_pipeline/models/runs'


def main():
    model = YOLO(MODEL_NAME)

    # save=True salva le immagini annotate in <RUNS_DIR>/predict/
    # conf=0.25 è la soglia minima di confidence (regolabile)
    results = model.predict(
        source=SOURCE_DIR,
        save=True,
        conf=0.25,
        project=RUNS_DIR,
        name='predict',
    )

    # Stampa a schermo un riepilogo delle detection per ogni immagine
    for r in results:
        print(f'\nImage: {r.path}')
        if len(r.boxes) == 0:
            print('  No object detected')
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f'  {cls_name}: {conf:.2f} - box: {[round(v, 1) for v in xyxy]}')

    print(f'\nFatto. Images annotated saved in: {RUNS_DIR}/predict/')


if __name__ == '__main__':
    main()
