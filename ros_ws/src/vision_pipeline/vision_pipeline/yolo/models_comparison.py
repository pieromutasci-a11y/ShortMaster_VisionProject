#!/usr/bin/env python3
"""Confronto visivo tra tutti i modelli dello sweep W&B su una singola immagine.

Prende un'immagine semplice del test set (coca frontale, ben visibile) e fa
inferenza con i pesi (`best.pt`) di ogni run dello sweep, salvando un'immagine
annotata per modello in un'unica cartella — comoda per un confronto rapido a
occhio tra le run, senza dover riaprire le cartelle di valutazione di ognuna.
"""
import os

import cv2
from ultralytics import YOLO

TEST_IMAGE = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/test/images/frame_000052_png.rf.GXgdiDepiONC8pcTcDzY.png"
SWEEP_RUNS_DIR = "/home/user/ros_workspace/src/vision_pipeline/models/wandb/runs/sqkfh2ka/training"
OUTPUT_DIR = "/home/user/ros_workspace/src/vision_pipeline/models/runs/models_comparison"

CONFIDENCE_THRESHOLD = 0.25


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_ids = sorted(os.listdir(SWEEP_RUNS_DIR))

    print(f"Immagine di test: {TEST_IMAGE}")
    print(f"Run trovate in {SWEEP_RUNS_DIR}: {len(run_ids)}")
    print(f"Salvo le immagini annotate in: {OUTPUT_DIR}\n")

    compared = 0
    for run_id in run_ids:
        weights_path = os.path.join(SWEEP_RUNS_DIR, run_id, "weights", "best.pt")
        if not os.path.isfile(weights_path):
            print(f"[{run_id}] nessun best.pt (run incompleta/interrotta) — salto")
            continue

        model = YOLO(weights_path)
        results = model.predict(source=TEST_IMAGE, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]

        annotated_image = result.plot()
        output_path = os.path.join(OUTPUT_DIR, f"{run_id}.png")
        cv2.imwrite(output_path, annotated_image)

        detections = ", ".join(
            f"{model.names[int(box.cls[0])]} {float(box.conf[0]) * 100:.0f}%"
            for box in result.boxes
        ) or "nessuna detection"
        print(f"[{run_id}] {detections} -> {output_path}")
        compared += 1

    print(f"\nFatto. {compared}/{len(run_ids)} modelli confrontati in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
