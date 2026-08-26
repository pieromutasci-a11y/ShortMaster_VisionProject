#!/usr/bin/env python3
"""Confronto visivo tra tutti i modelli dello sweep W&B su una singola immagine.

Prende un'immagine del test set e fa inferenza con i pesi (`best.pt`) di ogni
run dello sweep, salvando un'immagine annotata per modello in un'unica
cartella — comoda per un confronto rapido a occhio tra le run, senza dover
riaprire le cartelle di valutazione di ognuna.

TEST_IMAGE di default e' un caso difficile: coca, pringles, biscotti e aruco
marker ammassati/sovrapposti (coca e pringles si sovrappongono per il 68%
dell'area del box piu' piccolo) -- buon test per l'occlusion handling, a
differenza di un frame facile e frontale dove quasi tutti i modelli vanno
bene comunque.
"""
import os

import cv2
from ultralytics import YOLO

TEST_IMAGE = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/test/images/frame_010127_png.rf.GhSgR0lXfu45RZdCG1QT.png"
SWEEP_RUNS_DIR = "/home/user/ros_workspace/src/vision_pipeline/models/wandb/runs/sqkfh2ka/training"
OUTPUT_DIR = "/home/user/ros_workspace/src/vision_pipeline/models/runs/models_comparison"

CONFIDENCE_THRESHOLD = 0.25


def main():
    # Sottocartella per immagine, cosi' confronti su immagini diverse
    # (es. un frame facile e uno con occlusioni) non si sovrascrivono a vicenda.
    image_name = os.path.splitext(os.path.basename(TEST_IMAGE))[0]
    output_dir = os.path.join(OUTPUT_DIR, image_name)
    os.makedirs(output_dir, exist_ok=True)

    run_ids = sorted(os.listdir(SWEEP_RUNS_DIR))

    print(f"Immagine di test: {TEST_IMAGE}")
    print(f"Run trovate in {SWEEP_RUNS_DIR}: {len(run_ids)}")
    print(f"Salvo le immagini annotate in: {output_dir}\n")

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
        output_path = os.path.join(output_dir, f"{run_id}.png")
        cv2.imwrite(output_path, annotated_image)

        detections = ", ".join(
            f"{model.names[int(box.cls[0])]} {float(box.conf[0]) * 100:.0f}%"
            for box in result.boxes
        ) or "nessuna detection"
        print(f"[{run_id}] {detections} -> {output_path}")
        compared += 1

    print(f"\nFatto. {compared}/{len(run_ids)} modelli confrontati in {output_dir}")


if __name__ == '__main__':
    main()
