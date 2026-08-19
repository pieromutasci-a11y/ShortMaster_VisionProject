#!/usr/bin/env python3
"""Training YOLOv8 sul dataset di grasping (coke can, pringles can, biscuits pack, aruco marker)."""
from ultralytics import YOLO

DATA_YAML = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/data.yaml"
RUNS_DIR = "/home/user/ros_workspace/src/vision_pipeline/models/runs"


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=4,
        name="short_master_detection_model",
        project=RUNS_DIR,
        device='cpu',
        patience=0,        # 0 = disabilita early stopping
        workers=0,
        plots=True,
        exist_ok=True,
    )

    # ─────────────────────────────────────────────
    # EVALUATION
    # ─────────────────────────────────────────────
    metrics = model.val(split="test")
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.p.mean():.3f}")
    print(f"Recall:    {metrics.box.r.mean():.3f}")


if __name__ == '__main__':
    main()
