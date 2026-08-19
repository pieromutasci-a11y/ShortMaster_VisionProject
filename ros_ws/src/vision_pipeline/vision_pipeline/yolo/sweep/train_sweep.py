import os
import csv
import time
from pathlib import Path

import wandb
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────
# CONFIGURAZIONE FISSA (non fa parte dello sweep)
# ─────────────────────────────────────────────────────────
DATA_YAML = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/data.yaml"

# Tutto ciò che riguarda W&B (log interni + le nostre cartelle training/evaluation
# + il CSV di riepilogo) sta sotto questa unica cartella.
WANDB_ROOT = Path("/home/user/ros_workspace/src/vision_pipeline/models/wandb")
BASE_DIR = WANDB_ROOT / "runs"          # training/ ed evaluation/ per-run
SUMMARY_CSV = WANDB_ROOT / "all_sweeps_summary.csv"

WANDB_ROOT.mkdir(parents=True, exist_ok=True)
# WANDB_DIR dice all'SDK di W&B dove scrivere i suoi file locali interni
# (cache, log di sincronizzazione, ecc.), invece del default "./wandb".
os.environ.setdefault("WANDB_DIR", str(WANDB_ROOT))

# Parametri "a costo zero" decisi in precedenza: fissi, non nello sweep.
FIXED_PARAMS = dict(
    epochs=100,
    batch=4,
    workers=0,
    patience=25,
    cache='disk',
    device="cpu",
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    plots=True,
    exist_ok=True,
)

# Le classi rilevanti per il task di grasping (posa target end effector).
GRASP_RELEVANT_CLASSES = ["coke can", "pringles can", "biscuits pack", "aruco marker"]

ENTITY = "pieromutasci-politecnico-di-bari"


def main():
    # wandb.init() dentro un run di sweep riceve automaticamente la combinazione
    # di iperparametri scelta da W&B in wandb.config.
    run = wandb.init(entity=ENTITY)
    config = wandb.config

    sweep_id = run.sweep_id if run.sweep_id else "manual_run"
    run_id = run.id

    training_dir = BASE_DIR / sweep_id / "training" / run_id
    evaluation_dir = BASE_DIR / sweep_id / "evaluation" / run_id
    training_dir.parent.mkdir(parents=True, exist_ok=True)
    evaluation_dir.parent.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────
    model = YOLO("yolov8n.pt")

    start_time = time.time()

    model.train(
        data=DATA_YAML,
        imgsz=config.imgsz,
        box=config.box_gain,
        cls=config.cls_gain,
        dfl=config.dfl_gain,
        lr0=config.lr0,
        lrf=config.lrf,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        scale=config.scale,
        translate=config.translate,
        copy_paste=config.copy_paste,
        mosaic=config.mosaic,
        project=str(training_dir.parent),
        name=run_id,
        **FIXED_PARAMS,
    )

    training_time_min = (time.time() - start_time) / 60

    best_weights = training_dir / "weights" / "best.pt"

    # ─────────────────────────────────────────────
    # EVALUATION SUL TEST SET
    # ─────────────────────────────────────────────
    trained_model = YOLO(str(best_weights))

    metrics = trained_model.val(
        data=DATA_YAML,
        split="test",
        plots=True,
        save_json=True,
        project=str(evaluation_dir.parent),
        name=run_id,
        exist_ok=True,
    )

    class_names = trained_model.names
    ap_class_index = metrics.box.ap_class_index
    maps_per_class = metrics.box.maps
    precision_per_class = metrics.box.p
    recall_per_class = metrics.box.r
    ap50_per_class = metrics.box.ap50

    per_class_map5095 = {}
    per_class_metrics_flat = {}
    for i, class_idx in enumerate(ap_class_index):
        name = class_names[int(class_idx)]
        ap5095 = float(maps_per_class[int(class_idx)])
        per_class_map5095[name] = ap5095
        per_class_metrics_flat[f"map5095_{name.replace(' ', '_')}"] = ap5095
        per_class_metrics_flat[f"precision_{name.replace(' ', '_')}"] = float(precision_per_class[i])
        per_class_metrics_flat[f"recall_{name.replace(' ', '_')}"] = float(recall_per_class[i])
        per_class_metrics_flat[f"ap50_{name.replace(' ', '_')}"] = float(ap50_per_class[i])

    # Funzione obiettivo: mAP50-95 media sulle classi rilevanti per il grasping.
    grasp_maps = [per_class_map5095[n] for n in GRASP_RELEVANT_CLASSES if n in per_class_map5095]
    map5095_grasp = sum(grasp_maps) / len(grasp_maps) if grasp_maps else 0.0

    aggregate_metrics = {
        "map50_all": float(metrics.box.map50),
        "map5095_all": float(metrics.box.map),
        "precision_all": float(metrics.box.p.mean()),
        "recall_all": float(metrics.box.r.mean()),
        "map5095_grasp": map5095_grasp,
        "training_time_min": training_time_min,
    }

    # ─────────────────────────────────────────────
    # LOG SU W&B (la dashboard mostrerà questi valori per ogni run)
    # ─────────────────────────────────────────────
    wandb.log({**aggregate_metrics, **per_class_metrics_flat})

    # ─────────────────────────────────────────────
    # APPEND AL CSV GLOBALE (tutti gli sweep, tutti i run)
    # ─────────────────────────────────────────────
    row = {
        "sweep_id": sweep_id,
        "run_id": run_id,
        **dict(config),
        **aggregate_metrics,
        **per_class_metrics_flat,
    }

    write_header = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n>>> Run {run_id} completato. mAP50-95 grasping-relevant: {map5095_grasp:.3f} <<<")
    print(f"Riga aggiunta a {SUMMARY_CSV}")

    wandb.finish()


if __name__ == "__main__":
    main()
