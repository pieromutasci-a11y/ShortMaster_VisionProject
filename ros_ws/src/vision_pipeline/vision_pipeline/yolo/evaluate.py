#!/usr/bin/env python3
"""Valuta uno o piu' modelli YOLO sul test set del dataset di grasping,
filtrando/rimappando il ground truth sulle classi note a ciascun modello.

  YOLO_MODEL_PATH=/path/al/modello.pt python3 evaluate.py   (un solo modello)
"""
import json
import os
import shutil
import tempfile

import yaml
from ultralytics import YOLO

MODEL_PATHS = [
    "/home/user/ros_workspace/src/vision_pipeline/models/wandb/runs/sqkfh2ka/training/xl1874f6/weights/best.pt",
    "/home/user/ros_workspace/src/vision_pipeline/models/extended_eterogeneous_dataset_model_best.pt",
    "/home/user/ros_workspace/src/vision_pipeline/models/small_omogeneous_dataset_model_best.pt",
    "/home/user/ros_workspace/src/vision_pipeline/models/small_eterogeneous_dataset_model_best.pt",
    "/home/user/ros_workspace/src/vision_pipeline/models/segmentation_model_best.pt",
]
if 'YOLO_MODEL_PATH' in os.environ:
    MODEL_PATHS = [os.environ['YOLO_MODEL_PATH']]

DATA_YAML = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/data.yaml"
TEST_IMAGES_DIR = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/test/images"
TEST_LABELS_DIR = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/test/labels"
EVALUATION_DIR = "/home/user/ros_workspace/src/vision_pipeline/models_evaluation"

GRASP_RELEVANT_CLASSES = ["coke can", "pringles can", "biscuits pack", "aruco marker"]


def normalize_yolo_class_name(raw_class_name):
    """Uniforma i nomi classe tra run di training diverse (underscore -> spazio, spazi multipli -> uno solo)."""
    return ' '.join(raw_class_name.replace('_', ' ').split())


def load_original_class_names():
    """Nomi classe del dataset completo, in ordine di indice (0..5)."""
    with open(DATA_YAML) as f:
        data = yaml.safe_load(f)
    return data['names']


def build_filtered_test_dataset(model_class_names, original_class_names, tmp_dir, as_polygons=False):
    """Ricostruisce, dentro tmp_dir, un mini dataset di valutazione con il
    ground truth filtrato/rimappato sulle classi note al modello.

    as_polygons=True (modelli di segmentazione): scrive ogni box come un
    rettangolo a 4 vertici, formato che il validator di segmentazione sa
    interpretare (le metriche Box restano corrette, quelle Mask sono solo
    un proxy approssimato).

    Ritorna il path al data.yaml risultante.
    """
    original_name_to_index = {
        normalize_yolo_class_name(name): idx for idx, name in enumerate(original_class_names)
    }
    original_to_model_index = {}
    for model_idx, raw_name in enumerate(model_class_names):
        name = normalize_yolo_class_name(raw_name)
        original_idx = original_name_to_index.get(name)
        if original_idx is not None:
            original_to_model_index[original_idx] = model_idx

    images_dir = os.path.join(tmp_dir, 'test', 'images')
    labels_dir = os.path.join(tmp_dir, 'test', 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for filename in os.listdir(TEST_IMAGES_DIR):
        shutil.copy2(os.path.join(TEST_IMAGES_DIR, filename), os.path.join(images_dir, filename))

    class_instance_counts = [0] * len(model_class_names)
    label_filenames = [f for f in os.listdir(TEST_LABELS_DIR) if f.endswith('.txt')]
    for filename in label_filenames:
        with open(os.path.join(TEST_LABELS_DIR, filename)) as f:
            lines = f.readlines()

        filtered_lines = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            original_class_idx = int(parts[0])
            model_class_idx = original_to_model_index.get(original_class_idx)
            if model_class_idx is not None:
                if as_polygons:
                    xc, yc, w, h = (float(v) for v in parts[1:5])
                    x1, y1 = max(0.0, min(1.0, xc - w / 2)), max(0.0, min(1.0, yc - h / 2))
                    x2, y2 = max(0.0, min(1.0, xc + w / 2)), max(0.0, min(1.0, yc + h / 2))
                    coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                    filtered_lines.append(
                        ' '.join([str(model_class_idx)] + [f'{c:.6f}' for c in coords]) + '\n'
                    )
                else:
                    filtered_lines.append(' '.join([str(model_class_idx)] + parts[1:]) + '\n')
                class_instance_counts[model_class_idx] += 1

        with open(os.path.join(labels_dir, filename), 'w') as f:
            f.writelines(filtered_lines)

    print("Istanze per classe nel ground truth filtrato (diagnostica -- se sono tutte 0, il filtro non ha funzionato):")
    for name, count in zip(model_class_names, class_instance_counts):
        print(f"  {normalize_yolo_class_name(name):<20}: {count}")

    data_yaml_path = os.path.join(tmp_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        yaml.safe_dump({
            'train': images_dir,
            'val': images_dir,
            'test': images_dir,
            'nc': len(model_class_names),
            'names': [normalize_yolo_class_name(n) for n in model_class_names],
        }, f)

    return data_yaml_path


def model_stem_for(model_path):
    """Nome usato per la sottocartella di output e nei log ("best" -> "sweep_best", per disambiguare)."""
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return 'sweep_best' if stem == 'best' else stem


def evaluate_one_model(model_path, original_class_names):
    model_stem = model_stem_for(model_path)
    model_output_dir = os.path.join(EVALUATION_DIR, "models_comparison", model_stem)

    model = YOLO(model_path)
    model_class_names = [model.names[i] for i in sorted(model.names)]
    model_class_names_normalized = [normalize_yolo_class_name(n) for n in model_class_names]

    print("\n" + "#" * 60)
    print(f"# MODELLO: {model_stem}")
    print(f"# Path: {model_path}")
    print(f"# Task: {model.task}")
    print(f"# Classi: {', '.join(model_class_names_normalized)}")
    print(f"# Output: {model_output_dir}")
    print("#" * 60)

    with tempfile.TemporaryDirectory(prefix=f'yolo_eval_{model_stem}_') as tmp_dir:
        filtered_data_yaml = build_filtered_test_dataset(
            model_class_names, original_class_names, tmp_dir,
            as_polygons=(model.task == 'segment'),
        )

        try:
            metrics = model.val(
                data=filtered_data_yaml,
                split="test",
                save_json=True,
                plots=True,
                project=model_output_dir,
                name="evaluation",
                exist_ok=True,
            )
        except Exception as error:
            print(f"\n{'!' * 60}")
            print(f"ATTENZIONE: valutazione (mAP/confusion matrix) fallita per {model_stem}: {error}")
            print("Procedo comunque con l'inferenza visiva (prediction/).")
            print("!" * 60)
            metrics = None

    if metrics is not None:
        maps_per_class = metrics.box.maps

        ap_class_index = metrics.box.ap_class_index
        precision_per_class = metrics.box.p
        recall_per_class = metrics.box.r
        ap50_per_class = metrics.box.ap50

        print("\n" + "=" * 60)
        print(f"METRICHE AGGREGATE (sulle {len(model_class_names)} classi di questo modello)")
        print("=" * 60)
        print(f"mAP50:     {metrics.box.map50:.3f}")
        print(f"mAP50-95:  {metrics.box.map:.3f}")
        print(f"Precision: {metrics.box.p.mean():.3f}")
        print(f"Recall:    {metrics.box.r.mean():.3f}")

        print("\n" + "=" * 60)
        print("METRICHE PER CLASSE")
        print("=" * 60)
        print(f"{'Classe':<20}{'Precision':>12}{'Recall':>12}{'mAP50':>12}{'mAP50-95':>12}")
        print("-" * 68)

        per_class_map5095 = {}
        for i, class_idx in enumerate(ap_class_index):
            name = model_class_names_normalized[int(class_idx)]
            p = precision_per_class[i]
            r = recall_per_class[i]
            ap50 = ap50_per_class[i]
            ap5095 = maps_per_class[int(class_idx)]
            per_class_map5095[name] = ap5095
            print(f"{name:<20}{p:>12.3f}{r:>12.3f}{ap50:>12.3f}{ap5095:>12.3f}")

        print("\n" + "=" * 60)
        print("FUNZIONE OBIETTIVO — mAP50-95 (classi rilevanti per grasping)")
        print("=" * 60)

        grasp_maps = []
        for name in GRASP_RELEVANT_CLASSES:
            if name not in model_class_names_normalized:
                print(f"  {name:<20}: il modello non conosce questa classe")
            elif name in per_class_map5095:
                val = per_class_map5095[name]
                grasp_maps.append(val)
                print(f"  {name:<20}: {val:.3f}")
            else:
                print(f"  {name:<20}: non presente nel test set (attenzione, controlla il dataset)")

        if grasp_maps:
            grasp_mean = sum(grasp_maps) / len(grasp_maps)
            print(f"\n>>> mAP50-95 media (grasping-relevant, sulle classi note a questo modello): {grasp_mean:.3f}  <<<")
            print("(confrontabile con altri modelli solo se conoscono le stesse classi grasping-relevant)")
        else:
            print("\nATTENZIONE: nessuna delle classi grasping-relevant è nota a questo modello o nel test set.")

        metrics_summary = {
            "model": model_stem,
            "model_path": model_path,
            "task": model.task,
            "classes": model_class_names_normalized,
            "aggregate": {
                "map50": round(float(metrics.box.map50), 4),
                "map50_95": round(float(metrics.box.map), 4),
                "precision_mean": round(float(metrics.box.p.mean()), 4),
                "recall_mean": round(float(metrics.box.r.mean()), 4),
            },
            "per_class": {},
            "grasping_relevant": {},
        }
        for i, class_idx in enumerate(ap_class_index):
            name = model_class_names_normalized[int(class_idx)]
            metrics_summary["per_class"][name] = {
                "precision": round(float(precision_per_class[i]), 4),
                "recall": round(float(recall_per_class[i]), 4),
                "map50": round(float(ap50_per_class[i]), 4),
                "map50_95": round(float(maps_per_class[int(class_idx)]), 4),
            }
        for name in GRASP_RELEVANT_CLASSES:
            if name in per_class_map5095:
                metrics_summary["grasping_relevant"][name] = round(float(per_class_map5095[name]), 4)
        if grasp_maps:
            metrics_summary["grasping_relevant"]["mean"] = round(sum(grasp_maps) / len(grasp_maps), 4)

        metrics_summary_path = os.path.join(str(metrics.save_dir), "metrics_summary.json")
        with open(metrics_summary_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\nMetriche numeriche salvate in: {metrics_summary_path}")

        print("\n" + "=" * 60)
        print("PLOT E FILE GENERATI")
        print("=" * 60)
        print(f"Cartella: {metrics.save_dir}")
        print("  - confusion_matrix.png")
        print("  - confusion_matrix_normalized.png")
        print("  - BoxP_curve.png   (Precision-Confidence)")
        print("  - BoxR_curve.png   (Recall-Confidence)")
        print("  - BoxF1_curve.png  (F1-Confidence)")
        print("  - BoxPR_curve.png  (Precision-Recall)")
        print("  - val_batch*_labels.jpg / val_batch*_pred.jpg  (campione a griglia, poche immagini)")
        print("  - predictions.json (risultati in formato COCO, se save_json=True)")
        print("  - metrics_summary.json (P, R, mAP50, mAP50-95: aggregate, per classe, grasping-relevant)")

    print("\n" + "=" * 60)
    print("INFERENCE CON BOUNDING BOX SU TUTTE LE IMMAGINI DI TEST")
    print("=" * 60)

    pred_results = model.predict(
        source=TEST_IMAGES_DIR,
        conf=0.25,
        save=True,
        show_labels=True,
        show_conf=True,
        project=model_output_dir,
        name="prediction",
        exist_ok=True,
    )

    for r in pred_results:
        img_name = os.path.basename(r.path)
        print(f"\nImmagine: {img_name}")
        if len(r.boxes) == 0:
            print("  → nessun oggetto rilevato")
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            name = model.names[cls]
            print(f"  → {name}: {conf*100:.1f}%")

    print(f"\nImmagini con bounding box salvate in: {model_output_dir}/prediction/")


def main():
    original_class_names = load_original_class_names()

    print(f"Modelli da valutare ({len(MODEL_PATHS)}):")
    for path in MODEL_PATHS:
        print(f"  - {path}")

    for model_path in MODEL_PATHS:
        evaluate_one_model(model_path, original_class_names)

    print("\n" + "#" * 60)
    print(f"# Fatto -- {len(MODEL_PATHS)} modelli valutati.")
    print(f"# Risultati in: {EVALUATION_DIR}/models_comparison/<nome_checkpoint>/")
    print("#" * 60)


if __name__ == '__main__':
    main()
