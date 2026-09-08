#!/usr/bin/env python3
"""Valutazione di uno o piu' modelli YOLOv8 sul test set del dataset di grasping.

Di default valuta TUTTI i modelli in MODEL_PATHS, uno alla volta, sullo
STESSO test set (data/training_dataset.yolov8/test/, 98 immagini, 6 classi
-- l'unico dataset "completo" presente nel repo, usato per tutto il
progetto finora): confronto equo, stesso metro per tutti.

Per valutarne uno solo, YOLO_MODEL_PATH sovrascrive MODEL_PATHS (comodo
mentre si aggiunge/tara un modello nuovo, senza rilanciare tutti gli
altri):
  YOLO_MODEL_PATH=/path/al/modello.pt python3 evaluate.py

Ogni modello stampa un'intestazione con il proprio path PRIMA dei
risultati (altrimenti, con piu' modelli in sequenza, l'output non direbbe
a quale modello si riferisce) e scrive in una propria sottocartella
dentro models/runs/models_comparison/<nome_checkpoint>/ -- cosi' i
risultati di un modello non sovrascrivono quelli del precedente. Ogni
sottocartella contiene SOLO i dati di quel modello (evaluation/,
prediction/): nessun grafico di confronto incrociato generato da questo
script -- per quello vedi models_comparison_sweep.py (solo run dello
sweep W&B pero', non questi 4 checkpoint).
"""
import os

from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────
MODEL_PATHS = [
    "/home/user/ros_workspace/src/vision_pipeline/models/wandb/runs/sqkfh2ka/training/xl1874f6/weights/best.pt",
    "/home/user/ros_workspace/src/vision_pipeline/models/runs/small_omogeneous_dataset_model_best.pt",
    "/home/user/ros_workspace/src/vision_pipeline/models/runs/small_eterogeneous_dataset_model_best.pt",
    "/home/user/ros_workspace/src/vision_pipeline/models/runs/segmentation_model_best.pt",
]
if 'YOLO_MODEL_PATH' in os.environ:
    MODEL_PATHS = [os.environ['YOLO_MODEL_PATH']]  # un solo modello, non tutti e 4

DATA_YAML = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/data.yaml"
TEST_IMAGES_DIR = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/test/images"
RUNS_DIR = "/home/user/ros_workspace/src/vision_pipeline/models/runs"

# Le classi che contano davvero per il task di grasping (posa target end effector).
# bookshelf e dinner table sono contesto di scena, non oggetti da afferrare/riferimento di posa.
GRASP_RELEVANT_CLASSES = ["coke can", "pringles can", "biscuits pack", "aruco marker"]


def evaluate_one_model(model_path):
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    model_output_dir = os.path.join(RUNS_DIR, "models_comparison", model_stem)

    print("\n" + "#" * 60)
    print(f"# MODELLO: {model_stem}")
    print(f"# Path: {model_path}")
    print(f"# Output: {model_output_dir}")
    print("#" * 60)

    # ─────────────────────────────────────────────
    # VALUTAZIONE SUL TEST SET
    # ─────────────────────────────────────────────
    model = YOLO(model_path)

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        save_json=True,     # salva anche i risultati in formato COCO json
        plots=True,          # genera confusion_matrix.png, PR/F1/P/R curves, ecc.
        project=model_output_dir,
        name="evaluation",
        exist_ok=True,
    )

    class_names = model.names  # dict {0: 'aruco marker', 1: 'bookshelf', ...}

    # metrics.box.maps è un array numpy con la mAP50-95 per ciascuna classe (stesso ordine di class_names)
    maps_per_class = metrics.box.maps

    # metrics.box.p, metrics.box.r, metrics.box.ap50 sono array allineati con class_names.
    # Nota: se una classe non compare nel test set, questi array potrebbero avere lunghezza minore
    # di len(class_names) — in quel caso Ultralytics riporta le classi effettivamente valutate
    # nell'ordine di metrics.box.ap_class_index.
    ap_class_index = metrics.box.ap_class_index  # indici delle classi effettivamente presenti nel test set
    precision_per_class = metrics.box.p
    recall_per_class = metrics.box.r
    ap50_per_class = metrics.box.ap50

    # ─────────────────────────────────────────────
    # STAMPA METRICHE AGGREGATE (riferimento generale, non l'obiettivo primario)
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("METRICHE AGGREGATE (tutte le classi)")
    print("=" * 60)
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.p.mean():.3f}")
    print(f"Recall:    {metrics.box.r.mean():.3f}")

    # ─────────────────────────────────────────────
    # STAMPA METRICHE PER CLASSE (diagnostica)
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("METRICHE PER CLASSE")
    print("=" * 60)
    print(f"{'Classe':<20}{'Precision':>12}{'Recall':>12}{'mAP50':>12}{'mAP50-95':>12}")
    print("-" * 68)

    per_class_map5095 = {}
    for i, class_idx in enumerate(ap_class_index):
        name = class_names[int(class_idx)]
        p = precision_per_class[i]
        r = recall_per_class[i]
        ap50 = ap50_per_class[i]
        ap5095 = maps_per_class[int(class_idx)]
        per_class_map5095[name] = ap5095
        print(f"{name:<20}{p:>12.3f}{r:>12.3f}{ap50:>12.3f}{ap5095:>12.3f}")

    # ─────────────────────────────────────────────
    # FUNZIONE OBIETTIVO: mAP50-95 media sulle classi rilevanti per il grasping
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FUNZIONE OBIETTIVO — mAP50-95 (classi rilevanti per grasping)")
    print("=" * 60)

    grasp_maps = []
    for name in GRASP_RELEVANT_CLASSES:
        if name in per_class_map5095:
            val = per_class_map5095[name]
            grasp_maps.append(val)
            print(f"  {name:<20}: {val:.3f}")
        else:
            print(f"  {name:<20}: non presente nel test set (attenzione, controlla il dataset)")

    if grasp_maps:
        grasp_mean = sum(grasp_maps) / len(grasp_maps)
        print(f"\n>>> mAP50-95 media (grasping-relevant): {grasp_mean:.3f}  <<<")
        print("(questo è il numero da confrontare tra un modello e l'altro)")
    else:
        print("\nATTENZIONE: nessuna delle classi grasping-relevant è stata trovata nel test set.")

    # ─────────────────────────────────────────────
    # DOVE TROVARE I PLOT GENERATI DA ULTRALYTICS
    # ─────────────────────────────────────────────
    save_dir = metrics.save_dir
    print("\n" + "=" * 60)
    print("PLOT E FILE GENERATI")
    print("=" * 60)
    print(f"Cartella: {save_dir}")
    print("  - confusion_matrix.png")
    print("  - confusion_matrix_normalized.png")
    print("  - BoxP_curve.png   (Precision-Confidence)")
    print("  - BoxR_curve.png   (Recall-Confidence)")
    print("  - BoxF1_curve.png  (F1-Confidence)")
    print("  - BoxPR_curve.png  (Precision-Recall)")
    print("  - val_batch*_labels.jpg / val_batch*_pred.jpg  (campione a griglia, poche immagini)")
    print("  - predictions.json (risultati in formato COCO, se save_json=True)")

    # ─────────────────────────────────────────────
    # INFERENCE SU TUTTE LE IMMAGINI DI TEST CON BOUNDING BOX DISEGNATI
    # (una immagine di output per ciascuna immagine di input, non solo il campione a griglia sopra)
    # ─────────────────────────────────────────────
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

    # Stampa i risultati per ogni immagine, come nello script di inferenza originale
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
    print(f"Modelli da valutare ({len(MODEL_PATHS)}):")
    for path in MODEL_PATHS:
        print(f"  - {path}")

    for model_path in MODEL_PATHS:
        evaluate_one_model(model_path)

    print("\n" + "#" * 60)
    print(f"# Fatto -- {len(MODEL_PATHS)} modelli valutati.")
    print(f"# Risultati in: {RUNS_DIR}/models_comparison/<nome_checkpoint>/")
    print("#" * 60)


if __name__ == '__main__':
    main()
