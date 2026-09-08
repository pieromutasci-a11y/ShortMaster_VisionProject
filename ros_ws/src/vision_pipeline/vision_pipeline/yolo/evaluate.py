#!/usr/bin/env python3
"""Valutazione di un modello YOLOv8 addestrato sul test set del dataset di grasping."""
import os

from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    'YOLO_MODEL_PATH',
    "/home/user/ros_workspace/src/vision_pipeline/models/wandb/runs/sqkfh2ka/training/xl1874f6/weights/best.pt",
)
DATA_YAML = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/data.yaml"
TEST_IMAGES_DIR = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/test/images"
RUNS_DIR = "/home/user/ros_workspace/src/vision_pipeline/models/runs"

# Cartella di output derivata dal checkpoint valutato (es.
# ".../small_omogeneous_dataset_model_best.pt" ->
# models_comparison/small_omogeneous_dataset_model_best/), non piu' fissa:
# lanciare questo script su piu' modelli in sequenza (cambiando
# YOLO_MODEL_PATH) altrimenti sovrascriveva ogni volta lo stesso
# "evaluation_test"/"predictions_test", perdendo i risultati dei modelli
# precedenti. Una sottocartella per modello dentro models_comparison/: a
# differenza di models_comparison_sweep.py (che genera grafici che
# confrontano piu' run tra loro), qui ogni sottocartella contiene solo i
# grafici/dati DI QUEL modello -- nessun confronto incrociato generato da
# questo script, il confronto lo si fa a occhio guardando le sottocartelle
# una accanto all'altra.
MODEL_STEM = os.path.splitext(os.path.basename(MODEL_PATH))[0]
MODEL_OUTPUT_DIR = os.path.join(RUNS_DIR, "models_comparison", MODEL_STEM)

# Le classi che contano davvero per il task di grasping (posa target end effector).
# bookshelf e dinner table sono contesto di scena, non oggetti da afferrare/riferimento di posa.
GRASP_RELEVANT_CLASSES = ["coke can", "pringles can", "biscuits pack", "aruco marker"]


def main():
    # ─────────────────────────────────────────────
    # VALUTAZIONE SUL TEST SET
    # ─────────────────────────────────────────────
    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        save_json=True,     # salva anche i risultati in formato COCO json
        plots=True,          # genera confusion_matrix.png, PR/F1/P/R curves, ecc.
        project=MODEL_OUTPUT_DIR,
        name="evaluation_test",
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
        print("(questo è il numero da confrontare tra un esperimento e l'altro)")
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
        project=MODEL_OUTPUT_DIR,
        name="predictions_test",
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

    print(f"\nImmagini con bounding box salvate in: {MODEL_OUTPUT_DIR}/predictions_test/")


if __name__ == '__main__':
    main()
