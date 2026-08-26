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

Oltre al confronto visivo su una singola immagine, genera anche alcuni grafici
riassuntivi a partire da all_sweeps_summary.csv (le metriche vere, calcolate
da train_sweep.py su tutto il test set, non su una singola immagine) --
quello che conta davvero per stabilire quale run e' la migliore in media.
"""
import csv
import os

import cv2
import matplotlib
matplotlib.use('Agg')  # nessun display nel container, solo salvataggio su file
import matplotlib.pyplot as plt
from ultralytics import YOLO

TEST_IMAGE = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/test/images/frame_010127_png.rf.GhSgR0lXfu45RZdCG1QT.png"
SWEEP_RUNS_DIR = "/home/user/ros_workspace/src/vision_pipeline/models/wandb/runs/sqkfh2ka/training"
OUTPUT_DIR = "/home/user/ros_workspace/src/vision_pipeline/models/runs/models_comparison"
SUMMARY_CSV = "/home/user/ros_workspace/src/vision_pipeline/models/wandb/all_sweeps_summary.csv"
SWEEP_ID = "sqkfh2ka"

CONFIDENCE_THRESHOLD = 0.25

# Il modello gia' scelto come migliore (vedi detection_and_ranging), da
# evidenziare nei grafici per un confronto diretto col resto dello sweep.
HIGHLIGHT_RUN_ID = "xl1874f6"

# Classi rilevanti per il grasping (escluse bookshelf/dinner table, che sono
# solo contesto di scena) -- stesse di GRASP_RELEVANT_CLASSES in evaluate.py.
GRASP_CLASSES = ["coke_can", "pringles_can", "biscuits_pack", "aruco_marker"]
GRASP_CLASS_LABELS = {
    "coke_can": "coke can",
    "pringles_can": "pringles can",
    "biscuits_pack": "biscuits pack",
    "aruco_marker": "aruco marker",
}

plt.rcParams.update({
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 10,
})


def load_summary_rows():
    """Legge all_sweeps_summary.csv e restituisce le righe dello sweep
    corrente, con i campi numerici gia' convertiti a float."""
    with open(SUMMARY_CSV) as f:
        rows = [row for row in csv.DictReader(f) if row["sweep_id"] == SWEEP_ID]

    numeric_fields = [
        "map50_all", "map5095_all", "precision_all", "recall_all", "map5095_grasp",
    ] + [f"map5095_{c}" for c in GRASP_CLASSES]

    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])
    return rows


def plot_grasp_ranking(rows, output_path):
    """Bar chart: mAP50-95 (classi rilevanti per grasping) per run, ordinate
    decrescente, con la run scelta (HIGHLIGHT_RUN_ID) evidenziata e la media
    dello sweep segnata da una linea tratteggiata."""
    rows_sorted = sorted(rows, key=lambda r: r["map5095_grasp"], reverse=True)
    run_ids = [r["run_id"] for r in rows_sorted]
    values = [r["map5095_grasp"] for r in rows_sorted]
    mean_value = sum(values) / len(values)
    colors = ['#d62728' if run_id == HIGHLIGHT_RUN_ID else '#4c72b0' for run_id in run_ids]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(run_ids, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(mean_value, color='gray', linestyle='--', linewidth=1,
               label=f'Media sweep ({mean_value:.4f})')

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f'{value:.3f}',
                 ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_ylabel('mAP50-95 (classi rilevanti per grasping)')
    ax.set_xlabel('Run dello sweep')
    ax.set_title('Confronto tra le run dello sweep — metrica ottimizzata (test set)')
    ax.set_ylim(0, max(values) * 1.15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_per_class_comparison(rows, output_path):
    """Bar chart raggruppato: mAP50-95 per classe rilevante, run evidenziata
    vs media delle altre run — dimostra che il vantaggio non viene da una
    sola classe ma e' distribuito su (quasi) tutte."""
    highlight_row = next(r for r in rows if r["run_id"] == HIGHLIGHT_RUN_ID)
    other_rows = [r for r in rows if r["run_id"] != HIGHLIGHT_RUN_ID]

    highlight_values = [highlight_row[f"map5095_{c}"] for c in GRASP_CLASSES]
    other_means = [
        sum(r[f"map5095_{c}"] for r in other_rows) / len(other_rows)
        for c in GRASP_CLASSES
    ]

    x = range(len(GRASP_CLASSES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], highlight_values, width,
           label=HIGHLIGHT_RUN_ID, color='#d62728', edgecolor='black', linewidth=0.5)
    ax.bar([i + width / 2 for i in x], other_means, width,
           label=f'Media altre {len(other_rows)} run', color='#4c72b0', edgecolor='black', linewidth=0.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels([GRASP_CLASS_LABELS[c] for c in GRASP_CLASSES])
    ax.set_ylabel('mAP50-95')
    ax.set_title(f'{HIGHLIGHT_RUN_ID} vs media dello sweep, per classe (test set)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_grasp_vs_all(rows, output_path):
    """Scatter: mAP50-95 sulle classi grasping-relevant vs mAP50-95 su tutte
    le classi — verifica che il vantaggio non sia un artefatto della sola
    metrica ottimizzata dallo sweep."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for row in rows:
        is_highlight = row["run_id"] == HIGHLIGHT_RUN_ID
        ax.scatter(
            row["map5095_all"], row["map5095_grasp"],
            color='#d62728' if is_highlight else '#4c72b0',
            s=90 if is_highlight else 50,
            zorder=3 if is_highlight else 2,
            edgecolor='black', linewidth=0.5,
        )
        ax.annotate(
            row["run_id"], (row["map5095_all"], row["map5095_grasp"]),
            fontsize=7, xytext=(4, 4), textcoords='offset points',
            fontweight='bold' if is_highlight else 'normal',
        )

    ax.set_xlabel('mAP50-95 (tutte le classi)')
    ax.set_ylabel('mAP50-95 (classi rilevanti per grasping)')
    ax.set_title('Metrica ottimizzata vs metrica generale, per run (test set)')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_summary_plots():
    """Genera i grafici riassuntivi dalle metriche vere dello sweep (CSV),
    non dal confronto qualitativo su una singola immagine."""
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    rows = load_summary_rows()
    print(f"\nRighe con metriche valide nel CSV: {len(rows)}")

    plot_grasp_ranking(rows, os.path.join(plots_dir, "map5095_grasp_ranking.png"))
    plot_per_class_comparison(rows, os.path.join(plots_dir, "per_class_vs_average.png"))
    plot_grasp_vs_all(rows, os.path.join(plots_dir, "grasp_vs_all.png"))

    print(f"Grafici salvati in: {plots_dir}")


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

    generate_summary_plots()


if __name__ == '__main__':
    main()
