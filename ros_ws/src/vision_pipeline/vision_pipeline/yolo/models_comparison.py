#!/usr/bin/env python3
"""Grafici riassuntivi di confronto tra i modelli dello sweep W&B.

Genera alcuni grafici a partire da all_sweeps_summary.csv (le metriche vere,
calcolate da train_sweep.py sull'intero test set, non su una singola
immagine) -- quello che conta davvero per stabilire quale run e' la migliore
in media.
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')  # nessun display nel container, solo salvataggio su file
import matplotlib.pyplot as plt

OUTPUT_DIR = "/home/user/ros_workspace/src/vision_pipeline/models/runs/models_comparison"
SUMMARY_CSV = "/home/user/ros_workspace/src/vision_pipeline/models/wandb/all_sweeps_summary.csv"
SWEEP_ID = "sqkfh2ka"

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


def main():
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    rows = load_summary_rows()
    print(f"Righe con metriche valide nel CSV: {len(rows)}")

    plot_grasp_ranking(rows, os.path.join(plots_dir, "map5095_grasp_ranking.png"))
    plot_per_class_comparison(rows, os.path.join(plots_dir, "per_class_vs_average.png"))
    plot_grasp_vs_all(rows, os.path.join(plots_dir, "grasp_vs_all.png"))

    print(f"Grafici salvati in: {plots_dir}")


if __name__ == '__main__':
    main()
