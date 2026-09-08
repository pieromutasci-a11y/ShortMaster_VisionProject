#!/usr/bin/env python3
"""Grafici di confronto tra le run COMPLETE dello sweep W&B.

"Completa" = presente in all_sweeps_summary.csv, cioe' ha finito sia il
training che la valutazione sul test set (train_sweep.py scrive quella riga
solo a fine run). Le run interrotte a meta' (niente riga nel summary, o
results.csv con poche epoche) sono escluse ovunque -- confrontare curve
troncate con curve complete non avrebbe senso.

Nessuna run e' evidenziata/forzata come "la migliore" -- stesso stile
neutro per tutte, colore coerente per run tra i grafici.

Solo le run dello SWEEP W&B -- per il confronto tra i modelli valutati
singolarmente sul test set (evaluate.py), vedi invece
models/runs/models_comparison/<nome_checkpoint>/, una sottocartella per
modello.

Output organizzato in una sottocartella per ogni cosa che si sta plottando,
dentro models/runs/models_comparison_sweep/:
  box_loss/, cls_loss/, dfl_loss/   -- train.png e val.png (results.csv)
  precision_recall/                  -- precision.png, recall.png (results.csv)
  map/                                -- map50.png, map50_95.png (results.csv)
  learning_rate/                     -- learning_rate.png (results.csv)
  confusion_matrix/absolute/         -- <run_id>.png (copiata da ogni run)
  confusion_matrix/normalized/       -- <run_id>.png (copiata da ogni run)
  summary/                           -- classifica, heatmap per classe,
                                         confronto con metrica generale,
                                         accuratezza vs tempo di training
                                         (tutte da all_sweeps_summary.csv)
"""
import csv
import os
import shutil

import matplotlib
matplotlib.use('Agg')  # nessun display nel container, solo salvataggio su file
import matplotlib.pyplot as plt

SWEEP_RUNS_DIR = "/home/user/ros_workspace/src/vision_pipeline/models/wandb/runs/sqkfh2ka/training"
OUTPUT_DIR = "/home/user/ros_workspace/src/vision_pipeline/models/runs/models_comparison_sweep"
SUMMARY_CSV = "/home/user/ros_workspace/src/vision_pipeline/models/wandb/all_sweeps_summary.csv"
SWEEP_ID = "sqkfh2ka"

# Classi rilevanti per il grasping (escluse bookshelf/dinner table, che sono
# solo contesto di scena) -- stesse di GRASP_RELEVANT_CLASSES in evaluate.py.
GRASP_CLASSES = ["coke_can", "pringles_can", "biscuits_pack", "aruco_marker"]
GRASP_CLASS_LABELS = {
    "coke_can": "coke can",
    "pringles_can": "pringles can",
    "biscuits_pack": "biscuits pack",
    "aruco_marker": "aruco marker",
}

NEUTRAL_COLOR = '#4c72b0'

plt.rcParams.update({
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 10,
})


# ----------------------------------------------------------------------
# Caricamento dati
# ----------------------------------------------------------------------

def load_summary_rows():
    """Legge all_sweeps_summary.csv e restituisce le righe dello sweep
    corrente (solo le run complete, per definizione: e' l'unico posto dove
    train_sweep.py scrive una riga a fine run), campi numerici in float."""
    with open(SUMMARY_CSV) as f:
        rows = [row for row in csv.DictReader(f) if row["sweep_id"] == SWEEP_ID]

    numeric_fields = [
        "map50_all", "map5095_all", "precision_all", "recall_all",
        "map5095_grasp", "training_time_min",
    ] + [f"map5095_{c}" for c in GRASP_CLASSES]

    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])
    return rows


def load_training_curves(complete_run_ids):
    """Legge results.csv delle sole run complete, restituendo
    {run_id: [righe per epoca, campi numerici convertiti a float]}."""
    curves = {}
    for run_id in sorted(complete_run_ids):
        results_path = os.path.join(SWEEP_RUNS_DIR, run_id, "results.csv")
        if not os.path.isfile(results_path):
            continue
        with open(results_path) as f:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
        if rows:
            curves[run_id] = rows
    return curves


def build_color_map(run_ids):
    """Un colore fisso per run, coerente su tutti i grafici."""
    cmap = plt.get_cmap('tab20')
    return {run_id: cmap(i % 20) for i, run_id in enumerate(sorted(run_ids))}


# ----------------------------------------------------------------------
# Grafici dalle curve per epoca (results.csv di ogni run)
# ----------------------------------------------------------------------

def plot_single_curve(curves, color_map, column, ylabel, title, output_path):
    """Una singola linea per run, stesso colore su tutti i grafici."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_id, rows in curves.items():
        epochs = [r["epoch"] for r in rows]
        values = [r[column] for r in rows]
        ax.plot(epochs, values, color=color_map[run_id], linewidth=1, label=run_id)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7, title='Run')
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Confusion matrix -- copiate cosi' come sono (gia' generate da Ultralytics
# durante il training di ogni run), non rigenerate
# ----------------------------------------------------------------------

def copy_confusion_matrices(complete_run_ids, output_dir):
    absolute_dir = os.path.join(output_dir, "absolute")
    normalized_dir = os.path.join(output_dir, "normalized")
    os.makedirs(absolute_dir, exist_ok=True)
    os.makedirs(normalized_dir, exist_ok=True)

    copied = 0
    for run_id in sorted(complete_run_ids):
        run_dir = os.path.join(SWEEP_RUNS_DIR, run_id)
        absolute_src = os.path.join(run_dir, "confusion_matrix.png")
        normalized_src = os.path.join(run_dir, "confusion_matrix_normalized.png")
        if os.path.isfile(absolute_src):
            shutil.copy2(absolute_src, os.path.join(absolute_dir, f"{run_id}.png"))
            copied += 1
        if os.path.isfile(normalized_src):
            shutil.copy2(normalized_src, os.path.join(normalized_dir, f"{run_id}.png"))
    return copied


# ----------------------------------------------------------------------
# Grafici dalle metriche finali (all_sweeps_summary.csv)
# ----------------------------------------------------------------------

def plot_map_ranking(rows, output_path):
    """Bar chart: mAP50-95 (classi rilevanti per grasping) per run, ordinate
    decrescente. Stesso colore per tutte le barre — chi e' in cima si vede
    dall'ordinamento, non da un colore diverso."""
    rows_sorted = sorted(rows, key=lambda r: r["map5095_grasp"], reverse=True)
    run_ids = [r["run_id"] for r in rows_sorted]
    values = [r["map5095_grasp"] for r in rows_sorted]
    mean_value = sum(values) / len(values)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(run_ids, values, color=NEUTRAL_COLOR, edgecolor='black', linewidth=0.5)
    ax.axhline(mean_value, color='gray', linestyle='--', linewidth=1,
               label=f'Media sweep ({mean_value:.4f})')

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f'{value:.3f}',
                 ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_ylabel('mAP50-95 (classi rilevanti per grasping)')
    ax.set_xlabel('Run dello sweep')
    ax.set_title('mAP50-95 sul test set, per run — ordinate')
    ax.set_ylim(0, max(values) * 1.15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_per_class_heatmap(rows, output_path):
    """Heatmap run x classe: mAP50-95 per ciascuna classe rilevante,
    ordinato per riga sulla stessa classifica del ranking generale."""
    rows_sorted = sorted(rows, key=lambda r: r["map5095_grasp"], reverse=True)
    run_ids = [r["run_id"] for r in rows_sorted]
    matrix = [[r[f"map5095_{c}"] for c in GRASP_CLASSES] for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(run_ids) + 2))
    im = ax.imshow(matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(GRASP_CLASSES)))
    ax.set_xticklabels([GRASP_CLASS_LABELS[c] for c in GRASP_CLASSES], rotation=30, ha='right')
    ax.set_yticks(range(len(run_ids)))
    ax.set_yticklabels(run_ids, fontsize=8)

    for i in range(len(run_ids)):
        for j in range(len(GRASP_CLASSES)):
            ax.text(j, i, f'{matrix[i][j]:.2f}', ha='center', va='center',
                     color='white' if matrix[i][j] < 0.5 else 'black', fontsize=7)

    fig.colorbar(im, ax=ax, label='mAP50-95')
    ax.set_title('mAP50-95 per classe, per run (test set)')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_grasp_vs_all(rows, output_path):
    """Scatter: mAP50-95 grasping-relevant vs mAP50-95 su tutte le classi."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for row in rows:
        ax.scatter(row["map5095_all"], row["map5095_grasp"], color=NEUTRAL_COLOR,
                   s=60, edgecolor='black', linewidth=0.5)
        ax.annotate(row["run_id"], (row["map5095_all"], row["map5095_grasp"]),
                     fontsize=7, xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel('mAP50-95 (tutte le classi)')
    ax.set_ylabel('mAP50-95 (classi rilevanti per grasping)')
    ax.set_title('Metrica grasping-relevant vs metrica generale, per run (test set)')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_grasp_vs_training_time(rows, output_path):
    """Scatter: mAP50-95 grasping-relevant vs tempo di training."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for row in rows:
        ax.scatter(row["training_time_min"], row["map5095_grasp"], color=NEUTRAL_COLOR,
                   s=60, edgecolor='black', linewidth=0.5)
        ax.annotate(row["run_id"], (row["training_time_min"], row["map5095_grasp"]),
                     fontsize=7, xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel('Tempo di training (minuti)')
    ax.set_ylabel('mAP50-95 (classi rilevanti per grasping)')
    ax.set_title('Accuratezza vs costo computazionale, per run')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    rows = load_summary_rows()
    complete_run_ids = {r["run_id"] for r in rows}
    print(f"Run complete (con riga in all_sweeps_summary.csv): {len(complete_run_ids)}")

    curves = load_training_curves(complete_run_ids)
    print(f"Di queste, con results.csv leggibile: {len(curves)}")
    color_map = build_color_map(curves.keys())

    def mkdir(*parts):
        path = os.path.join(OUTPUT_DIR, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    # --- Loss di training/validazione, una cartella per componente ---
    for loss_name, csv_prefix in [("box_loss", "box_loss"), ("cls_loss", "cls_loss"), ("dfl_loss", "dfl_loss")]:
        loss_dir = mkdir(loss_name)
        plot_single_curve(curves, color_map, f"train/{csv_prefix}", "Loss (training)",
                           f"{loss_name} — training, tutte le run", os.path.join(loss_dir, "train.png"))
        plot_single_curve(curves, color_map, f"val/{csv_prefix}", "Loss (validazione)",
                           f"{loss_name} — validazione, tutte le run", os.path.join(loss_dir, "val.png"))

    # --- Precision / recall ---
    pr_dir = mkdir("precision_recall")
    plot_single_curve(curves, color_map, "metrics/precision(B)", "Precision",
                       "Precision per epoca, tutte le run", os.path.join(pr_dir, "precision.png"))
    plot_single_curve(curves, color_map, "metrics/recall(B)", "Recall",
                       "Recall per epoca, tutte le run", os.path.join(pr_dir, "recall.png"))

    # --- mAP ---
    map_dir = mkdir("map")
    plot_single_curve(curves, color_map, "metrics/mAP50(B)", "mAP50",
                       "mAP50 per epoca, tutte le run", os.path.join(map_dir, "map50.png"))
    plot_single_curve(curves, color_map, "metrics/mAP50-95(B)", "mAP50-95",
                       "mAP50-95 per epoca, tutte le run", os.path.join(map_dir, "map50_95.png"))

    # --- Learning rate ---
    lr_dir = mkdir("learning_rate")
    plot_single_curve(curves, color_map, "lr/pg0", "Learning rate (param group 0)",
                       "Learning rate schedule per epoca, tutte le run", os.path.join(lr_dir, "learning_rate.png"))

    # --- Confusion matrix (copiate, non rigenerate) ---
    cm_dir = mkdir("confusion_matrix")
    copied = copy_confusion_matrices(complete_run_ids, cm_dir)
    print(f"Confusion matrix copiate: {copied}")

    # --- Metriche finali sul test set ---
    summary_dir = mkdir("summary")
    plot_map_ranking(rows, os.path.join(summary_dir, "map5095_grasp_ranking.png"))
    plot_per_class_heatmap(rows, os.path.join(summary_dir, "per_class_heatmap.png"))
    plot_grasp_vs_all(rows, os.path.join(summary_dir, "grasp_vs_all.png"))
    plot_grasp_vs_training_time(rows, os.path.join(summary_dir, "grasp_vs_training_time.png"))

    print(f"\nFatto. Output in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
