# vision_pipeline

Pacchetto ROS2 (ament_python) che raccoglie tutto quello che serve per la
pipeline di visione del progetto: raccolta dataset dal robot, teleop di
supporto, e training/valutazione/inferenza YOLOv8.

## Contenuto

### Simulazione (`launch/`)

`ros2 launch vision_pipeline simulation.launch.py` avvia la simulazione
Gazebo del TIAGo Pro (mondo, SLAM, navigazione, MoveIt) — e' il punto di
partenza comune prima di lanciare raccolta dataset, detection o
pianificazione. Argomenti: `world_name` (default `poliBaMaster`),
`is_public_sim`, `slam`, `navigation`, `moveit` (default `True` per tutti —
`moveit` esplicito e non lasciato al default di sistema, perche'
[`pose_optimizer`](../pose_optimizer) ha bisogno di `move_group` attivo).

```bash
ros2 launch vision_pipeline simulation.launch.py
```

### Nodi ROS2 (`vision_pipeline/`)

| Nodo | Eseguibile | Descrizione |
|---|---|---|
| `camera_saver_node.py` | `camera_saver` | Salva un frame ogni N dalla camera del robot in `save_dir` (parametro ROS). |
| `orbit_around_table_node.py` | `orbit_around_table` | Fa percorrere al robot un'orbita omnidirezionale attorno al tavolo, fermandosi periodicamente a muovere la testa (per variare il punto di vista durante la raccolta dataset). |
| `teleop_node.py` | `teleop` | Teleoperazione da tastiera (base, testa, torso). |

Esempi:

```bash
ros2 run vision_pipeline teleop
ros2 run vision_pipeline camera_saver --ros-args -p save_dir:=/home/user/ros_workspace/src/vision_pipeline/data/raw_captures/coke_nordest
ros2 run vision_pipeline orbit_around_table

# oppure, per lanciare orbita + salvataggio camera insieme:
ros2 launch vision_pipeline dataset_collection.launch.py save_dir:=/home/user/ros_workspace/src/vision_pipeline/data/raw_captures/coke_nordest
```

### Script YOLO (`vision_pipeline/yolo/`)

Non sono nodi ROS (non dipendono da rclpy), ma condividono lo stesso workspace
dati/modelli. Le dipendenze Python (`ultralytics`, `wandb`, `pyyaml`,
`matplotlib`) sono installate nell'immagine Docker (`docker_ws/Dockerfile.PAL_YOLO`).

| Script | Eseguibile | Descrizione |
|---|---|---|
| `train.py` | `yolo_train` | Training YOLOv8 su `data/training_dataset.yolov8`. |
| `evaluate.py` | `yolo_evaluate` | Valutazione di un modello sul test set (metriche globali + per classe + inferenza con box disegnati). Modello configurabile via env var `YOLO_MODEL_PATH`. Output in `models/runs/models_comparison/<nome_checkpoint>/` — una sottocartella per modello (`evaluation_test/`: confusion matrix, curve P/R/F1/PR, `predictions.json`; `predictions_test/`: immagini di test con le box disegnate), cosi' lanciandolo su piu' modelli in sequenza (cambiando `YOLO_MODEL_PATH`) i risultati non si sovrascrivono a vicenda. Nessun grafico di confronto incrociato generato da questo script — ogni sottocartella contiene solo i dati DI QUEL modello, il confronto tra sottocartelle si fa a occhio (per un confronto grafico vero, vedi `models_comparison_sweep.py`, pero' solo per le run dello sweep W&B). |
| `predict.py` | `yolo_predict` | Inferenza con un modello YOLOv8 pre-addestrato (COCO) su una cartella di immagini. |
| `models_comparison_sweep.py` | `yolo_models_comparison_sweep` | Grafici di confronto tra le run **complete** dello sweep W&B (quelle con una riga in `all_sweeps_summary.csv`, cioe' che hanno finito training e valutazione — le run interrotte a meta' sono escluse ovunque). Nessuna run evidenziata/forzata come "la migliore", stesso stile neutro per tutte, colore coerente per run tra i grafici. Output in `models/runs/models_comparison_sweep/`, una sottocartella per ogni cosa plottata: `box_loss/`, `cls_loss/`, `dfl_loss/` (train.png + val.png, da `results.csv`), `precision_recall/`, `map/` (map50.png + map50_95.png), `learning_rate/` — tutti curve per epoca; `confusion_matrix/absolute/` e `confusion_matrix/normalized/` — copiate cosi' come sono da ogni run, non rigenerate; `summary/` — classifica sulla metrica ottimizzata, heatmap mAP per classe, confronto con la metrica generale, accuratezza vs tempo di training (da `all_sweeps_summary.csv`). Riguarda solo le run dello sweep, non i checkpoint aggiunti a mano in `models/runs/` (per quelli vedi `evaluate.py` sopra). |
| `sweep/` | — | Script per hyperparameter sweep con W&B (`run_sweep.py`, `resume_sweep.py`, `train_sweep.py`, `sweep_config.yaml`). Vanno lanciati direttamente con `python3` dalla cartella `sweep/` (non sono entry point ROS/console perché dipendono da `sweep_config.yaml` nella stessa cartella). |

```bash
ros2 run vision_pipeline yolo_train
ros2 run vision_pipeline yolo_evaluate
ros2 run vision_pipeline yolo_predict
ros2 run vision_pipeline yolo_models_comparison_sweep
cd install/vision_pipeline/lib/python3*/site-packages/vision_pipeline/yolo/sweep  # oppure src/vision_pipeline/vision_pipeline/yolo/sweep in sviluppo
python3 run_sweep.py
```

## Dati e modelli

Vivono dentro il pacchetto stesso (workspace montato in Docker su
`/home/user/ros_workspace`), cosi' codice, dati e modelli restano insieme:

```
ros_ws/src/vision_pipeline/
├── data/
│   ├── raw_captures/            # frame grezzi catturati da camera_saver, un sottodir per sessione
│   └── training_dataset.yolov8/ # dataset annotato (formato YOLOv8, esportato da Roboflow)
└── models/
    ├── pretrained/yolov8n.pt    # pesi pre-addestrati di partenza
    ├── runs/                    # output di train.py / evaluate.py / predict.py
    └── wandb/                   # output degli sweep W&B (run, pesi per-run, CSV riassuntivo)
```

Nota: `data/` e `models/` non sono dichiarati in `setup.py` (non vanno
installati in `install/`, sono troppo grandi e mutano di continuo) — restano
nell'albero sorgente e vengono referenziati dagli script con path assoluti.
