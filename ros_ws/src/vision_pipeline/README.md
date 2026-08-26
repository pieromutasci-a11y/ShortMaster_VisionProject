# vision_pipeline

Pacchetto ROS2 (ament_python) che raccoglie tutto quello che serve per la
pipeline di visione del progetto: raccolta dataset dal robot, teleop di
supporto, e training/valutazione/inferenza YOLOv8.

## Contenuto

### Simulazione (`launch/`)

`ros2 launch vision_pipeline simulation.launch.py` avvia la simulazione
Gazebo del TIAGo Pro (mondo, SLAM, navigazione) — e' il punto di partenza
comune prima di lanciare raccolta dataset, detection o pianificazione.
Argomenti: `world_name` (default `poliBaMaster`), `is_public_sim`, `slam`,
`navigation` (default `True`).

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
| `evaluate.py` | `yolo_evaluate` | Valutazione di un modello sul test set (metriche globali + per classe + inferenza con box disegnati). Modello configurabile via env var `YOLO_MODEL_PATH`. |
| `predict.py` | `yolo_predict` | Inferenza con un modello YOLOv8 pre-addestrato (COCO) su una cartella di immagini. |
| `models_comparison.py` | `yolo_models_comparison` | Due cose distinte: (1) confronto visivo qualitativo — inferenza con i pesi (`best.pt`) di ogni run su una singola immagine del test set (default: `frame_010127`, caso difficile con coca/pringles/biscotti/aruco ammassati e sovrapposti), un'immagine annotata per modello in `models/runs/models_comparison/<nome_immagine>/<run_id>.png`; (2) grafici riassuntivi **dalle metriche vere** (`all_sweeps_summary.csv`, mAP50-95 calcolata da `train_sweep.py` sull'intero test set, non su una singola immagine) in `models/runs/models_comparison/plots/`: classifica di tutte le run sulla metrica ottimizzata (`map5095_grasp_ranking.png`), la run scelta (`HIGHLIGHT_RUN_ID`) vs media dello sweep per singola classe (`per_class_vs_average.png`), e metrica ottimizzata vs metrica generale per verificare che il vantaggio non sia un artefatto (`grasp_vs_all.png`). Salta automaticamente le run senza `best.pt` (interrotte/incomplete). |
| `sweep/` | — | Script per hyperparameter sweep con W&B (`run_sweep.py`, `resume_sweep.py`, `train_sweep.py`, `sweep_config.yaml`). Vanno lanciati direttamente con `python3` dalla cartella `sweep/` (non sono entry point ROS/console perché dipendono da `sweep_config.yaml` nella stessa cartella). |

```bash
ros2 run vision_pipeline yolo_train
ros2 run vision_pipeline yolo_evaluate
ros2 run vision_pipeline yolo_predict
ros2 run vision_pipeline yolo_models_comparison
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
