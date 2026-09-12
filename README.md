# TIAGo Pro — Vision & Grasping Project

Progetto di visione e manipolazione per il robot **TIAGo Pro** in simulazione
Gazebo: il robot riconosce alcuni oggetti su un tavolo (lattina di coca,
confezione di pringles, pacco di biscotti, marker ArUco), ne stima la
posizione 3D dalla camera RGB-D, e usa quella posizione per pianificare ed
eseguire una presa con il braccio, evitando gli altri oggetti e il tavolo
come ostacoli.

La pipeline è divisa in quattro fasi, ciascuna corrispondente a uno o più
pacchetti ROS2:

1. **Visione**: raccolta dataset, training e valutazione di un modello YOLO
   ([`vision_pipeline`](ros_ws/src/vision_pipeline)).
2. **Detection e stima 3D a runtime**: due stack alternativi, con e senza
   gestione delle occlusioni tra oggetti
   ([`detection_and_ranging_occlusion_free`](ros_ws/src/detection_and_ranging_occlusion_free),
   [`detection_and_ranging_occlusion_handler`](ros_ws/src/detection_and_ranging_occlusion_handler)).
3. **Messaggi custom** per l'output multi-punto/multi-oggetto della detection
   ([`tiago_vision_msgs`](ros_ws/src/tiago_vision_msgs)).
4. **Pianificazione ed esecuzione della presa**: IK via MoveIt, scelta del
   candidato migliore secondo due criteri alternativi
   ([`pose_optimizer`](ros_ws/src/pose_optimizer)).

Il resto (`pal_*`, `tiago_pro_*`) sono pacchetti forniti da PAL Robotics per
la descrizione del robot e la simulazione Gazebo: non sono stati scritti né
modificati per questo progetto (a parte le pose degli oggetti nel file
mondo, vedi sotto), e non sono documentati qui.

---

## Indice

- [Struttura del repository](#struttura-del-repository)
- [Requisiti e build](#requisiti-e-build)
- [Avvio rapido](#avvio-rapido)
- [Il mondo Gazebo](#il-mondo-gazebo)
- [Pacchetti in dettaglio](#pacchetti-in-dettaglio)
- [Dove trovare dati e modelli](#dove-trovare-dati-e-modelli)

---

## Struttura del repository

```
SHORT_MASTER_HANDS_ON_ROS/
├── README.md                                    # questo file
├── docker_ws/
│   ├── Dockerfile.PAL_YOLO                       # immagine: simulazione PAL + dipendenze YOLO/KDL
│   └── build.sh                                  # docker build -t palrobotics/public-simulation-humble-public
└── ros_ws/                                       # workspace ROS2 (colcon)
    └── src/
        ├── vision_pipeline/                      # dataset, training/valutazione YOLO, nodi di raccolta dati, launch simulazione
        ├── detection_and_ranging_occlusion_free/ # detection + stima 3D a runtime (multi-oggetto, senza gestione occlusioni)
        ├── detection_and_ranging_occlusion_handler/ # come sopra, con gestione robusta delle occlusioni tra oggetti tracciati
        ├── tiago_vision_msgs/                    # messaggi custom usati da detection_and_ranging_occlusion_handler
        ├── pose_optimizer/                       # pianificazione IK/posa per il grasping (MoveIt + KDL)
        ├── pal_gazebo_worlds/                     # [vendor PAL] mondi Gazebo, incluso poliBaMaster.world (il nostro)
        ├── pal_pro_gripper/                       # [vendor PAL] descrizione del gripper
        ├── pal_sea_arm/                           # [vendor PAL] descrizione del braccio
        ├── tiago_pro_robot/                       # [vendor PAL] descrizione/URDF del TIAGo Pro
        ├── tiago_pro_navigation/                  # [vendor PAL] stack di navigazione (SLAM, Nav2)
        └── tiago_pro_simulation/                  # [vendor PAL] launch/config della simulazione Gazebo
```

## Requisiti e build

Il progetto gira dentro l'immagine Docker fornita da PAL Robotics, con in
più le dipendenze Python per YOLO e per la cinematica (`ultralytics`,
`wandb`, `pyyaml`, `matplotlib`, `pin`, `scipy`, con `numpy<2` fissato per
compatibilità binaria con `cv_bridge`/OpenCV/Ultralytics).

```bash
cd docker_ws
./build.sh          # costruisce palrobotics/public-simulation-humble-public
```

All'interno del container, il workspace ROS2 (`ros_ws/`) è montato su
`/home/user/ros_workspace` — è questo il path assoluto che compare hardcoded
in vari script (es. `MODEL_WEIGHTS_PATH`, cartelle `data/`/`models/`).

```bash
cd ros_ws
colcon build --symlink-install
source install/setup.bash
```

## Avvio rapido

Ci sono due workflow indipendenti, a seconda di cosa si vuole fare.

### Detection e presa (uso normale)

`full_stack_*.launch.py` include già la simulazione (Gazebo, SLAM,
navigazione, MoveIt): **non** serve lanciarla a parte prima. Basta:

```bash
# 1. Simulazione + detection + stima 3D, tutto in un comando (scegliere UNO dei due)
ros2 launch detection_and_ranging_occlusion_free full_stack_all.launch.py
# oppure, con gestione robusta delle occlusioni:
ros2 launch detection_and_ranging_occlusion_handler full_stack_occlusion_handler.launch.py

# 2. Pianificazione ed esecuzione della presa, in un altro terminale (uno dei due criteri)
ros2 run pose_optimizer joint_margin_optimizer
# oppure:
ros2 run pose_optimizer manipulability_optimizer
```

`pose_optimizer` pianifica/esegue una volta e termina; gli altri restano in
esecuzione. Se si vuole testare la ricostruzione della posa del tavolo da un
punto diverso dallo spawn nominale, si può spostare il robot con
`ros2 run vision_pipeline teleop` (vedi sotto) prima di lanciare
`pose_optimizer` — non è necessario per il funzionamento base.

### Raccolta dataset

`vision_pipeline simulation.launch.py` si usa a parte solo per questo
workflow (costruzione/ampliamento del dataset YOLO), non per la detection:

```bash
# 1. Solo simulazione (nessuna detection, nessun MoveIt necessario per raccogliere immagini)
ros2 launch vision_pipeline simulation.launch.py

# 2. Muovi il robot a mano attorno alla scena, in un altro terminale...
ros2 run vision_pipeline teleop
# ...oppure fallo orbitare automaticamente attorno al tavolo mentre salva i frame:
ros2 launch vision_pipeline dataset_collection.launch.py \
    save_dir:=/home/user/ros_workspace/src/vision_pipeline/data/raw_captures/nome_sessione
```

## Il mondo Gazebo

Il mondo usato di default è `poliBaMaster.world`
([`pal_gazebo_worlds/worlds/`](ros_ws/src/pal_gazebo_worlds/worlds/poliBaMaster.world)):
un tavolo (`table_0m8`, piano 1.0×0.8×0.03m, superficie a z=0.815m) con sopra
lattina (`s3_cocacola`), pringles (`s3_pringles`) e biscotti
(`s3_biscuits`), tutti modellati come cilindri in Gazebo (raggi rispettivamente
0.04m, 0.04m, 0.029m — il pacco di biscotti **non** è una scatola in questo
modello). Il robot nasce (spawn) in `(5.0, 3.5)`, rivolto verso il tavolo
lungo l'asse +Y.

Le pose di questi tre oggetti sono duplicate come costanti
(`KNOWN_OBJECT_WORLD_XY`) nei due nodi di `pose_optimizer`, dove servono a
ricostruire la posa del tavolo rispetto al robot — **se si spostano gli
oggetti nel world file, va aggiornata anche quella costante** (vedi
`pose_optimizer`).

## Pacchetti in dettaglio

### `vision_pipeline`

Raccolta dataset, training/valutazione YOLO, e il launch della simulazione
condiviso da tutto il progetto. Documentazione completa nel suo
[README](ros_ws/src/vision_pipeline/README.md).

Punti chiave:
- `ros2 launch vision_pipeline simulation.launch.py` — avvia Gazebo, SLAM,
  navigazione e `move_group` (MoveIt). Include anche gli argomenti condivisi
  (`world_name`, `is_public_sim`, `slam`, `navigation`, `moveit`) usati dai
  `full_stack_*.launch.py` degli altri pacchetti, che la includono a loro
  volta — di norma non va lanciato a parte per la detection (vedi
  [Avvio rapido](#avvio-rapido)), solo per la raccolta dataset.
- `ros2 run vision_pipeline teleop` — teleoperazione da tastiera (base, testa,
  torso): `W/A/S/D` muovono la base, `I/P` inclinano la testa su/giù, `J/L`
  la ruotano sin/destra, `Q/E` alzano/abbassano il torso, `SPAZIO` ferma la
  base, `ESC`/`Ctrl+C` esce.
- `ros2 run vision_pipeline orbit_around_table` + `camera_saver` (o il
  launch combinato `dataset_collection.launch.py`) — raccolta dataset:
  il robot orbita attorno al tavolo mentre la camera salva frame.
  Package: `vision_pipeline/vision_pipeline/{orbit_around_table_node.py, camera_saver_node.py}`.
- `vision_pipeline/yolo/` — script standalone (non nodi ROS) per training
  (`train.py`), sweep di iperparametri su Weights & Biases (`sweep/`),
  valutazione sul test set (`evaluate.py`) e confronto tra i modelli
  (`models_comparison_sweep.py`).
- Dati e modelli vivono dentro il pacchetto stesso: vedi
  [Dove trovare dati e modelli](#dove-trovare-dati-e-modelli).

### `detection_and_ranging_occlusion_free`

Detection multi-oggetto (YOLO) + stima 3D dalla depth, **senza** gestione
robusta delle occlusioni tra oggetti tracciati. Documentazione completa nel
suo [README](ros_ws/src/detection_and_ranging_occlusion_free/README.md).

```bash
ros2 launch detection_and_ranging_occlusion_free full_stack_all.launch.py
```
Argomenti: `launch_simulation` (default `true`), `rviz` (default `true`),
`rviz_config`.

Nodi: `rt_object_detection_all` (detection + depth, un topic per classe:
`yolo_all/<classe>_position`), `center_computation_all` (corregge il punto
di superficie in un centro d'asse, `centers_all/<classe>_center*` e
`centers_all/<classe>_axis`, usando un raggio noto a priori per classe).

### `detection_and_ranging_occlusion_handler`

Stessa idea, ma con gestione esplicita delle occlusioni tra oggetti
tracciati: esclude i pixel condivisi con un oggetto più vicino prima di
stimare la depth di ciascuno, e ricava il raggio dell'oggetto misurandolo (fit
di una circonferenza su 3 punti di superficie) invece di assumerlo a priori.
Pubblica sugli **stessi topic** di `detection_and_ranging_occlusion_free`
(`centers_all/<classe>_center*`, `centers_all/<classe>_axis`): i due stack
sono intercambiabili dal punto di vista di `pose_optimizer`. Documentazione
completa nel suo [README](ros_ws/src/detection_and_ranging_occlusion_handler/README.md).

```bash
ros2 launch detection_and_ranging_occlusion_handler full_stack_occlusion_handler.launch.py
```
Argomenti: `launch_simulation`, `rviz`, `rviz_config`, e
`use_occlusion_handling` (default `true`; `false` disattiva la gestione
occlusioni, tornando a un centro bounding box + depth semplice).

Nodi: `rt_object_detection_occlusion_handler`, `center_computation_occlusion_handler`.

### `tiago_vision_msgs`

Due messaggi custom (`TrackedObjectPoints`, `TrackedObjectsArray`) usati da
`detection_and_ranging_occlusion_handler` per trasportare, per ogni oggetto
tracciato, la classe, la forma geometrica nota e fino a 3 punti sulla
superficie visibile. Dettagli nel suo
[README](ros_ws/src/tiago_vision_msgs/README.md).

### `pose_optimizer`

Riceve la posizione 3D di lattina (target), pringles e biscotti (ostacoli)
da uno dei due stack di detection sopra, ricostruisce anche la posa del
tavolo (nota a priori nel mondo, non dalla detection diretta), e usa MoveIt
+ KDL per pianificare **ed eseguire per davvero** una presa con il braccio
sinistro. Documentazione completa nel suo
[README](ros_ws/src/pose_optimizer/README.md).

Richiede `move_group` attivo (avviato da `simulation.launch.py` con
`moveit:=true`, il default). Due nodi, stesso meccanismo di esplorazione
(campiona pose candidate su una circonferenza attorno al target, verifica
IK e collisioni), criterio di scelta diverso:

```bash
ros2 run pose_optimizer joint_margin_optimizer     # sceglie il candidato più lontano dai limiti di giunto
ros2 run pose_optimizer manipulability_optimizer   # sceglie il candidato con manipolabilità (Yoshikawa) più alta
```

Non chiude il gripper: porta solo il braccio nella posa di presa scelta.

## Dove trovare dati e modelli

Tutto vive dentro [`vision_pipeline`](ros_ws/src/vision_pipeline) (montato
su `/home/user/ros_workspace` nel container):

```
ros_ws/src/vision_pipeline/
├── data/
│   ├── raw_captures/             # frame grezzi per sessione di raccolta (camera_saver)
│   └── training_dataset.yolov8/  # dataset annotato, formato YOLOv8 (esportato da Roboflow)
├── models/                       # SOLO pesi veri (.pt), niente output derivato
│   ├── pretrained/yolov8n.pt
│   ├── wandb/                    # run dello sweep W&B: pesi per-run + all_sweeps_summary.csv
│   ├── small_omogeneous_dataset_model_best.pt
│   ├── small_eterogeneous_dataset_model_best.pt
│   ├── extended_eterogeneous_dataset_model_best.pt
│   └── segmentation_model_best.pt
└── models_evaluation/            # output di train.py / evaluate.py / predict.py / models_comparison_sweep.py
    ├── models_comparison/        # una sottocartella per modello (metriche, confusion matrix, predizioni)
    └── models_comparison_sweep/  # grafici di confronto tra le run dello sweep W&B
```

Il modello attivo nei nodi di detection a runtime
(`MODEL_WEIGHTS_PATH` in `rt_object_detection_node_all.py` e
`rt_object_detection_node_occlusion_handler.py`) è il best dello sweep W&B
(`models/wandb/runs/sqkfh2ka/training/xl1874f6/weights/best.pt`).
