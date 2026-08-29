# ShortMaster Vision Project

Progetto di visione per TIAGo Pro: il robot impara a riconoscere alcuni
oggetti sul tavolo (lattina di coca, confezione di pringles, biscotti,
marker aruco) come riferimento di posa per un task di grasping.

Il cuore del progetto e' il pacchetto ROS2 [`vision_pipeline`](ros_ws/src/vision_pipeline)
(dettagli tecnici nel suo [README](ros_ws/src/vision_pipeline/README.md)). A
grandi linee, il lavoro e' passato per queste fasi:

## 1. Raccolta del dataset

Il robot orbita attorno al tavolo (movimento omnidirezionale, testa che si
muove su/giu' ad ogni tappa) mentre un nodo salva periodicamente i frame
della camera. La raccolta e' stata ripetuta in piu' sessioni, variando ogni
volta la scena per rendere il modello robusto a situazioni realistiche:
lattina in piedi, ruotata, capovolta, nascosta dietro altri oggetti, vista da
angolazioni diverse (nord-est, sud-ovest, ...), scena senza l'oggetto target.
Le immagini grezze raccolte sono poi state annotate (bounding box) e
organizzate in un dataset in formato YOLOv8.

## 2. Training del modello

Un modello YOLOv8 (nano, per poter girare anche su CPU) e' stato addestrato
sul dataset annotato per riconoscere le classi di interesse.

## 3. Sweep di iperparametri (Weights & Biases)

Per non affidarsi a una singola configurazione scelta a mano, e' stata
lanciata una ricerca automatica (sweep bayesiano su W&B) su iperparametri di
training e augmentation, alla ricerca della combinazione che massimizza la
metrica rilevante per il task (mAP50-95 sulle sole classi utili al
grasping). Ogni run dello sweep viene tracciato con i suoi pesi e le sue
metriche, cosi' da poter confrontare gli esperimenti e recuperare il modello
migliore.

## 4. Valutazione

Il modello finale viene validato sul test set (metriche globali e per
classe) e usato per fare inferenza con bounding box disegnati, per
verificare visivamente la qualita' delle predizioni.

## 5. Dal riconoscimento alla presa

Con un modello affidabile, il passo successivo e' usarlo a runtime sul
robot per portare il braccio verso l'oggetto:

- [`detection_and_ranging`](ros_ws/src/detection_and_ranging) — rileva
  l'oggetto in tempo reale (YOLOv8) e, fondendo la detection con la depth
  della camera, ne stima la posizione 3D. Include anche
  `detection_and_occlusion_handler.py`, un nodo distinto con gestione delle
  occlusioni tra gli oggetti tracciati e campionamento di piu' punti sulla
  superficie del target ([`tiago_vision_msgs`](ros_ws/src/tiago_vision_msgs))
  per il fit del cerchio/asse a valle.
- [`pose_optimizer`](ros_ws/src/pose_optimizer) — collegato alla posizione
  reale della lattina (target) e di pringles/biscotti (ostacoli), campiona
  piu' angoli di presa attorno all'oggetto e sceglie, tra quelli
  raggiungibili (MoveIt + KDL), la configurazione del braccio piu' lontana
  dai limiti di giunto. Lavoro in corso.

## Struttura del repo

```
docker_ws/                 # immagine Docker (simulazione PAL + dipendenze YOLO)
ros_ws/                     # workspace ROS2 (colcon)
└── src/
    ├── vision_pipeline/       # dataset + training/eval YOLO + nodi di raccolta dati + launch simulazione
    ├── detection_and_ranging/ # detection + stima posizione 3D a runtime (incl. gestione occlusioni)
    ├── tiago_vision_msgs/     # messaggi custom usati da detection_and_ranging
    ├── pose_optimizer/        # pianificazione IK/posa per il grasping (MoveIt + KDL, sweep sullo yaw)
    └── pal_*, tiago_pro_*/    # pacchetti PAL Robotics per simulazione/robot TIAGo Pro
```

Il launch della simulazione Gazebo (`ros2 launch vision_pipeline simulation.launch.py`)
vive in `vision_pipeline` — e' il punto di partenza comune prima di lanciare
raccolta dataset, detection o pianificazione.
