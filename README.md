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

## Struttura del repo

```
docker_ws/        # immagine Docker (simulazione PAL + dipendenze YOLO)
ros_ws/            # workspace ROS2 (colcon)
├── src/
│   ├── vision_pipeline/   # pacchetto del progetto: nodi + script YOLO + dataset + modelli
│   └── pal_*, tiago_pro_*/ # pacchetti PAL Robotics per simulazione/robot TIAGo Pro
└── scripts/
```
