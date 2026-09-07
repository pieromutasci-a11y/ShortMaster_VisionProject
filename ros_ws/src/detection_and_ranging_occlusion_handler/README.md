# detection_and_ranging_occlusion_handler

Nodo ROS2 che rileva gli oggetti tracciati (lattina, pringles, biscotti) in
tempo reale con YOLOv8 e fonde la detection con l'immagine di depth
sincronizzata per stimarne la posizione 3D, **gestendo esplicitamente le
occlusioni tra oggetti tracciati** — a differenza di
[`detection_and_ranging_occlusion_free`](../detection_and_ranging_occlusion_free),
che non lo fa (da qui il nome di entrambi i pacchetti).

**Stato: lavoro in corso.**

| Nodo | Eseguibile | Descrizione |
|---|---|---|
| `rt_object_detection_node_occlusion_handler.py` | `rt_object_detection_occlusion_handler` | Sottoscrive RGB + depth (sincronizzati), pubblica il frame annotato (`yolo/annotated_frame`) e la posizione 3D di ogni oggetto tracciato, nel frame ottico della camera. Due modalita', selezionabili col parametro ROS2 `use_occlusion_handling` (default `True`): **robust**, che per ogni coppia di bounding box sovrapposte esclude i pixel ambigui prima di stimare la depth di ciascun oggetto (cosi' un oggetto occludente e uno occluso non si contaminano a vicenda la stima), ordina gli oggetti per depth pulita, e per l'oggetto target esclude solo i pixel condivisi con oggetti genuinamente piu' vicini; oppure **light** (comportamento semplice, centro bounding box + depth, nessuna gestione occlusioni — adatta solo se gli oggetti non si occludono mai in scena). In modalita' robust campiona anche fino a 3 punti sulla superficie visibile di ciascun oggetto (target compreso) e li pubblica su `yolo/tracked_objects_points` (`tiago_vision_msgs/TrackedObjectsArray`), per un fit del cerchio/asse a valle — con marker RViz dedicati (`yolo/grasp_circle_points`, rosso, e `yolo/anchor_points_marker`, blu, quest'ultimo pubblicato in entrambe le modalita'). |
| `center_computation_occlusion_handler.py` | `center_computation_occlusion_handler` | Sottoscrive `yolo/tracked_objects_points`. Per ciascun oggetto (coke can, pringles can, biscuits pack — tutti e tre cilindri anche nel modello Gazebo usato, nessun raggio noto necessario per il calcolo, a differenza di `center_computation_all.py`), trasforma i suoi 3 punti di superficie in `base_footprint` via TF e ne fitta la circonferenza passante nel piano orizzontale (soluzione esatta per 3 punti, formula chiusa — non un fit ai minimi quadrati): il raggio non e' piu' assunto, e' misurato dai punti stessi. Due controlli scartano il frame invece di pubblicare un centro inaffidabile — stessa filosofia "meglio niente che un dato inventato" usata altrove nel progetto: (1) area del triangolo quasi nulla (3 punti coincidenti, es. quando il nodo di detection non ha trovato una riga abbastanza larga); (2) raggio misurato troppo diverso da quello nominale della classe (`RADIUS_SANITY_FACTOR`) — capita quando un oggetto e' molto occluso da un altro piu' vicino: l'arco visibile e' stretto, i 3 punti sono vicini/quasi allineati, l'area puo' non essere ~0 ma il fit resta comunque mal condizionato (piccolo rumore -> centro molto spostato). Pubblica sugli **stessi topic** di `center_computation_all.py` (`centers_all/<classe>_center`, `centers_all/<classe>_center_base_footprint`, `centers_all/<classe>_axis`): i due stack sono intercambiabili, `pose_optimizer` non richiede nessuna modifica per usare l'uno o l'altro. |

`rviz/complete_visualization_occlusion_handler.rviz` mostra l'immagine
annotata, la point cloud di depth, il robot, i 3 punti di superficie per
oggetto (`yolo/grasp_circle_points`, rosso), il punto ancora per oggetto
(`yolo/anchor_points_marker`, blu) e l'asse verticale del centro ricostruito
per classe (`centers_all/<classe>_axis`) — stessa impostazione della vista
in `detection_and_ranging_occlusion_free`, adattata ai topic di questo
stack (che non ha una posizione grezza separata per classe, solo il punto
legacy della sola lattina target su `yolo/coke_can_position`).

**Un solo comando** per l'intero stack (simulazione + MoveIt + detection
con gestione occlusioni + center_computation + RViz):
```bash
ros2 launch detection_and_ranging_occlusion_handler full_stack_occlusion_handler.launch.py
# se la simulazione e' gia' avviata altrove:
ros2 launch detection_and_ranging_occlusion_handler full_stack_occlusion_handler.launch.py launch_simulation:=false
# senza RViz:
ros2 launch detection_and_ranging_occlusion_handler full_stack_occlusion_handler.launch.py rviz:=false
# modalita' light (niente gestione occlusioni -- niente 3 punti, niente centro pubblicato):
ros2 launch detection_and_ranging_occlusion_handler full_stack_occlusion_handler.launch.py use_occlusion_handling:=false
```
Include `vision_pipeline/launch/simulation.launch.py` per la simulazione
(nessuna duplicazione della configurazione del mondo/robot) e avvia
direttamente i nodi di detection, center computation e RViz.

`MODEL_WEIGHTS_PATH` in cima al file di detection e' ancora hardcoded a un
path assoluto di una macchina specifica — da parametrizzare (vedi TODO nel
file).
