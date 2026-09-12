# detection_and_ranging_occlusion_free

Nodi ROS2 che rilevano in tempo reale (YOLOv8) tutti gli oggetti di
interesse sul tavolo (lattina, pringles, biscotti, tavolo stesso) e, fondendo
la detection con l'immagine di depth sincronizzata, ne stimano la posizione
3D in `base_footprint`. **Multi-oggetto**, ma **non** gestisce le occlusioni
tra oggetti tracciati in modo robusto (da qui il nome) — per quello vedi
[`detection_and_ranging_occlusion_handler`](../detection_and_ranging_occlusion_handler),
che pubblica sugli stessi topic ed è quindi intercambiabile con questo.

## Nodi

| Nodo | Eseguibile | Descrizione |
|---|---|---|
| `rt_object_detection_node_all.py` | `rt_object_detection_all` | Rileva tutte le istanze di tutte le classi in `TRACKED_CLASSES` (coke can, pringles can, biscuits pack, dinner table), non solo la prima detection di una classe target. Sottoscrive RGB + depth sincronizzati (`/head_front_camera/color/image_raw`, `/head_front_camera/depth/image_rect_raw`, `/head_front_camera/depth/camera_info`), pubblica il frame annotato (`yolo_all/annotated_frame`) e, per classe, la posizione 3D nel frame ottico della camera (`yolo_all/<classe>_position`, `geometry_msgs/PointStamped`) — più istanze della stessa classe nello stesso frame sono più messaggi con lo stesso `header.stamp`. La depth è stimata come mediana di una patch centrale del bounding box (non un singolo pixel), escludendo i pixel coperti da oggetti più vicini in camera. Usa il modello migliore dello sweep W&B (`MODEL_WEIGHTS_PATH`, path assoluto in cima al file). |
| `center_computation_all.py` | `center_computation_all` | Sottoscrive `yolo_all/<classe>_position` per ogni classe in `RADIUS_BY_CLASS` (coke can, pringles can, biscuits pack — non `dinner table`, il cui centro non serve a nessuno: il tavolo è noto a priori in `pose_optimizer`, non dalla detection). Trasforma il punto in `base_footprint` via TF (necessario: la correzione ragiona su un "piano orizzontale" che ha senso solo in un frame con asse Z verticale vero, e il frame camera è inclinato) e corregge x,y spostando il punto di un raggio noto per classe (`RADIUS_BY_CLASS`: 0.04m per le due lattine, 0.03m per i biscotti come approssimazione a mezzo spessore) lungo la componente orizzontale della direzione camera→punto; z resta invariata. Pubblica il centro corretto su due topic per classe, in due frame: `centers_all/<classe>_center` (ri-trasformato nel frame camera, confrontabile con `yolo_all/<classe>_position`) e `centers_all/<classe>_center_base_footprint` (pronto per chi lo usa nel frame del robot, es. `pose_optimizer`); più `centers_all/<classe>_axis` (`visualization_msgs/Marker`, `LINE_STRIP` verticale in `base_footprint`) per la visualizzazione RViz. |

## Uso

**Un solo comando** per l'intero stack (simulazione + MoveIt + detection
multi-oggetto + center_computation + RViz):
```bash
ros2 launch detection_and_ranging_occlusion_free full_stack_all.launch.py
# se la simulazione è già avviata altrove:
ros2 launch detection_and_ranging_occlusion_free full_stack_all.launch.py launch_simulation:=false
# senza RViz:
ros2 launch detection_and_ranging_occlusion_free full_stack_all.launch.py rviz:=false
```
Argomenti: `launch_simulation` (default `true`), `rviz` (default `true`),
`rviz_config` (default `rviz/complete_visualization_all.rviz`). Il launch
include `vision_pipeline/launch/simulation.launch.py` per la simulazione
(nessuna duplicazione della configurazione del mondo/robot) e avvia
direttamente i nodi di detection, center computation e RViz.

`rviz/complete_visualization_all.rviz` mostra il frame annotato, la point
cloud di depth, il modello del robot e l'asse verticale del centro per ogni
classe tracciata in un'unica vista — per vederci qualcosa serve la
simulazione (o il robot reale) già avviata.

## Per pose_optimizer

`centers_all/coke_can_center_base_footprint` (target),
`centers_all/pringles_can_center_base_footprint` e
`centers_all/biscuits_pack_center_base_footprint` (ostacoli) sono l'input che
[`pose_optimizer`](../pose_optimizer) usa per la pianificazione IK — già nel
frame giusto, nessuna trasformazione aggiuntiva necessaria lì.
