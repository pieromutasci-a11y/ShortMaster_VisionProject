# detection_and_ranging_occlusion_free

Nodi ROS2 che rilevano in tempo reale (YOLOv8) tutti gli oggetti di
interesse sul tavolo (lattina, pringles, biscotti, tavolo stesso) e, fondendo
la detection con l'immagine di depth sincronizzata, ne stimano la posizione
3D in `base_footprint`. **Multi-oggetto**, ma **non** gestisce le occlusioni
in modo robusto (da qui il nome) — un tentativo precedente in quella
direzione (`detection_and_occlusion_handler.py`, con
[`tiago_vision_msgs`](../tiago_vision_msgs) per il fit del cerchio/asse su
piu' punti campionati) e' stato rimosso insieme ai nodi single-object che lo
affiancavano (`rt_object_detection_node.py`, `center_computation.py`):
restano solo le loro versioni multi-oggetto.

| Nodo | Eseguibile | Descrizione |
|---|---|---|
| `rt_object_detection_node_all.py` | `rt_object_detection_all` | Rileva TUTTE le istanze di TUTTE le classi in `TRACKED_CLASSES` (coke can, pringles can, biscuits pack, dinner table), non solo la prima detection di una classe target. Sottoscrive RGB + depth (sincronizzati), pubblica il frame annotato (`yolo/annotated_frame`) e, per classe, la posizione 3D nel frame ottico della camera (`yolo_all/<classe>_position`, `geometry_msgs/PointStamped`) — piu' istanze della stessa classe nello stesso frame sono piu' messaggi con lo stesso `header.stamp`. Depth stimata come mediana di una patch centrale del box (non un singolo pixel), escludendo i pixel coperti da oggetti piu' vicini (`bbox_intersection`) — piu' robusta di una lettura a singolo pixel quando ci sono piu' oggetti sovrapposti in scena. Usa il modello migliore dello sweep W&B (run `xl1874f6`, path hardcoded in cima al file). |
| `center_computation_all.py` | `center_computation_all` | Sottoscrive `yolo_all/<classe>_position` per classe. Internamente trasforma il punto in `base_footprint` via TF (necessario: la correzione ragiona su un "piano orizzontale" che ha senso solo in un frame con asse Z verticale vero — il frame camera e' inclinato) e corregge x,y spostando il punto di un raggio noto (`RADIUS_BY_CLASS`, prior sull'oggetto — 0.04 per le lattine, 0.03 per i biscotti come approssimazione mezzo-spessore) lungo la componente orizzontale della direzione camera→punto (z lasciata invariata: e' gia' un punto valido sull'asse). `dinner table` **non** e' in `RADIUS_BY_CLASS`: il tavolo e' noto a priori in `pose_optimizer` (non dalla detection), quindi centro/asse per questa classe non servirebbero a nessuno — il rilevamento YOLO grezzo resta comunque attivo in `rt_object_detection_node_all.py` (`yolo_all/dinner_table_position`), solo non viene piu' elaborato qui. Per le altre classi il centro va su **due topic per classe, in due frame**: `centers_all/<classe>_center` (ri-trasformato nel frame camera, confrontabile con `yolo_all/<classe>_position`) e `centers_all/<classe>_center_base_footprint` (gia' in `base_footprint`, pronto per chi lo usa nel frame del robot, es. `pose_optimizer`) — piu' `centers_all/<classe>_axis` (`visualization_msgs/Marker`, `LINE_STRIP`, in `base_footprint` perche' deve essere davvero verticale) per il marker RViz, con id e colore diversi per classe e per istanza nello stesso frame. |

`rviz/complete_visualization_all.rviz` mostra questi output (immagine
annotata, point cloud di depth, robot model, assi dei centri per ogni
classe) in un'unica vista — ma per vederci qualcosa (robot, camera, tf) serve
la simulazione (o il robot reale) gia' avviata, altrimenti RViz non ha nulla
da disegnare.

**Un solo comando** per l'intero stack (simulazione + MoveIt + detection
multi-oggetto + center_computation + RViz):
```bash
ros2 launch detection_and_ranging_occlusion_free full_stack_all.launch.py
# se la simulazione e' gia' avviata altrove:
ros2 launch detection_and_ranging_occlusion_free full_stack_all.launch.py launch_simulation:=false
# senza RViz:
ros2 launch detection_and_ranging_occlusion_free full_stack_all.launch.py rviz:=false
```
Include `vision_pipeline/launch/simulation.launch.py` per la simulazione
(nessuna duplicazione della configurazione del mondo/robot) e avvia
direttamente i nodi di detection, center computation e RViz.

`centers_all/coke_can_center_base_footprint` (target),
`centers_all/pringles_can_center_base_footprint` e
`centers_all/biscuits_pack_center_base_footprint` (ostacoli) sono l'input che
[`pose_optimizer`](../pose_optimizer) usa per la pianificazione IK — gia' nel
frame giusto, nessuna trasformazione aggiuntiva necessaria li'.
