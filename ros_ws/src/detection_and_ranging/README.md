# detection_and_ranging

Nodo ROS2 che rileva l'oggetto target (lattina, pringles, biscotti) in
tempo reale con YOLOv8 e fonde la detection con l'immagine di depth
sincronizzata per stimarne la posizione 3D nel frame ottico della camera.

| Nodo | Eseguibile | Descrizione |
|---|---|---|
| `rt_object_detection_node.py` | `rt_object_detection` | Sottoscrive RGB + depth (sincronizzati), pubblica il frame annotato (`yolo/annotated_frame`) e la posizione 3D dell'oggetto target (`yolo/coke_can_position`, `geometry_msgs/PointStamped`) — il punto misurato sta sulla superficie tangente della lattina (verso la camera), non sul suo asse centrale. Usa il modello migliore dello sweep W&B (run `xl1874f6`, path hardcoded in cima al file). |
| `center_computation.py` | `center_computation` | Sottoscrive `yolo/coke_can_position`. Internamente trasforma il punto in `base_footprint` via TF (necessario: la correzione ragiona su un "piano orizzontale" che ha senso solo in un frame con asse Z verticale vero — il frame camera e' inclinato) e corregge x,y spostando il punto di un raggio noto (`CAN_RADIUS`, prior sull'oggetto) lungo la componente orizzontale della direzione camera→punto (z lasciata invariata: e' gia' un punto valido sull'asse). Pubblica lo stesso centro corretto su **due topic, in due frame**: `cokecan_center` (ri-trasformato nel frame camera, confrontabile direttamente con `yolo/coke_can_position`) e `cokecan_center_base_footprint` (gia' in `base_footprint`, pronto per chi lo usa nel frame del robot, es. `pose_optimizer`). Pubblica anche un marker RViz (`cokecan_center_axis`, `visualization_msgs/Marker` di tipo `LINE_STRIP`, in `base_footprint` perche' deve essere davvero verticale) che disegna un segmento verticale attraverso il centro. |
| `detection_and_occlusion_handler.py` | `detection_and_occlusion_handler` | Evoluzione a se' stante di `rt_object_detection_node.py` (nodo ROS distinto, `detection_and_occlusion_handler_node`, per non collidere con quello sopra se lanciati insieme): stessa detection YOLO+depth, ma con una modalita' aggiuntiva selezionabile col parametro `use_occlusion_handling` (default `True`) — **robust**, che esclude i pixel contaminati da oggetti tracciati piu' vicini prima di stimare la depth del target, oppure **light** (comportamento del nodo base). In modalita' robust campiona anche fino a 3 punti sulla superficie visibile del target e li pubblica su `yolo/tracked_objects_points` (`tiago_vision_msgs/TrackedObjectsArray`), per un fit del cerchio/asse a valle — con marker RViz (`yolo/grasp_circle_points`). |

`rviz/complete_visualization.rviz` mostra questi output (immagine annotata,
point cloud di depth, robot model, punto di superficie, asse del centro) in
un'unica vista — ma per vederci qualcosa (robot, camera, tf) serve la
simulazione (o il robot reale) gia' avviata, altrimenti RViz non ha nulla da
disegnare.

**Un solo comando** per l'intero stack (simulazione + detection + center_computation + RViz):
```bash
ros2 launch detection_and_ranging full_stack.launch.py
# se la simulazione e' gia' avviata altrove:
ros2 launch detection_and_ranging full_stack.launch.py launch_simulation:=false
# senza RViz:
ros2 launch detection_and_ranging full_stack.launch.py rviz:=false
```
Include `vision_pipeline/launch/simulation.launch.py` per la simulazione
(nessuna duplicazione della configurazione del mondo/robot) e avvia
direttamente i nodi di detection, center computation e RViz.

`cokecan_center_base_footprint` (e la posa che ne deriva) e' l'input che
[`pose_optimizer`](../pose_optimizer) usera' come target per la
pianificazione IK (non ancora collegato: il target li' e' ancora fisso
hardcoded) — gia' nel frame giusto, nessuna trasformazione aggiuntiva
necessaria li'.
