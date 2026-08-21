# detection_and_ranging

Nodo ROS2 che rileva l'oggetto target (lattina, pringles, biscotti) in
tempo reale con YOLOv8 e fonde la detection con l'immagine di depth
sincronizzata per stimarne la posizione 3D nel frame ottico della camera.

| Nodo | Eseguibile | Descrizione |
|---|---|---|
| `rt_object_detection_node.py` | `rt_object_detection` | Sottoscrive RGB + depth (sincronizzati), pubblica il frame annotato (`yolo/annotated_frame`) e la posizione 3D dell'oggetto target (`yolo/coke_can_position`, `geometry_msgs/PointStamped`) — il punto misurato sta sulla superficie tangente della lattina (verso la camera), non sul suo asse centrale. Usa il modello migliore dello sweep W&B (run `xl1874f6`, path hardcoded in cima al file). |
| `center_computation.py` | `center_computation` | Sottoscrive `yolo/coke_can_position` e ne corregge x,y spostando il punto di un raggio noto (`CAN_RADIUS`, prior sull'oggetto) lungo la componente orizzontale del raggio ottico camera→punto, per stimare il centro dell'asse verticale del cilindro (z lasciata invariata: e' gia' un punto valido sull'asse, nessuna correzione necessaria). Pubblica il centro corretto su `cokecan_center` e un marker RViz (`cokecan_center_axis`, `visualization_msgs/Marker` di tipo `LINE_STRIP`) che disegna un segmento verticale attraverso il centro, in `base_footprint` (serve TF per essere davvero verticale — il frame camera e' inclinato). |

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

`cokecan_center` (e la posa che ne deriva) e' l'input che
[`pose_optimizer`](../pose_optimizer) usera' come target per la
pianificazione IK (non ancora collegato: il target li' e' ancora fisso
hardcoded).
