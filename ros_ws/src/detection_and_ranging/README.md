# detection_and_ranging

Nodo ROS2 che rileva l'oggetto target (lattina, pringles, biscotti) in
tempo reale con YOLOv8 e fonde la detection con l'immagine di depth
sincronizzata per stimarne la posizione 3D nel frame ottico della camera.

| Nodo | Eseguibile | Descrizione |
|---|---|---|
| `rt_object_detection_node.py` | `rt_object_detection` | Sottoscrive RGB + depth (sincronizzati), pubblica il frame annotato (`yolo/annotated_frame`) e la posizione 3D dell'oggetto target (`yolo/coke_can_position`, `geometry_msgs/PointStamped`). Usa il modello migliore dello sweep W&B (run `xl1874f6`, path hardcoded in cima al file). |

`rviz/complete_visualization.rviz` mostra questi output (immagine annotata,
point cloud di depth, robot model, marker di posizione) in un'unica vista —
ma per vederci qualcosa (robot, camera, tf) serve la simulazione (o il
robot reale) gia' avviata, altrimenti RViz non ha nulla da disegnare.

**Un solo comando** per l'intero stack (simulazione + detection + RViz):
```bash
ros2 launch detection_and_ranging full_stack.launch.py
# se la simulazione e' gia' avviata altrove:
ros2 launch detection_and_ranging full_stack.launch.py launch_simulation:=false
# senza RViz:
ros2 launch detection_and_ranging full_stack.launch.py rviz:=false
```
Include `vision_pipeline/launch/simulation.launch.py` per la simulazione
(nessuna duplicazione della configurazione del mondo/robot) e avvia
direttamente il nodo di detection e RViz.

Questa posizione 3D e' l'input che [`pose_optimizer`](../pose_optimizer) usa
come target per la pianificazione IK.
