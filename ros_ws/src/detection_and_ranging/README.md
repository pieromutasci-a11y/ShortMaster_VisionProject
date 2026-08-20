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

**Terminale 1 — simulazione:**
```bash
ros2 launch detection_and_ranging simulation.launch.py
# (include vision_pipeline/launch/simulation.launch.py, nessuna duplicazione)
```

**Terminale 2 — detection + RViz**, una volta che la simulazione e' su:
```bash
ros2 launch detection_and_ranging rt_object_detection.launch.py
# oppure senza RViz:
ros2 launch detection_and_ranging rt_object_detection.launch.py rviz:=false
```

I due `ros2 launch` sono processi indipendenti che comunicano sulla stessa
rete ROS: nessun problema a tenerli su in due terminali separati.

In alternativa, nodo e RViz separatamente:

```bash
ros2 run detection_and_ranging rt_object_detection
rviz2 -d install/detection_and_ranging/share/detection_and_ranging/rviz/complete_visualization.rviz
```

Questa posizione 3D e' l'input che [`pose_optimizer`](../pose_optimizer) usa
come target per la pianificazione IK.
