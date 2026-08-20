# detection_and_ranging

Nodo ROS2 che rileva l'oggetto target (lattina, pringles, biscotti) in
tempo reale con YOLOv8 e fonde la detection con l'immagine di depth
sincronizzata per stimarne la posizione 3D nel frame ottico della camera.

| Nodo | Eseguibile | Descrizione |
|---|---|---|
| `rt_object_detection_node.py` | `rt_object_detection` | Sottoscrive RGB + depth (sincronizzati), pubblica il frame annotato (`yolo/annotated_frame`) e la posizione 3D dell'oggetto target (`yolo/coke_can_position`, `geometry_msgs/PointStamped`). Usa il modello migliore dello sweep W&B (run `xl1874f6`, path hardcoded in cima al file). |

`rviz/complete_visualization.rviz` mostra questi output (immagine annotata,
point cloud di depth, robot model, marker di posizione) in un'unica vista:

```bash
ros2 run detection_and_ranging rt_object_detection
rviz2 -d install/detection_and_ranging/share/detection_and_ranging/rviz/complete_visualization.rviz
```

Questa posizione 3D e' l'input che [`pose_optimizer`](../pose_optimizer) usa
come target per la pianificazione IK.
