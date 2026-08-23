# occlusion_handler

Nodo ROS2 di detection+ranging con gestione delle occlusioni tra gli oggetti
tracciati (lattina, pringles, biscotti) — evoluzione indipendente di
[`detection_and_ranging`](../detection_and_ranging), tenuta come pacchetto a
se' per non toccare quello gia' esistente.

| Nodo | Eseguibile | Descrizione |
|---|---|---|
| `detection_and_occlusion_handler.py` | `detection_and_occlusion_handler` | Rileva l'oggetto target con YOLOv8 e ne stima la posizione fondendola con la depth. Due modalita', selezionabili col parametro ROS `use_occlusion_handling` (default `True`): **robust** — tiene conto delle occlusioni tra gli oggetti tracciati e dei pixel di sfondo che finiscono nel bounding box; **light** — legge direttamente il centro del bounding box, senza gestione occlusioni (adatta solo se il target non e' mai occluso). |

In modalita' robust pubblica anche `yolo/tracked_objects_points`
(`tiago_vision_msgs/TrackedObjectsArray`): fino a 3 punti sulla superficie
visibile del target, campionati lungo una riga a quota costante, pensati per
un fit di un cerchio a valle e recuperare l'asse centrale dell'oggetto
cilindrico — e un marker RViz (`yolo/grasp_circle_points`) che li visualizza.

```bash
ros2 run occlusion_handler detection_and_occlusion_handler
ros2 run occlusion_handler detection_and_occlusion_handler --ros-args -p use_occlusion_handling:=false
```
