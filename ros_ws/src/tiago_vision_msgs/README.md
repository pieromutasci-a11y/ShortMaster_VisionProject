# tiago_vision_msgs

Messaggi custom per l'output multi-punto, multi-oggetto della pipeline di
visione.

| Messaggio | Campi | Descrizione |
|---|---|---|
| `TrackedObjectPoints` | `class_name` (string), `shape` (string), `points` (`geometry_msgs/Point[<=3]`) | Un oggetto riconosciuto: nome classe YOLO (es. `'coke can'`), forma geometrica nota (es. `'cylinder'`), e fino a 3 punti sulla superficie visibile (frame ottico della camera) — usati a valle per fittare un cerchio e recuperare l'asse centrale di un oggetto cilindrico. |
| `TrackedObjectsArray` | `header` (`std_msgs/Header`), `objects` (`TrackedObjectPoints[]`) | Tutti gli oggetti tracciati in un frame RGB-D, pubblicati insieme. L'oggetto target è sempre il primo elemento, quando presente. |

Usati da `rt_object_detection_node_occlusion_handler.py` in
[`detection_and_ranging_occlusion_handler`](../detection_and_ranging_occlusion_handler)
per pubblicare i punti di superficie campionati per ogni oggetto tracciato
(topic `yolo/tracked_objects_points` e `yolo/tracked_objects_anchor_point`),
e da `center_computation_occlusion_handler.py` nello stesso pacchetto per
riceverli e fittare la circonferenza che ne ricava il centro.
