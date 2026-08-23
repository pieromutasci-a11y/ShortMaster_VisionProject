# tiago_vision_msgs

Messaggi custom per l'output multi-punto, multi-oggetto della pipeline di
visione.

| Messaggio | Campi | Descrizione |
|---|---|---|
| `TrackedObjectPoints` | `class_name`, `shape`, `points[<=3]` | Un oggetto riconosciuto: nome classe YOLO, forma geometrica nota (es. `cylinder`), e fino a 3 punti sulla superficie visibile (frame ottico della camera) — usati a valle per fittare un cerchio e recuperare l'asse centrale di un oggetto cilindrico. |
| `TrackedObjectsArray` | `header`, `objects[]` | Tutti gli oggetti tracciati in un frame RGB-D, pubblicati insieme. L'oggetto target e' sempre il primo elemento, quando presente. |

Usato da [`occlusion_handler`](../occlusion_handler).
