# tiago_vision_msgs

Messaggi custom per l'output multi-punto, multi-oggetto della pipeline di
visione.

| Messaggio | Campi | Descrizione |
|---|---|---|
| `TrackedObjectPoints` | `class_name`, `shape`, `points[<=3]` | Un oggetto riconosciuto: nome classe YOLO, forma geometrica nota (es. `cylinder`), e fino a 3 punti sulla superficie visibile (frame ottico della camera) — usati a valle per fittare un cerchio e recuperare l'asse centrale di un oggetto cilindrico. |
| `TrackedObjectsArray` | `header`, `objects[]` | Tutti gli oggetti tracciati in un frame RGB-D, pubblicati insieme. L'oggetto target e' sempre il primo elemento, quando presente. |

Usati da `rt_object_detection_node_occlusion_handler.py` in
[`detection_and_ranging_occlusion_handler`](../detection_and_ranging_occlusion_handler),
per la gestione robusta delle occlusioni tra oggetti tracciati (fit di
cerchio/asse su piu' punti campionati). Un primo tentativo in questa
direzione, `detection_and_occlusion_handler.py`, era stato rimosso dal
pacchetto `detection_and_ranging_occlusion_free` (all'epoca ancora chiamato
`detection_and_ranging`) — l'idea e' poi tornata come pacchetto a se
stante.
