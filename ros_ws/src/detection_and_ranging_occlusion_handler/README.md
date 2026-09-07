# detection_and_ranging_occlusion_handler

Nodo ROS2 che rileva gli oggetti tracciati (lattina, pringles, biscotti) in
tempo reale con YOLOv8 e fonde la detection con l'immagine di depth
sincronizzata per stimarne la posizione 3D, **gestendo esplicitamente le
occlusioni tra oggetti tracciati** — a differenza di
[`detection_and_ranging_occlusion_free`](../detection_and_ranging_occlusion_free),
che non lo fa (da qui il nome di entrambi i pacchetti).

**Stato: lavoro in corso.** Manca ancora un launch file dedicato (per ora i
due nodi vanno lanciati a mano, insieme alla simulazione avviata da
`vision_pipeline`).

| Nodo | Eseguibile | Descrizione |
|---|---|---|
| `rt_object_detection_node_occlusion_handler.py` | `rt_object_detection_occlusion_handler` | Sottoscrive RGB + depth (sincronizzati), pubblica il frame annotato (`yolo/annotated_frame`) e la posizione 3D di ogni oggetto tracciato, nel frame ottico della camera. Due modalita', selezionabili col parametro ROS2 `use_occlusion_handling` (default `True`): **robust**, che per ogni coppia di bounding box sovrapposte esclude i pixel ambigui prima di stimare la depth di ciascun oggetto (cosi' un oggetto occludente e uno occluso non si contaminano a vicenda la stima), ordina gli oggetti per depth pulita, e per l'oggetto target esclude solo i pixel condivisi con oggetti genuinamente piu' vicini; oppure **light** (comportamento semplice, centro bounding box + depth, nessuna gestione occlusioni — adatta solo se gli oggetti non si occludono mai in scena). In modalita' robust campiona anche fino a 3 punti sulla superficie visibile di ciascun oggetto (target compreso) e li pubblica su `yolo/tracked_objects_points` (`tiago_vision_msgs/TrackedObjectsArray`), per un fit del cerchio/asse a valle — con marker RViz dedicati (`yolo/grasp_circle_points`, rosso, e `yolo/anchor_points_marker`, blu, quest'ultimo pubblicato in entrambe le modalita'). |
| `center_computation_occlusion_handler.py` | `center_computation_occlusion_handler` | Sottoscrive `yolo/tracked_objects_points`. Per ciascun oggetto (coke can, pringles can, biscuits pack — nessun raggio noto necessario, a differenza di `center_computation_all.py`), trasforma i suoi 3 punti di superficie in `base_footprint` via TF e ne fitta la circonferenza passante nel piano orizzontale (soluzione esatta per 3 punti, formula chiusa — non un fit ai minimi quadrati): il raggio non e' piu' assunto, e' misurato dai punti stessi. Se i 3 punti sono troppo allineati/coincidenti (fit numericamente instabile — capita quasi sempre per `biscuits pack`, una scatola non un cilindro: sulla sua faccia piatta i 3 punti di una riga sono quasi in linea, non su un arco) il frame viene scartato, non pubblicato — stessa filosofia "meglio niente che un dato inventato" usata altrove nel progetto. Pubblica sugli **stessi topic** di `center_computation_all.py` (`centers_all/<classe>_center`, `centers_all/<classe>_center_base_footprint`, `centers_all/<classe>_axis`): i due stack sono intercambiabili, `pose_optimizer` non richiede nessuna modifica per usare l'uno o l'altro. |

`MODEL_WEIGHTS_PATH` in cima al file di detection e' ancora hardcoded a un
path assoluto di una macchina specifica — da parametrizzare (vedi TODO nel
file).
