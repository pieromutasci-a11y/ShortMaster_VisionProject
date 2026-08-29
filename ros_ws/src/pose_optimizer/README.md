# pose_optimizer

Riceve la posizione 3D degli oggetti rilevati da
[`detection_and_ranging`](../detection_and_ranging) (gia' in `base_footprint`)
e la manda a MoveIt (`move_group`) per pianificare **ed eseguire per
davvero** una configurazione del braccio che raggiunga la lattina, evitando
pringles, biscotti e il tavolo come ostacoli.

**Stato: lavoro in corso.** Non ha ancora un proprio launch file / config
MoveIt (rimossi insieme al vecchio client basato su PickIK) — usa
`move_group` avviato da `vision_pipeline/simulation.launch.py`
(`moveit:=True`, incluso anche da `detection_and_ranging/full_stack.launch.py`).

| File | Cosa fa |
|---|---|
| `pose_optimizer/optimizer.py` (`pose_optimizer_node`) | Client dell'action `/move_action`. Target e ostacoli vengono letti tutti da `centers_all/<classe>_center_base_footprint` (stack multi-oggetto: `rt_object_detection_node_all.py` + `center_computation_all.py`, lanciati da `full_stack_all.launch.py` insieme al tavolo/ostacoli) — `coke can` come target, `pringles can`/`biscuits pack` come ostacoli cilindrici (dimensioni vere dai modelli Gazebo). Resta anche una sottoscrizione a `cokecan_center_base_footprint` (stack single-object, solo per log/compatibilita'), ma non e' piu' la fonte del target: nel workflow reale (stack `_all`, con gli ostacoli) quel topic non viene mai pubblicato. Il tavolo e' invece **noto a priori** (`add_table_obstacle`, non dalla detection — difficile da ricostruire dalla camera): un box pieno da terra al piano, con la posa vera del mondo Gazebo, ancorato al frame `map` (fisso nel mondo) invece che a `base_footprint`, perche' il robot si muove attorno al tavolo — un offset fisso rispetto al robot sarebbe corretto solo nell'istante in cui e' stato misurato. Invece di affidarsi a un solver IK "intelligente" (PickIK), usa KDL (analitico) ma **campiona piu' angoli di presa (yaw) attorno all'oggetto** — sfruttando la simmetria assiale di un cilindro, per cui l'oggetto non ha un lato "giusto" da cui essere afferrato, ma il braccio si' — e tiene, tra i candidati riusciti, quello i cui giunti restano piu' lontani dai limiti (`calcola_distanza_limiti`). Lo sweep è **solo pianificazione** (`plan_only=True`, ma partenza dallo stato **vero** attuale del braccio — il braccio non si muove comunque, perché non viene eseguito, ma la fattibilità/costo di ogni candidato è valutata da dove il robot si trova davvero, non da una posa arbitraria); trovato il migliore, lo **esegue per davvero** con un'unica chiamata finale (`plan_only=False`, stessa partenza dallo stato vero del braccio) — il braccio si muove in Gazebo/RViz. Tutti i punti candidati della circonferenza vengono disegnati insieme su `/grasp_candidates_marker` (`MarkerArray`): grigio finché non ancora provati, verde se l'IK ha successo, rosso se fallisce (IK impossibile o collisione con gli ostacoli), oro per il candidato scelto. |

```bash
ros2 launch detection_and_ranging full_stack_all.launch.py   # simulazione + MoveIt + detection multi-oggetto + ostacoli
ros2 run pose_optimizer pose_optimizer_node               # calcola ed esegue la presa
```

Non chiude il gripper — porta solo il braccio nella posa di presa scelta,
la chiusura è un passo separato non ancora affrontato.
