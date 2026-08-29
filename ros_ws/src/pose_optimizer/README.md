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
| `pose_optimizer/optimizer.py` (`pose_optimizer_node`) | Client dell'action `/move_action`. Target e ostacoli vengono letti tutti da `centers_all/<classe>_center_base_footprint` (stack multi-oggetto: `rt_object_detection_node_all.py` + `center_computation_all.py`, lanciati da `full_stack_all.launch.py` insieme al tavolo/ostacoli) — `coke can` come target, `pringles can`/`biscuits pack` come ostacoli cilindrici (dimensioni vere dai modelli Gazebo). Resta anche una sottoscrizione a `cokecan_center_base_footprint` (stack single-object, solo per log/compatibilita'), ma non e' piu' la fonte del target: nel workflow reale (stack `_all`, con gli ostacoli) quel topic non viene mai pubblicato. Il tavolo e' invece **noto a priori** (`add_table_obstacle`, non dalla detection — difficile da ricostruire dalla camera): un box pieno da terra al piano, ancorato al frame `map` (fisso nel mondo) invece che a `base_footprint`, perche' il robot si muove attorno al tavolo — un offset fisso rispetto al robot sarebbe corretto solo nell'istante in cui e' stato misurato. La posa del tavolo rispetto al robot allo spawn e' nota per certo (da due pose vere: il world file e `robot_spawn.launch.py`) — ma convertirla a mano in coordinate `map` assumendo una convenzione per l'origine di `map` si e' rivelato **sbagliato** (verificato in RViz: il box finiva praticamente sotto al robot, causa della collisione reale col tavolo). Ora quella conversione la fa TF per davvero: all'avvio, prima che il robot si muova dallo spawn, il nodo legge la trasformazione VERA `base_footprint -> map` e la applica alla posa nota — corretto qualunque sia la reale convenzione di `map`, perche' non viene piu' assunta. Resta pero' un'imprecisione pratica: la stima di `map` da parte di `slam_toolbox` non e' ancora ben assestata cosi' presto dopo l'avvio, e la posa calcolata del tavolo e' variata anche di ~17cm da un run all'altro rispetto al valore vero — non un errore di formula, ma rumore intrinseco della stima SLAM che aspettare di piu' non elimina in modo affidabile. Il box del tavolo viene quindi reso volutamente **piu' grande** del tavolo vero (`TABLE_SAFETY_MARGIN`, +0.4m per lato orizzontale) per assorbire questa incertezza nota. Invece di affidarsi a un solver IK "intelligente" (PickIK), usa KDL (analitico) ma **campiona piu' angoli di presa (yaw) attorno all'oggetto** — sfruttando la simmetria assiale di un cilindro, per cui l'oggetto non ha un lato "giusto" da cui essere afferrato, ma il braccio si' — e tiene, tra i candidati riusciti, quello i cui giunti restano piu' lontani dai limiti (`calcola_distanza_limiti`). Lo sweep è **solo pianificazione** (`plan_only=True`, ma partenza dallo stato **vero** attuale del braccio — il braccio non si muove comunque, perché non viene eseguito, ma la fattibilità/costo di ogni candidato è valutata da dove il robot si trova davvero, non da una posa arbitraria); trovato il migliore, lo **esegue per davvero** con un'unica chiamata finale (`plan_only=False`, stessa partenza dallo stato vero del braccio) — il braccio si muove in Gazebo/RViz. Tutti i punti candidati della circonferenza vengono disegnati insieme su `/grasp_candidates_marker` (`MarkerArray`): grigio finché non ancora provati, verde se l'IK ha successo, rosso se fallisce (IK impossibile o collisione con gli ostacoli), oro per il candidato scelto. Non pubblica un marker a parte per il centro della lattina: lo fa già `detection_and_ranging` (`centers_all/coke_can_axis`, asse verticale reale) — ripeterlo qui sarebbe ridondante. |

```bash
ros2 launch detection_and_ranging full_stack_all.launch.py   # simulazione + MoveIt + detection multi-oggetto + ostacoli
ros2 run pose_optimizer pose_optimizer_node               # calcola ed esegue la presa
```

Non chiude il gripper — porta solo il braccio nella posa di presa scelta,
la chiusura è un passo separato non ancora affrontato.
