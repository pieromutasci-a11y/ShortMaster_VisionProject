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
| `pose_optimizer/optimizer.py` (`pose_optimizer_node`) | Client dell'action `/move_action`. Target e ostacoli vengono letti tutti da `centers_all/<classe>_center_base_footprint` (stack multi-oggetto: `rt_object_detection_node_all.py` + `center_computation_all.py`, lanciati da `full_stack_all.launch.py` insieme al tavolo/ostacoli) — `coke can` come target, `pringles can`/`biscuits pack` come ostacoli cilindrici (dimensioni vere dai modelli Gazebo). Resta anche una sottoscrizione a `cokecan_center_base_footprint` (stack single-object, solo per log/compatibilita'), ma non e' piu' la fonte del target: nel workflow reale (stack `_all`, con gli ostacoli) quel topic non viene mai pubblicato. Il tavolo e' invece **noto a priori** (`add_table_obstacle`, non dalla detection — difficile da ricostruire dalla camera): un box pieno da terra al piano, con un margine di sicurezza (`TABLE_SAFETY_MARGIN`, +0.4m per lato orizzontale), ancorato a **`odom`**. Storia dei tentativi, per chi si ritrova a rivedere questa scelta in futuro:
1. Coordinate del world file usate direttamente come coordinate `map`, assumendo che `map` nascesse coincidente con lo spawn del robot — sbagliato, verificato in RViz (il box finiva sotto al robot).
2. `map` letto via TF vera invece di assunto — corretto in linea di principio, ma la stima di `map` da parte di `slam_toolbox` non e' assestata cosi' presto dopo l'avvio: la posa calcolata del tavolo e' variata da un run all'altro fino a 37cm rispetto al valore vero (non un bug di formula, rumore della stima SLAM che aspettare di piu' non elimina in modo affidabile).
3. Ancorato a `base_footprint` — risolve l'errore, ma smette di valere nel momento in cui la base viene comandata a muoversi, che e' un requisito reale del progetto (il robot deve potersi spostare), non un'ipotesi remota. Scartato.
4. **Soluzione adottata**: `odom` — fisso nel mondo come `map` (quindi resta valido quando il robot si muove), ma disponibile da subito, senza bisogno di convergenza SLAM: e' pura integrazione odometrica dalle ruote, non una stima che deve "assestarsi". L'unico costo e' un lento drift nel tempo/nella distanza percorsa (normale per qualunque odometria a ruote), un problema diverso e molto piu' piccolo del rumore di convergenza iniziale visto su `map`. Stessa logica di prima (la posa del tavolo e' nota per certo rispetto al robot allo spawn, da due pose vere: il world file e `robot_spawn.launch.py`) — solo che ora si converte in `odom` via TF vera invece che in `map`. Invece di affidarsi a un solver IK "intelligente" (PickIK), usa KDL (analitico) ma **campiona piu' angoli di presa (yaw) attorno all'oggetto** — sfruttando la simmetria assiale di un cilindro, per cui l'oggetto non ha un lato "giusto" da cui essere afferrato, ma il braccio si' — e tiene, tra i candidati riusciti, quello i cui giunti restano piu' lontani dai limiti (`calcola_distanza_limiti`). Lo sweep è **solo pianificazione** (`plan_only=True`, ma partenza dallo stato **vero** attuale del braccio — il braccio non si muove comunque, perché non viene eseguito, ma la fattibilità/costo di ogni candidato è valutata da dove il robot si trova davvero, non da una posa arbitraria); trovato il migliore, lo **esegue per davvero** con un'unica chiamata finale (`plan_only=False`, stessa partenza dallo stato vero del braccio) — il braccio si muove in Gazebo/RViz. Tutti i punti candidati della circonferenza vengono disegnati insieme su `/grasp_candidates_marker` (`MarkerArray`): grigio finché non ancora provati, verde se l'IK ha successo, rosso se fallisce (IK impossibile o collisione con gli ostacoli), oro per il candidato scelto. Non pubblica un marker a parte per il centro della lattina: lo fa già `detection_and_ranging` (`centers_all/coke_can_axis`, asse verticale reale) — ripeterlo qui sarebbe ridondante. |

```bash
ros2 launch detection_and_ranging full_stack_all.launch.py   # simulazione + MoveIt + detection multi-oggetto + ostacoli
ros2 run pose_optimizer pose_optimizer_node               # calcola ed esegue la presa
```

Non chiude il gripper — porta solo il braccio nella posa di presa scelta,
la chiusura è un passo separato non ancora affrontato.
