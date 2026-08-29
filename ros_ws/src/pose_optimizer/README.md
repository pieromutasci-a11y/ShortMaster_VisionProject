# pose_optimizer

Riceve la posizione 3D degli oggetti rilevati da
[`detection_and_ranging`](../detection_and_ranging) (gia' in `base_footprint`)
e la manda a MoveIt (`move_group`) per pianificare una configurazione del
braccio che raggiunga la lattina, evitando pringles e biscotti come ostacoli.

**Stato: lavoro in corso.** Non ha ancora un proprio launch file / config
MoveIt (rimossi insieme al vecchio client basato su PickIK) — per ora si
presume un `move_group` gia' avviato altrove (es. lo stack standard di
`tiago_pro_moveit_config`), da chiarire come prossimo passo.

| File | Cosa fa |
|---|---|
| `pose_optimizer/optimizer.py` (`pose_optimizer_node`) | Client dell'action `/move_action`. Si iscrive a `cokecan_center_base_footprint` (target) e a `centers_all/<classe>_center_base_footprint` per `pringles can`/`biscuits pack` (ostacoli, aggiunti alla planning scene come cilindri con le dimensioni vere dei modelli Gazebo). Invece di affidarsi a un solver IK "intelligente" (PickIK), usa KDL (analitico) ma **campiona piu' angoli di presa (yaw) attorno all'oggetto** — sfruttando la simmetria assiale di un cilindro, per cui l'oggetto non ha un lato "giusto" da cui essere afferrato, ma il braccio si' — e tiene la soluzione i cui giunti restano piu' lontani dai limiti (`calcola_distanza_limiti`). Pubblica un marker RViz per ogni candidato provato e la configurazione finale scelta su `/joint_states`, per l'ispezione visiva. |

```bash
ros2 run pose_optimizer pose_optimizer_node
```

Il tavolo come ostacolo (`add_table_obstacle`, gia' presente ma disattivato)
e' un discorso a parte, non ancora affrontato.
