# pose_optimizer

Riceve la posizione 3D di lattina (target), pringles e biscotti (ostacoli)
da uno stack di detection (`detection_and_ranging_occlusion_free` o
`detection_and_ranging_occlusion_handler`, già in `base_footprint`), la
manda a MoveIt (`move_group`) per pianificare **ed eseguire per davvero**
una configurazione del braccio sinistro che raggiunga la lattina, evitando
pringles, biscotti e il tavolo come ostacoli.

Non ha un proprio launch file: usa `move_group`, già avviato da
`vision_pipeline/simulation.launch.py` (`moveit:=true`, il default) o da uno
dei due `full_stack_*.launch.py` che la includono.

## Come funziona

Entrambi i nodi del pacchetto seguono lo stesso schema:

1. **Input**: sottoscrivono `centers_all/<classe>_center_base_footprint` per
   `coke can` (target), `pringles can` e `biscuits pack` (ostacoli) — quello
   che pubblica uno dei due stack di detection. Resta anche una
   sottoscrizione legacy a `cokecan_center_base_footprint` (stack
   single-object, non più usato nel workflow corrente), solo per log.
2. **Ostacoli**: pringles e biscotti vengono aggiunti come `CollisionObject`
   cilindrici (dimensioni vere dai modelli Gazebo, `OBSTACLE_DIMENSIONS`).
   Anche la lattina target è aggiunta come ostacolo vero (`COKE_DIMENSIONS`),
   così MoveIt scarta le pose in cui le dita del gripper la toccherebbero
   prima di arrivare al punto di presa.
3. **Tavolo**: aggiunto come ostacolo (`add_table_obstacle`), ma la sua posa
   non viene dalla detection diretta (bordi larghi, gambe sottili, poca
   texture — difficile da riconoscere). È invece nota a priori nel mondo
   Gazebo (`TABLE_WORLD_POSITION_XY`), e viene ricostruita in `base_footprint`
   così: le posizioni note nel mondo degli oggetti tracciati
   (`KNOWN_OBJECT_WORLD_XY`, **deve restare sincronizzata con
   `poliBaMaster.world`**) vengono confrontate con le posizioni di quegli
   stessi oggetti viste ORA dal robot; la trasformazione rigida (rotazione +
   traslazione) che allinea le due è calcolata con un fit di Procrustes
   ortogonale (`_fit_rigid_transform_2d`, SVD), e applicata alla posizione
   nota del tavolo. Servono almeno 2 oggetti tracciati (un solo punto non
   determina la rotazione); sotto quella soglia il tavolo non viene
   aggiunto. Il box del tavolo è ancorato a `base_footprint`: resta valido
   finché il robot non si sposta (se lo si sposta, va rilanciato il nodo).
4. **Sweep**: campiona `n_campioni` (20) pose di presa su una circonferenza
   di raggio 0.05m attorno al target (`costruisci_target`), con orientamento
   del gripper fisso (approccio dall'alto, dita orizzontali) e yaw variabile
   — sfrutta la simmetria assiale di un cilindro, che non ha un lato
   "giusto" da cui essere afferrato. Per ognuna verifica l'IK e l'assenza di
   collisioni con MoveIt (solo pianificazione, partenza dallo stato vero
   attuale del braccio).
5. **Scelta**: tra i candidati riusciti, sceglie quello con il punteggio
   migliore secondo il proprio criterio (vedi sotto), poi lo **esegue per
   davvero** con un'unica chiamata finale.

I due nodi si distinguono solo per il criterio di scelta al punto 5:

| Nodo | Eseguibile | Criterio |
|---|---|---|
| `joint_margin_optimizer.py` | `joint_margin_optimizer` | Giunti più lontani possibile dai propri limiti (`calcola_distanza_limiti`: somma dei quadrati dello scarto normalizzato dal centro del range, negata — 0 = tutti i giunti centrati). |
| `manipulability_optimizer.py` | `manipulability_optimizer` | Indice di manipolabilità di Yoshikawa `w(q) = sqrt(det(J(q)·J(q)ᵀ))` (0 = singolarità cinematica, più alto = configurazione meglio condizionata) — criterio complementare al primo: una configurazione può avere i giunti centrati ed essere comunque vicina a una singolarità (es. gomito quasi disteso). Lo Jacobiano è calcolato con **PyKDL**, costruendo a mano la catena cinematica `torso_lift_link → gripper_left_grasping_link` dall'URDF pubblicato su `/robot_description` (`costruisci_catena_kdl` — `kdl_parser_py` non esiste per ROS2 Humble). |

## Topic e interfacce

- **Sottoscrive**: `centers_all/coke_can_center_base_footprint`,
  `centers_all/pringles_can_center_base_footprint`,
  `centers_all/biscuits_pack_center_base_footprint` (`geometry_msgs/PointStamped`).
- **Pubblica**: `/planning_scene` (`moveit_msgs/PlanningScene`, gli ostacoli),
  `/grasp_candidates_marker` (`visualization_msgs/MarkerArray`, un marker
  per candidato: grigio = non ancora provato, verde = riuscito, rosso =
  fallito, oro = scelto), `/obstacles_debug_marker` (`visualization_msgs/Marker`,
  gli stessi ostacoli inviati a MoveIt, ma su un topic separato — la
  `PlanningScene` in RViz li mischierebbe con l'octomap della percezione).
- **Action client**: `/move_action` (`moveit_msgs/action/MoveGroup`), gruppo
  di planning `arm_left`, link `gripper_left_grasping_link`.

## Uso

```bash
ros2 launch detection_and_ranging_occlusion_free full_stack_all.launch.py   # simulazione + MoveIt + detection + ostacoli
ros2 run pose_optimizer joint_margin_optimizer                              # calcola ed esegue la presa (criterio: limiti di giunto)
# oppure:
ros2 run pose_optimizer manipulability_optimizer                            # calcola ed esegue la presa (criterio: manipolabilità)
```

Funziona identicamente con `detection_and_ranging_occlusion_handler` al
posto di `detection_and_ranging_occlusion_free`, dato che pubblica sugli
stessi topic.

Non chiude il gripper: porta solo il braccio nella posa di presa scelta, la
chiusura è un passo separato non affrontato da questo pacchetto.

## Output

Ogni nodo salva i propri grafici di debug nella cartella sorgente del
pacchetto (non nella copia installata, che verrebbe rigenerata a ogni build):

- `joint_margin_results/`: `joint_margin_candidates.png` (i 7 angoli di
  giunto di ogni candidato riuscito, confrontati con i limiti veri) e
  `optimal_end_effector_pose.png` (posa vera dell'end effector dopo
  l'esecuzione, letta da TF).
- `manipulability_results/`: `manipulability_candidates.png` (manipolabilità
  vs yaw per candidato), `manipulability_joint_values.png` (stessi 7 grafici
  per giunto, qui solo come riferimento visivo), `manipulability_ellipsoid.png`
  (ellissoide di manipolabilità del candidato scelto, parte lineare dello
  Jacobiano) e `optimal_end_effector_pose.png`.
