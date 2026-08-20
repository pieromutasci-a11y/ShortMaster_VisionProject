# pose_optimizer

Prende una posa target (per ora impostata manualmente in
`pose_optimizer_client.py`; in futuro verra' dalla posizione 3D pubblicata da
[`detection_and_ranging`](../detection_and_ranging)) e la manda a MoveIt
(`move_group`) per pianificare una configurazione del braccio che la
raggiunga, usando il plugin di cinematica inversa **PickIK** (ottimizzazione
numerica della posa, invece della sola soluzione analitica KDL) per il
braccio sinistro.

| File | Cosa fa |
|---|---|
| `pose_optimizer/pose_optimizer_client.py` (`pose_optimizer_client`) | Client dell'action `/move_action`: calcola l'orientamento del tool per una presa dall'alto, manda la posa target a MoveIt, pubblica un marker RViz e la configurazione finale su `/joint_states`. |
| `launch/move_group_pickik.launch.py` | Copia locale (modificabile) del `move_group.launch.py` di `tiago_pro_moveit_config`, per lanciare `move_group` con la configurazione PickIK. |
| `config/kinematics_pickik.yaml` | Solver IK per gruppo: `pick_ik/PickIkPlugin` per `arm_left`, KDL per gli altri gruppi. |
| `config/tiago_pro_for_moveit.urdf` | URDF del TIAGo Pro usato per la pianificazione. |

```bash
ros2 launch pose_optimizer move_group_pickik.launch.py
ros2 run pose_optimizer pose_optimizer_client
```
