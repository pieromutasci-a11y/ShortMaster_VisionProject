import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint,
    CollisionObject, PlanningScene, RobotState
)
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R
import numpy as np


# Topic del centro dell'oggetto (gia' in base_footprint, pubblicato da
# detection_and_ranging -- vedi center_computation.py/center_computation_all.py).
# Nessuna trasformazione TF necessaria qui: e' il frame in cui lavora gia'
# tutto questo nodo (target_pose.header.frame_id = "base_footprint").

# Pipeline single-object: solo la lattina, un'istanza per frame.
COKE_CENTER_TOPIC = 'cokecan_center_base_footprint'

# Pipeline multi-oggetto: un topic per classe (stesso schema di nomi di
# center_computation_all.py -- vedi topic_slug() li').
TRACKED_CLASSES = ['coke can', 'pringles can', 'biscuits pack', 'dinner table']

# Classi trattate come OSTACOLI (non target) nella scena di grasping. Il
# tavolo (dinner table) e' un discorso a parte, non e' qui.
OBSTACLE_CLASSES = ['pringles can', 'biscuits pack']

# Dimensioni reali (raggio, altezza in metri) dei modelli Gazebo -- vedi
# pal_gazebo_worlds/models/<nome>/model.sdf (<cylinder><radius>/<length>).
# Stesso principio del raggio in center_computation_all.py: e' un prior
# sull'OGGETTO, verificato sul modello vero, non un numero a caso.
OBSTACLE_DIMENSIONS = {
    'pringles can':  (0.04, 0.235),
    'biscuits pack': (0.029, 0.15),
}

# --- Tavolo: nota a priori, non dalla detection (difficile da ricostruire
# dalla camera -- bordi larghi, gambe sottili, poca texture) ---
#
# IMPORTANTE: il tavolo e' fisso nel MONDO, non rispetto al robot -- che
# invece si muove (e' letteralmente cosa fa orbit_around_table_node.py). Va
# quindi ancorato a un frame fisso nel mondo (TABLE_FRAME = 'map', presente
# perche' la simulazione parte con slam:=True/navigation:=True), non a
# base_footprint: un offset fisso in base_footprint sarebbe corretto solo
# nell'istante in cui e' stato misurato, e sbagliato appena il robot si
# sposta -- MoveIt calcola da solo, via TF, dove sta il tavolo rispetto al
# braccio in ogni momento, qualunque sia la posizione del robot.
TABLE_FRAME = 'map'

# Posa vera del tavolo nel mondo Gazebo (pal_gazebo_worlds/worlds/poliBaMaster.world,
# modello s3_table, nessuna rotazione).
TABLE_POSITION_XY = (5.0, 5.0)

# Geometria vera (pal_gazebo_worlds/models/table_0m8/table.sdf): piano
# 1.0 x 0.8 x 0.03 a z locale 0.8 -> superficie a z=0.815 (coerente col resto
# del progetto, es. CENTER_Z in orbit_around_table_node.py). Le 4 gambe
# sottili (cilindri r=0.02) non sono modellate qui: un unico box pieno da
# terra alla superficie e' piu' semplice e comunque MAI meno sicuro della
# realta' (il gruppo di planning e' solo il braccio, non la base -- non
# serve che passi "tra le gambe").
TABLE_SURFACE_TOP_Z = 0.815
TABLE_FOOTPRINT_XY = (1.0, 0.8)


def topic_slug(class_name):
    """'coke can' -> 'coke_can'. Deve restare identica a quella in center_computation_all.py."""
    return class_name.replace(' ', '_')


class MoveGroupClient(Node):
    def __init__(self):
        super().__init__('move_group_client_kdl')
        self._client = ActionClient(self, MoveGroup, '/move_action')
        self._joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self._scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self._candidates_marker_pub = self.create_publisher(
            MarkerArray, '/grasp_candidates_marker', 10
        )

        # --- Sottoscrizioni ai centri pubblicati da detection_and_ranging ---
        # La coca (topic single-object) e' il target; pringles/biscuits
        # (topic multi-oggetto) sono usati come ostacoli in main().

        self.latest_coke_center = None
        self.coke_center_sub = self.create_subscription(
            PointStamped, COKE_CENTER_TOPIC, self._coke_center_callback, 10
        )

        self.latest_centers_all = {}
        self._centers_all_subs = {}
        for class_name in TRACKED_CLASSES:
            slug = topic_slug(class_name)
            topic = f'centers_all/{slug}_center_base_footprint'
            # La lambda cattura class_name come default argument, altrimenti
            # tutte le callback userebbero l'ultimo valore del ciclo.
            self._centers_all_subs[class_name] = self.create_subscription(
                PointStamped, topic,
                lambda msg, name=class_name: self._centers_all_callback(msg, name),
                10,
            )

        self.get_logger().info(
            f"In ascolto su '{COKE_CENTER_TOPIC}' e su centers_all/* per: "
            f"{', '.join(TRACKED_CLASSES)}"
        )

    def _coke_center_callback(self, msg: PointStamped):
        self.latest_coke_center = msg
        self.get_logger().info(
            f"[{COKE_CENTER_TOPIC}] x={msg.point.x:.3f} y={msg.point.y:.3f} z={msg.point.z:.3f}",
            throttle_duration_sec=2.0,
        )

    def _centers_all_callback(self, msg: PointStamped, class_name: str):
        self.latest_centers_all[class_name] = msg
        slug = topic_slug(class_name)
        self.get_logger().info(
            f"[centers_all/{slug}_center_base_footprint] "
            f"x={msg.point.x:.3f} y={msg.point.y:.3f} z={msg.point.z:.3f}",
            throttle_duration_sec=2.0,
        )

    def add_table_obstacle(self, frame_id=TABLE_FRAME,
                            position=(*TABLE_POSITION_XY, TABLE_SURFACE_TOP_Z / 2.0),
                            dimensions=(*TABLE_FOOTPRINT_XY, TABLE_SURFACE_TOP_Z)):
        """
        Box pieno da terra (z=0) alla superficie del tavolo (z=TABLE_SURFACE_TOP_Z),
        ancorato a TABLE_FRAME (fisso nel mondo, non in base_footprint -- vedi
        commento sopra le costanti TABLE_*).
        """
        obj = CollisionObject()
        obj.header.frame_id = frame_id
        obj.id = "table"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(dimensions)

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = position
        pose.orientation.w = 1.0

        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.world.collision_objects.append(obj)
        scene.is_diff = True

        for _ in range(5):
            self._scene_pub.publish(scene)
            rclpy.spin_once(self, timeout_sec=0.3)

    def wait_for_topics(self, attribute_getters, timeout_sec=10.0):
        """
        Spinna il nodo finche' tutti gli attribute_getters (funzioni node -> valore)
        restituiscono un valore non-None, o scade il timeout. Serve ad aspettare
        che le detection siano arrivate prima di iniziare a pianificare, invece
        di partire subito con dati mancanti.
        """
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            if all(getter(self) is not None for getter in attribute_getters):
                return True
            rclpy.spin_once(self, timeout_sec=0.2)
        return all(getter(self) is not None for getter in attribute_getters)

    def add_object_obstacle(self, object_id, center: PointStamped, radius, height):
        """
        Aggiunge un ostacolo cilindrico verticale alla planning scene, nel
        punto (gia' corretto, gia' in base_footprint) pubblicato da
        detection_and_ranging. Un cilindro e' la forma giusta qui: sia
        pringles che biscuits pack sono modellati come cilindri anche in
        Gazebo (vedi OBSTACLE_DIMENSIONS), e il centro che riceviamo e' gia'
        sull'asse verticale dell'oggetto -- lo stesso ragionamento geometrico
        di center_computation.py/center_computation_all.py.
        """
        obj = CollisionObject()
        obj.header.frame_id = center.header.frame_id
        obj.id = object_id

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.CYLINDER
        # Ordine richiesto da SolidPrimitive per CYLINDER: [altezza, raggio].
        primitive.dimensions = [height, radius]

        pose = Pose()
        pose.position = center.point
        pose.orientation.w = 1.0

        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.world.collision_objects.append(obj)
        scene.is_diff = True

        for _ in range(5):
            self._scene_pub.publish(scene)
            rclpy.spin_once(self, timeout_sec=0.3)

    def publish_candidates_markers(self, candidati, frame_id="base_footprint"):
        """
        Disegna TUTTI i punti della circonferenza di presa in un colpo solo
        -- ogni candidato ha un suo id, quindi restano visibili tutti insieme.

        candidati: lista di dict {x, y, z, stato}, stato in
        {'in_attesa', 'ok', 'fallito', 'migliore'}.
        """
        colori = {
            'in_attesa': (0.6, 0.6, 0.6, 0.6),  # grigio: non ancora provato
            'ok':        (0.0, 1.0, 0.0, 0.8),  # verde: IK riuscita, no collisioni
            'fallito':   (1.0, 0.0, 0.0, 0.8),  # rosso: IK fallita o collisione
            'migliore':  (1.0, 0.84, 0.0, 1.0),  # oro: il candidato scelto
        }
        array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, c in enumerate(candidati):
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = "grasp_candidates"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = c['x']
            m.pose.position.y = c['y']
            m.pose.position.z = c['z']
            m.pose.orientation.w = 1.0
            is_migliore = c['stato'] == 'migliore'
            scala = 0.035 if is_migliore else 0.02
            m.scale.x = m.scale.y = m.scale.z = scala
            r, g, b, a = colori[c['stato']]
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
            array.markers.append(m)

        for _ in range(5):
            self._candidates_marker_pub.publish(array)
            rclpy.spin_once(self, timeout_sec=0.2)

    def send_goal(self, target_pose: PoseStamped, group_name="arm_left",
                  link_name="gripper_left_grasping_link", start_positions=None,
                  plan_only=True):
        """
        start_positions=None -> pianifica dalla posizione VERA del braccio
        (req.start_state.is_diff = True); altrimenti da quella indicata (usato
        durante lo sweep esplorativo, per non doversi affidare allo stato
        reale a ogni candidato).

        plan_only=True (default) -> solo pianificazione, il braccio non si
        muove. plan_only=False -> MoveIt esegue anche la traiettoria sul
        controller vero: usarlo SOLO con start_positions=None (partenza
        reale), altrimenti la traiettoria eseguita non partirebbe da dove il
        braccio si trova davvero.
        """
        self._client.wait_for_server()

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = group_name
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0

        if start_positions is not None:
            start_state = RobotState()
            start_state.joint_state.name = [
                'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint',
                'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint',
                'arm_left_7_joint'
            ]
            start_state.joint_state.position = start_positions
            req.start_state = start_state
        else:
            req.start_state.is_diff = True

        pc = PositionConstraint()
        pc.header.frame_id = target_pose.header.frame_id
        pc.link_name = link_name
        primitive = SolidPrimitive()   
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.005]  # raggio della sfera di tolleranza
        pc.constraint_region.primitives.append(primitive)
        pc.constraint_region.primitive_poses.append(target_pose.pose)
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = target_pose.header.frame_id
        oc.link_name = link_name
        oc.orientation = target_pose.pose.orientation
        oc.absolute_x_axis_tolerance = 0.05
        oc.absolute_y_axis_tolerance = 0.05
        oc.absolute_z_axis_tolerance = 0.05
        oc.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)
        req.goal_constraints.append(constraints)

        goal_msg.request = req
        goal_msg.planning_options.plan_only = plan_only

        future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal rifiutato")
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result()

    def publish_joint_state(self, joint_names, positions):
        msg = JointState()
        msg.name = joint_names
        msg.position = positions

        for _ in range(10):
            msg.header.stamp = self.get_clock().now().to_msg()
            self._joint_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.5)


def orientamento_pinza_orizzontale(yaw=0.0):
    """
    Orientamento diretto per gripper_left_grasping_link:
    asse di approccio verso il basso, dita orizzontali, yaw libero.
    """
    R_world_grasp = R.from_euler('z', yaw) * R.from_euler('x', np.pi)
    return R_world_grasp.as_quat()


def costruisci_target(position, yaw, raggio=0.05):
    """
    Posa di presa per un dato yaw: punto sul cerchio di raggio 'raggio'
    attorno a 'position' (approccio laterale, non dall'alto esatto), con
    l'orientamento del gripper coerente con quell'angolo.

    Stessa formula usata sia nello sweep esplorativo sia per ricostruire il
    target vincitore al momento dell'esecuzione vera -- deve restare
    IDENTICA nei due punti, o la posa eseguita non sarebbe quella per cui
    l'IK e' stato verificato.

    #yaw = 0       x = -r  y = 0
    #yaw = pi/2    x = 0   y = -r
    #yaw = pi      x = r   y = 0
    #yaw = 3*pi/2  x = 0   y = r
    """
    target = PoseStamped()
    target.header.frame_id = "base_footprint"
    target.pose.position.x = position[0] - raggio * np.cos(yaw)
    target.pose.position.y = position[1] - raggio * np.sin(yaw)
    target.pose.position.z = position[2]

    quat = orientamento_pinza_orizzontale(yaw=yaw)
    target.pose.orientation.x = quat[0]
    target.pose.orientation.y = quat[1]
    target.pose.orientation.z = quat[2]
    target.pose.orientation.w = quat[3]

    return target


def trova_yaw_ottimale(node, position, n_campioni, raggio=0.05):
    """
    Campiona diversi valori di yaw, prova l'IK per ciascuno (con KDL,
    solo pianificazione -- il braccio non si muove qui), e restituisce
    la soluzione più lontana dai limiti di giunto.

    Ogni candidato viene pianificato a partire dallo stato VERO attuale del
    braccio (start_positions=None -> is_diff=True), non da una posa finta.
    Motivo: la fattibilita' IK e il costo (distanza dai limiti) dipendono da
    dove il braccio si trova davvero adesso -- una configurazione a giunti
    tutti a zero e' arbitraria e potrebbe risultare irraggiungibile o
    avere un costo che non rispecchia affatto cosa succedera' quando poi si
    esegue per davvero. Valutare i candidati dallo stesso stato reale da cui
    poi si esegue e' l'unico modo per far si' che la classifica trovata qui
    sia effettivamente quella giusta per il robot vero.
    """
    migliore = None
    migliore_costo = -5
    migliore_indice = None

    # Punti della circonferenza calcolati tutti in anticipo (stesso raggio,
    # posizione, yaw campionati uniformemente) cosi' da poterli disegnare
    # subito tutti insieme in RViz, prima ancora di provare l'IK su ognuno.
    yaws = [2 * np.pi * i / n_campioni for i in range(n_campioni)]
    targets = [costruisci_target(position, yaw, raggio) for yaw in yaws]
    candidati = [
        {'x': t.pose.position.x, 'y': t.pose.position.y, 'z': t.pose.position.z,
         'stato': 'in_attesa'}
        for t in targets
    ]
    node.publish_candidates_markers(candidati)

    for i, (yaw, target) in enumerate(zip(yaws, targets)):
        result = node.send_goal(target, start_positions=None, plan_only=True)

        if result and result.result.error_code.val == 1:
            traj = result.result.planned_trajectory.joint_trajectory
            positions = traj.points[-1].positions

            costo = calcola_distanza_limiti(traj.joint_names, positions)

            print(f"yaw={yaw:.2f} rad -> SUCCESSO, costo={costo:.4f}")

            node.publish_joint_state(
                        list(traj.joint_names) + ['torso_lift_joint'],
                        list(positions) + [0.0]
                    )

            candidati[i]['stato'] = 'ok'

            if costo > migliore_costo:
                migliore_costo = costo
                migliore = (yaw, traj.joint_names, positions)
                migliore_indice = i
        else:
            print(f"yaw={yaw:.2f} rad -> fallito")
            candidati[i]['stato'] = 'fallito'

        # Aggiorna i colori via via che ogni candidato viene valutato.
        node.publish_candidates_markers(candidati)

    if migliore_indice is not None:
        candidati[migliore_indice]['stato'] = 'migliore'
        node.publish_candidates_markers(candidati)

    return migliore
    


def calcola_distanza_limiti(joint_names, positions):
    # Limiti reali del braccio sinistro, presi dall'URDF (rad)
    limiti = {
        'arm_left_1_joint': (-0.524, 4.712),
        'arm_left_2_joint': (-2.443, 1.134),
        'arm_left_3_joint': (-2.618, 2.618),
        'arm_left_4_joint': (-2.443, 1.134),
        'arm_left_5_joint': (-3.665, 1.571),
        'arm_left_6_joint': (-1.885, 3.002),
        'arm_left_7_joint': (-2.443, 2.443),
    }
    costo_totale = 0.0
    for name, pos in zip(joint_names, positions):
        if name in limiti:
            lower, upper = limiti[name]
            centro = (lower + upper) / 2
            semi_range = (upper - lower) / 2
            costo_totale += ((pos - centro) / semi_range) ** 2
    return -costo_totale  # negativo: più vicino a 0 = più centrato = meglio


def main():
    rclpy.init()
    node = MoveGroupClient()

    # --- Aspetta la detection del target (la coca) ---
    # Il target viene letto da centers_all/coke_can_center_base_footprint,
    # cioe' dallo stack multi-oggetto (rt_object_detection_node_all.py +
    # center_computation_all.py, lanciati da full_stack_all.launch.py) —
    # e' quello stack che pubblica anche gli ostacoli (pringles/biscotti),
    # quindi e' l'unico che nella pratica gira davvero insieme a questo nodo.
    # Il topic single-object cokecan_center_base_footprint (stack
    # full_stack.launch.py, senza ostacoli) NON viene piu' usato come target:
    # se non giri quello stack, restava in attesa di un topic mai pubblicato.
    print("Aspetto la detection della coca su centers_all/coke_can_center_base_footprint...")
    if not node.wait_for_topics(
        [lambda n: n.latest_centers_all.get('coke can')], timeout_sec=15.0
    ):
        print("Nessuna detection della coca ricevuta in tempo, esco.")
        node.destroy_node()
        rclpy.shutdown()
        return

    coke_center = node.latest_centers_all['coke can']
    posizione_target = (coke_center.point.x, coke_center.point.y, coke_center.point.z)
    print(f"Target (coca): x={posizione_target[0]:.3f} y={posizione_target[1]:.3f} z={posizione_target[2]:.3f}")

    # --- Ostacoli: pringles e biscotti, se rilevati in questo momento ---
    # Diamo una finestra breve per raccogliere quello che e' arrivato nel
    # frattempo (le callback girano solo mentre il nodo spinna) -- un
    # ostacolo mancante qui non blocca il planning, semplicemente MoveIt non
    # sapra' di doverlo evitare.
    rclpy.spin_once(node, timeout_sec=1.0)
    for class_name in OBSTACLE_CLASSES:
        center = node.latest_centers_all.get(class_name)
        if center is None:
            print(f"Nessuna detection recente per '{class_name}', ostacolo non aggiunto.")
            continue
        radius, height = OBSTACLE_DIMENSIONS[class_name]
        print(f"Aggiungo ostacolo '{class_name}' (r={radius}, h={height}) "
              f"a x={center.point.x:.3f} y={center.point.y:.3f} z={center.point.z:.3f}")
        node.add_object_obstacle(topic_slug(class_name), center, radius, height)

    # --- Ostacolo: tavolo (noto a priori, non dalla detection -- vedi TABLE_*) ---
    print(f"Aggiungo ostacolo 'table' in {TABLE_FRAME} "
          f"a x={TABLE_POSITION_XY[0]} y={TABLE_POSITION_XY[1]}...")
    node.add_table_obstacle()

    # Niente marker per il centro della lattina qui: lo pubblica gia'
    # center_computation_all.py su centers_all/coke_can_axis (asse verticale
    # reale, non solo un punto) -- ripeterlo qui sarebbe ridondante.

    # --- Sweep dello yaw (solo pianificazione, partenza dallo stato vero) ---
    raggio_sweep = 0.05
    migliore = trova_yaw_ottimale(node, position=posizione_target, n_campioni=10, raggio=raggio_sweep)

    if migliore:
        yaw, joint_names, positions = migliore
        print(f"\nMiglior yaw trovato: {yaw:.3f} rad")
        print("Configurazione finale (rad, dal planning esplorativo):")
        for name, pos in zip(joint_names, positions):
            print(f"  {name}: {pos:.4f}")

        # --- Esecuzione REALE della posa vincitrice ---
        # A differenza dello sweep sopra: partenza vera del braccio
        # (start_positions=None) e plan_only=False, quindi MoveIt pianifica
        # E ESEGUE per davvero -- il braccio si muove in Gazebo/RViz.
        target_finale = costruisci_target(posizione_target, yaw, raggio_sweep)
        print("\nEseguo la posa scelta sul braccio reale...")
        risultato = node.send_goal(target_finale, start_positions=None, plan_only=False)

        if risultato and risultato.result.error_code.val == 1:
            print("Esecuzione completata con successo.")
        else:
            codice = risultato.result.error_code.val if risultato else "nessuna risposta"
            print(f"Esecuzione fallita (error_code={codice}).")
    else:
        print("Nessuna soluzione trovata per nessun valore di yaw.")

    rclpy.shutdown()


    


if __name__ == "__main__":
    main()