"""Nodo di pose planning per il grasping: esplora pose candidate attorno al
target (sweep di yaw su una circonferenza), verifica IK e assenza di
collisioni via MoveIt, e sceglie il candidato piu' lontano dai limiti di
giunto (criterio joint margin), poi esegue la posa vincente sul braccio."""
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint,
    CollisionObject, PlanningScene, RobotState
)
from geometry_msgs.msg import PoseStamped, Pose, PointStamped, Point
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R
import numpy as np

import tf2_ros


# Real Time Factor osservato della simulazione su questa macchina (~0.4x rispetto al tempo reale).
SIM_REAL_TIME_FACTOR = 0.4


def real_seconds_for(sim_seconds):
    """Secondi di OROLOGIO REALE da aspettare per lasciar passare
    'sim_seconds' di tempo simulato, dato SIM_REAL_TIME_FACTOR."""
    return sim_seconds / SIM_REAL_TIME_FACTOR


# Topic del centro dell'oggetto (gia' in base_footprint, pubblicato da detection_and_ranging_occlusion_free).
COKE_CENTER_TOPIC = 'cokecan_center_base_footprint'

# Pipeline multi-oggetto: un topic per classe (stesso schema di center_computation_all.py).
TRACKED_CLASSES = ['coke can', 'pringles can', 'biscuits pack']

# Classi trattate come ostacoli (non target) nella scena di grasping.
OBSTACLE_CLASSES = ['pringles can', 'biscuits pack']

# Dimensioni reali (raggio, altezza in metri) dei modelli Gazebo.
OBSTACLE_DIMENSIONS = {
    'pringles can':  (0.04, 0.235),
    'biscuits pack': (0.029, 0.15),
}

# La coca (il target) e' aggiunta anche lei come ostacolo vero, senza eccezioni per il
# gripper: costringe MoveIt a scartare candidati che la toccherebbero durante l'approccio.
COKE_DIMENSIONS = (0.04, 0.15)

# Tavolo: posa nota a priori (non dalla detection diretta, difficile da ricostruire dalla
# camera). E' fisso nel MONDO, non rispetto al robot (che puo' essere altrove per teleop).
# Il robot rileva gia' in base_footprint la posizione degli oggetti tracciati (dati veri);
# le stesse posizioni nel mondo Gazebo sono note dal world file (KNOWN_OBJECT_WORLD_XY).
# Confrontando le due si ricostruisce la trasformazione rigida mondo -> base_footprint
# (Procrustes ortogonale, _fit_rigid_transform_2d), e da li' la posa del tavolo. Servono
# almeno 2 oggetti rilevati (un solo punto non determina la rotazione); altrimenti il
# tavolo non viene aggiunto.
TABLE_FRAME = 'base_footprint'

# Posizioni vere nel mondo Gazebo (pal_gazebo_worlds/worlds/poliBaMaster.world) degli
# stessi oggetti che il robot rileva, per ricostruire la trasformazione mondo -> base_footprint.
KNOWN_OBJECT_WORLD_XY = {
    # Deve restare sincronizzato con poliBaMaster.world.
    'coke can':      (5.0, 4.65),
    'pringles can':  (5.25, 5.05),
    'biscuits pack': (4.90, 4.95),
}
TABLE_WORLD_POSITION_XY = (5.0, 5.0)


def _fit_rigid_transform_2d(world_points, local_points):
    """Rotazione + traslazione che allineano al meglio (Procrustes ortogonale via SVD)
    due insiemi di punti corrispondenti. Ritorna (R 2x2, t) tali che local ~= R @ world + t."""
    world = np.array(world_points)
    local = np.array(local_points)
    world_centroid = world.mean(axis=0)
    local_centroid = local.mean(axis=0)
    world_centered = world - world_centroid
    local_centered = local - local_centroid

    H = world_centered.T @ local_centered
    U, _, Vt = np.linalg.svd(H)
    rotation = Vt.T @ U.T
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    translation = local_centroid - rotation @ world_centroid
    return rotation, translation


# Geometria vera (pal_gazebo_worlds/models/table_0m8/table.sdf): piano 1.0x0.8x0.03,
# superficie a z=0.815. Un unico box pieno da terra alla superficie (niente gambe modellate),
# sufficiente perche' il gruppo di planning e' solo il braccio, non la base.
TABLE_SURFACE_TOP_Z = 0.815
TABLE_FOOTPRINT_XY = (1.0, 0.8)

# Margine di sicurezza orizzontale (metri), per il rumore residuo della detection.
TABLE_SAFETY_MARGIN = 0.15
TABLE_FOOTPRINT_XY_WITH_MARGIN = (
    TABLE_FOOTPRINT_XY[0] + TABLE_SAFETY_MARGIN,
    TABLE_FOOTPRINT_XY[1] + TABLE_SAFETY_MARGIN,
)

# Limiti reali del braccio sinistro, dall'URDF (rad).
JOINT_LIMITS = {
    'arm_left_1_joint': (-0.524, 4.712),
    'arm_left_2_joint': (-2.443, 1.134),
    'arm_left_3_joint': (-2.618, 2.618),
    'arm_left_4_joint': (-2.443, 1.134),
    'arm_left_5_joint': (-3.665, 1.571),
    'arm_left_6_joint': (-1.885, 3.002),
    'arm_left_7_joint': (-2.443, 2.443),
}

JOINT_MARGIN_RESULTS_DIR = '/home/user/ros_workspace/src/pose_optimizer/joint_margin_results'


def topic_slug(class_name):
    """'coke can' -> 'coke_can'. Deve restare identica a quella in center_computation_all.py."""
    return class_name.replace(' ', '_')


class MoveGroupClient(Node):
    def __init__(self):
        super().__init__('joint_margin_optimizer')
        self._client = ActionClient(self, MoveGroup, '/move_action')
        self._scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self._candidates_marker_pub = self.create_publisher(
            MarkerArray, '/grasp_candidates_marker', 10
        )
        # Topic separato per i marker di debug degli ostacoli: PlanningScene mischia i
        # nostri CollisionObject con l'octomap della percezione, rendendoli indistinguibili.
        self._obstacles_marker_pub = self.create_publisher(
            Marker, '/obstacles_debug_marker', 10
        )
        self._obstacle_markers = {}
        self.create_timer(1.0, self._republish_obstacle_markers)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
                            dimensions=(*TABLE_FOOTPRINT_XY_WITH_MARGIN, TABLE_SURFACE_TOP_Z)):
        """Aggiunge il tavolo come box pieno da terra alla sua superficie, ancorato a
        TABLE_FRAME. La posa si ricostruisce confrontando gli oggetti tracciati visti
        ora dal robot con le loro posizioni note nel mondo Gazebo (_fit_rigid_transform_2d)."""
        world_points = []
        local_points = []
        for class_name, world_xy in KNOWN_OBJECT_WORLD_XY.items():
            center = self.latest_centers_all.get(class_name)
            if center is None:
                continue
            world_points.append(world_xy)
            local_points.append((center.point.x, center.point.y))

        if len(world_points) < 2:
            self.get_logger().error(
                f"Solo {len(world_points)} oggetto/i tracciato/i rilevato/i "
                f"(servono almeno 2 per ricostruire posizione E rotazione del "
                f"mondo rispetto al robot) -- tavolo NON aggiunto come ostacolo."
            )
            return

        rotation, translation = _fit_rigid_transform_2d(world_points, local_points)
        table_xy = rotation @ np.array(TABLE_WORLD_POSITION_XY) + translation
        table_yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        self.get_logger().info(
            f"Tavolo ricostruito da {len(world_points)} oggetti tracciati "
            f"({', '.join(c for c in KNOWN_OBJECT_WORLD_XY if self.latest_centers_all.get(c) is not None)}): "
            f"x={table_xy[0]:.3f} y={table_xy[1]:.3f} yaw={table_yaw:.3f} rad."
        )

        pose = Pose()
        pose.position.x = float(table_xy[0])
        pose.position.y = float(table_xy[1])
        pose.position.z = TABLE_SURFACE_TOP_Z / 2.0
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
         ) = R.from_euler('z', table_yaw).as_quat()

        obj = CollisionObject()
        obj.header.frame_id = frame_id
        obj.id = "table"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(dimensions)

        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.world.collision_objects.append(obj)
        scene.is_diff = True

        for _ in range(5):
            self._scene_pub.publish(scene)
            rclpy.spin_once(self, timeout_sec=0.3)

        self.get_logger().info(
            f"Tavolo aggiunto in {frame_id} a "
            f"x={pose.position.x:.3f} y={pose.position.y:.3f} "
            f"z={pose.position.z:.3f} (CollisionObject + marker debug pubblicati)."
        )

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.ns = "table_obstacle_debug"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x, marker.scale.y, marker.scale.z = dimensions
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 0.5
        self._obstacle_markers["table"] = marker
        for _ in range(5):
            marker.header.stamp = self.get_clock().now().to_msg()
            self._obstacles_marker_pub.publish(marker)
            rclpy.spin_once(self, timeout_sec=0.3)

        axis_marker = Marker()
        axis_marker.header.frame_id = frame_id
        axis_marker.ns = "table_axis_debug"
        axis_marker.id = 0
        axis_marker.type = Marker.LINE_STRIP
        axis_marker.action = Marker.ADD
        axis_marker.scale.x = 0.02
        axis_marker.color.r = 1.0
        axis_marker.color.g = 0.5
        axis_marker.color.b = 0.0
        axis_marker.color.a = 1.0
        axis_length = 0.60
        axis_marker.points = [
            Point(x=pose.position.x, y=pose.position.y, z=pose.position.z - axis_length / 2.0),
            Point(x=pose.position.x, y=pose.position.y, z=pose.position.z + axis_length / 2.0),
        ]
        self._obstacle_markers["table_axis"] = axis_marker
        for _ in range(5):
            axis_marker.header.stamp = self.get_clock().now().to_msg()
            self._obstacles_marker_pub.publish(axis_marker)
            rclpy.spin_once(self, timeout_sec=0.3)

    def _republish_obstacle_markers(self):
        """Ripubblica periodicamente ogni marker di debug gia' noto, con uno stamp fresco."""
        for marker in self._obstacle_markers.values():
            marker.header.stamp = self.get_clock().now().to_msg()
            self._obstacles_marker_pub.publish(marker)

    def wait_for_topics(self, attribute_getters, timeout_sec=10.0):
        """Spinna il nodo finche' tutti gli attribute_getters restituiscono un valore non-None, o scade il timeout."""
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            if all(getter(self) is not None for getter in attribute_getters):
                return True
            rclpy.spin_once(self, timeout_sec=0.2)
        return all(getter(self) is not None for getter in attribute_getters)

    def add_object_obstacle(self, object_id, center: PointStamped, radius, height):
        """Aggiunge un ostacolo cilindrico verticale alla planning scene, nel punto
        (gia' in base_footprint) pubblicato da detection_and_ranging_occlusion_free."""
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

        marker = Marker()
        marker.header.frame_id = center.header.frame_id
        marker.ns = f"{object_id}_obstacle_debug"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = marker.scale.y = radius * 2.0  # CYLINDER Marker vuole il diametro, non il raggio
        marker.scale.z = height
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 0.5
        self._obstacle_markers[object_id] = marker
        for _ in range(5):
            marker.header.stamp = self.get_clock().now().to_msg()
            self._obstacles_marker_pub.publish(marker)
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
        """Invia un goal MoveGroup. start_positions=None pianifica dalla posizione vera del
        braccio; plan_only=False esegue davvero la traiettoria (usare solo con start_positions=None)."""
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


def orientamento_pinza_orizzontale(yaw=0.0):
    """Orientamento per gripper_left_grasping_link: approccio verso il basso, dita orizzontali, yaw libero."""
    R_world_grasp = R.from_euler('z', yaw) * R.from_euler('x', np.pi)
    return R_world_grasp.as_quat()


def costruisci_target(position, yaw, raggio=0.05):
    """Posa di presa per un dato yaw: punto sul cerchio di raggio 'raggio' attorno a
    'position' (approccio laterale), con l'orientamento del gripper coerente con l'angolo."""
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
    """Campiona diversi valori di yaw, prova l'IK per ciascuno (pianificazione, il
    braccio non si muove) partendo dallo stato vero attuale del braccio, e restituisce
    la soluzione piu' lontana dai limiti di giunto (calcola_distanza_limiti)."""
    migliore = None
    migliore_costo = -float('inf')
    migliore_indice = None
    candidati_riusciti = []

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

            candidati[i]['stato'] = 'ok'
            candidati_riusciti.append({
                'yaw': yaw, 'indice_globale': i,
                'joint_names': list(traj.joint_names), 'positions': list(positions),
                'costo': costo,
            })

            if costo > migliore_costo:
                migliore_costo = costo
                migliore = (yaw, traj.joint_names, positions)
                migliore_indice = i
        else:
            print(f"yaw={yaw:.2f} rad -> fallito")
            candidati[i]['stato'] = 'fallito'

        node.publish_candidates_markers(candidati)

    if migliore_indice is not None:
        candidati[migliore_indice]['stato'] = 'migliore'
        node.publish_candidates_markers(candidati)

    plot_candidati_giunti(candidati_riusciti, migliore_indice)

    return migliore


def calcola_distanza_limiti(joint_names, positions):
    """Costo negativo: 0 = tutti i giunti centrati nel range, sempre piu' negativo verso i limiti."""
    costo_totale = 0.0
    for name, pos in zip(joint_names, positions):
        if name in JOINT_LIMITS:
            lower, upper = JOINT_LIMITS[name]
            centro = (lower + upper) / 2
            semi_range = (upper - lower) / 2
            costo_totale += ((pos - centro) / semi_range) ** 2
    return -costo_totale


def plot_candidati_giunti(candidati_riusciti, migliore_indice_globale,
                           output_path=None):
    """Per ogni candidato riuscito, disegna i suoi 7 angoli di giunto confrontati
    con i rispettivi limiti (un grafico per giunto)."""
    if not candidati_riusciti:
        print("Nessun candidato riuscito, niente da plottare.")
        return

    if output_path is None:
        os.makedirs(JOINT_MARGIN_RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(JOINT_MARGIN_RESULTS_DIR, 'joint_margin_candidates.png')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    joint_order = list(JOINT_LIMITS.keys())
    fig, axes = plt.subplots(len(joint_order), 1, figsize=(9, 2.0 * len(joint_order)), sharex=True)

    for ax, joint_name in zip(axes, joint_order):
        lower, upper = JOINT_LIMITS[joint_name]
        centro = (lower + upper) / 2

        xs, ys, colors = [], [], []
        for c in candidati_riusciti:
            idx = c['joint_names'].index(joint_name)
            xs.append(c['yaw'])
            ys.append(c['positions'][idx])
            colors.append('gold' if c['indice_globale'] == migliore_indice_globale else 'tab:blue')

        ax.axhline(lower, color='red', linestyle='--', linewidth=1, label='limite')
        ax.axhline(upper, color='red', linestyle='--', linewidth=1)
        ax.axhline(centro, color='green', linestyle=':', linewidth=1, alpha=0.6, label='centro range')
        ax.scatter(xs, ys, c=colors, zorder=3, edgecolors='black', linewidths=0.5)
        margin = (upper - lower) * 0.1
        ax.set_ylim(lower - margin, upper + margin)
        ax.set_ylabel(joint_name.replace('_joint', ''), fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[0].legend(fontsize=7, loc='upper right')
    axes[-1].set_xlabel('yaw del candidato (rad)')
    fig.suptitle('Angoli di giunto per candidato riuscito (tratteggio = limiti, oro = scelto)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Grafico giunti-vs-candidati salvato in {output_path}")


def leggi_posa_end_effector_reale(node, link_name="gripper_left_grasping_link",
                                   frame_id="base_footprint", timeout_sec=5.0):
    """Legge da TF la posa vera (non quella comandata) di 'link_name' rispetto a 'frame_id',
    dopo l'esecuzione reale. Ritorna (x, y, z, roll, pitch, yaw), o None se la TF non arriva in tempo."""
    deadline = time.time() + timeout_sec
    messaggio_errore = "timeout"
    while time.time() < deadline:
        try:
            transform = node.tf_buffer.lookup_transform(frame_id, link_name, rclpy.time.Time())
            t = transform.transform.translation
            q = transform.transform.rotation
            roll, pitch, yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')
            return (t.x, t.y, t.z, roll, pitch, yaw)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as errore:
            messaggio_errore = str(errore)
            rclpy.spin_once(node, timeout_sec=0.2)
    node.get_logger().error(f"TF {frame_id} <- {link_name} non arrivata in tempo: {messaggio_errore}")
    return None


def plot_posa_end_effector(node, output_path=None):
    """Grafico a barre di x, y, z, roll, pitch, yaw della posa vera raggiunta dall'end
    effector (letta da TF dopo l'esecuzione, non il target comandato)."""
    posa = leggi_posa_end_effector_reale(node)
    if posa is None:
        print("Impossibile leggere la posa reale dell'end effector, nessun grafico.")
        return
    x, y, z, roll, pitch, yaw = posa
    print(
        f"Posa reale end effector (dopo l'esecuzione): "
        f"x={x:.4f} y={y:.4f} z={z:.4f} m -- roll={roll:.4f} pitch={pitch:.4f} yaw={yaw:.4f} rad"
    )

    if output_path is None:
        os.makedirs(JOINT_MARGIN_RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(JOINT_MARGIN_RESULTS_DIR, 'optimal_end_effector_pose.png')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax_pos, ax_rot) = plt.subplots(1, 2, figsize=(9, 4.5))

    ax_pos.bar(['x', 'y', 'z'], [x, y, z], color='tab:blue')
    ax_pos.axhline(0, color='black', linewidth=0.8)
    ax_pos.set_ylabel('metri')
    ax_pos.set_title('Posizione (base_footprint)')
    ax_pos.grid(True, alpha=0.2)

    gradi = np.degrees([roll, pitch, yaw])
    ax_rot.bar(['roll', 'pitch', 'yaw'], gradi, color='tab:orange')
    ax_rot.axhline(0, color='black', linewidth=0.8)
    ax_rot.set_ylabel('gradi')
    ax_rot.set_title('Orientamento (base_footprint)')
    ax_rot.grid(True, alpha=0.2)

    fig.suptitle("Posa REALE dell'end effector, candidato ottimo (dopo l'esecuzione)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Grafico posa end effector salvato in {output_path}")


def main():
    rclpy.init()
    node = MoveGroupClient()

    print("Aspetto la detection della coca su centers_all/coke_can_center_base_footprint...")
    if not node.wait_for_topics(
        [lambda n: n.latest_centers_all.get('coke can')],
        timeout_sec=real_seconds_for(15.0),
    ):
        print("Nessuna detection della coca ricevuta in tempo, esco.")
        node.destroy_node()
        rclpy.shutdown()
        return

    coke_center = node.latest_centers_all['coke can']
    posizione_target = (coke_center.point.x, coke_center.point.y, coke_center.point.z)
    print(f"Target (coca): x={posizione_target[0]:.3f} y={posizione_target[1]:.3f} z={posizione_target[2]:.3f}")

    coke_radius, coke_height = COKE_DIMENSIONS
    print(f"Aggiungo ostacolo 'coke can' (il target stesso, r={coke_radius}, h={coke_height}) "
          f"a x={coke_center.point.x:.3f} y={coke_center.point.y:.3f} z={coke_center.point.z:.3f}")
    node.add_object_obstacle('coke', coke_center, coke_radius, coke_height)

    node.wait_for_topics(
        [lambda n, c=c: n.latest_centers_all.get(c) for c in OBSTACLE_CLASSES],
        timeout_sec=real_seconds_for(8.0),
    )

    for class_name in OBSTACLE_CLASSES:
        center = node.latest_centers_all.get(class_name)
        if center is None:
            print(f"Nessuna detection recente per '{class_name}', ostacolo non aggiunto.")
            continue
        radius, height = OBSTACLE_DIMENSIONS[class_name]
        print(f"Aggiungo ostacolo '{class_name}' (r={radius}, h={height}) "
              f"a x={center.point.x:.3f} y={center.point.y:.3f} z={center.point.z:.3f}")
        node.add_object_obstacle(topic_slug(class_name), center, radius, height)

    print(f"Aggiungo ostacolo 'table' in {TABLE_FRAME} "
          f"(ricostruito dagli oggetti tracciati)...")
    node.add_table_obstacle()

    raggio_sweep = 0.05
    migliore = trova_yaw_ottimale(node, position=posizione_target, n_campioni=20, raggio=raggio_sweep)

    if migliore:
        yaw, joint_names, positions = migliore
        print(f"\nMiglior yaw trovato: {yaw:.3f} rad")
        print("Configurazione finale (rad, dal planning esplorativo):")
        for name, pos in zip(joint_names, positions):
            print(f"  {name}: {pos:.4f}")

        target_finale = costruisci_target(posizione_target, yaw, raggio_sweep)
        print("\nEseguo la posa scelta sul braccio reale...")
        risultato = node.send_goal(target_finale, start_positions=None, plan_only=False)

        if risultato and risultato.result.error_code.val == 1:
            print("Esecuzione completata con successo.")
            plot_posa_end_effector(node)
        else:
            codice = risultato.result.error_code.val if risultato else "nessuna risposta"
            print(f"Esecuzione fallita (error_code={codice}).")
    else:
        print("Nessuna soluzione trovata per nessun valore di yaw.")

    rclpy.shutdown()


if __name__ == "__main__":
    main()