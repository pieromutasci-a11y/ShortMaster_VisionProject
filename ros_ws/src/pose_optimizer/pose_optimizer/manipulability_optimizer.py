import os
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint,
    CollisionObject, PlanningScene, RobotState
)
from geometry_msgs.msg import PoseStamped, Pose, PointStamped, Point
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R
import numpy as np

import PyKDL
from urdf_parser_py.urdf import URDF


# Real Time Factor osservato della simulazione in questa macchina: gira a
# circa 0.4x rispetto al tempo reale (CPU quasi satura, YOLO su CPU pesa
# parecchio) -- un secondo di QUESTO OROLOGIO (wall-clock, quello con cui
# lavorano i nostri timeout/deadline in Python) corrisponde a solo ~0.4
# secondi di tempo simulato/ROS, che e' quello con cui sono timestampati i
# dati che aspettiamo (TF, detection). Tutte le attese sotto sono espresse
# in secondi di tempo simulato "voluto" e convertite in secondi reali da
# aspettare per davvero -- altrimenti si rischia di arrendersi mentre, dal
# punto di vista della simulazione, e' passato solo un attimo.
SIM_REAL_TIME_FACTOR = 0.4


def real_seconds_for(sim_seconds):
    """Secondi di OROLOGIO REALE da aspettare per lasciar passare
    'sim_seconds' di tempo simulato, dato SIM_REAL_TIME_FACTOR."""
    return sim_seconds / SIM_REAL_TIME_FACTOR


# Topic del centro dell'oggetto (gia' in base_footprint, pubblicato da
# detection_and_ranging -- vedi center_computation.py/center_computation_all.py).
COKE_CENTER_TOPIC = 'cokecan_center_base_footprint'
TRACKED_CLASSES = ['coke can', 'pringles can', 'biscuits pack']
OBSTACLE_CLASSES = ['pringles can', 'biscuits pack']

OBSTACLE_DIMENSIONS = {
    'pringles can':  (0.04, 0.235),
    'biscuits pack': (0.029, 0.15),
}
COKE_DIMENSIONS = (0.04, 0.15)  # raggio, altezza -- vero, da s3_cocacola/model.sdf

TABLE_FRAME = 'base_footprint'
KNOWN_OBJECT_WORLD_XY = {
    'coke can':      (5.0, 4.65),
    'pringles can':  (5.25, 5.05),
    'biscuits pack': (4.90, 4.95),
}
TABLE_WORLD_POSITION_XY = (5.0, 5.0)


def _fit_rigid_transform_2d(world_points, local_points):
    """
    Dati >= 2 punti corrispondenti (stesso ordine) in due sistemi di
    riferimento diversi ("mondo" e "locale"), ricava la rotazione + la
    traslazione che meglio li allinea (minimi quadrati) -- problema di
    Procrustes ortogonale standard, risolto via SVD. Restituisce (R 2x2,
    t) tali che: local_point ~= R @ world_point + t.
    """
    world = np.array(world_points)
    local = np.array(local_points)
    world_centroid = world.mean(axis=0)
    local_centroid = local.mean(axis=0)
    world_centered = world - world_centroid
    local_centered = local - local_centroid

    H = world_centered.T @ local_centered
    U, _, Vt = np.linalg.svd(H)
    rotation = Vt.T @ U.T
    if np.linalg.det(rotation) < 0:  # evita una riflessione invece di una rotazione pura
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    translation = local_centroid - rotation @ world_centroid
    return rotation, translation


TABLE_SURFACE_TOP_Z = 0.815
TABLE_FOOTPRINT_XY = (1.0, 0.8)
TABLE_SAFETY_MARGIN = 0.15  # metri, aggiunti a ciascuna dimensione orizzontale
TABLE_FOOTPRINT_XY_WITH_MARGIN = (
    TABLE_FOOTPRINT_XY[0] + TABLE_SAFETY_MARGIN,
    TABLE_FOOTPRINT_XY[1] + TABLE_SAFETY_MARGIN,
)

# Catena cinematica per lo Jacobiano: dal tiago_pro.urdf.xacro, arm_left e'
# montato con parent="torso_lift_link" -- e' la' che inizia la catena dei 7
# giunti (arm_left_1..7_joint) fino al punto di presa.
KDL_BASE_LINK = 'torso_lift_link'
KDL_TIP_LINK = 'gripper_left_grasping_link'

# Dentro l'albero SORGENTE del pacchetto, non nella copia installata (che
# viene rigenerata ad ogni build e perderebbe i grafici) -- stessa
# convenzione gia' usata in vision_pipeline per data//models/ e in
# joint_margin_optimizer per i suoi risultati.
MANIPULABILITY_RESULTS_DIR = '/home/user/ros_workspace/src/pose_optimizer/manipulability_results'

# Limiti reali del braccio sinistro, presi dall'URDF (rad) -- QUI servono
# solo come riferimento visivo nel grafico dei giunti (plot_candidati_giunti),
# non per il criterio di scelta (quello e' calcola_manipolabilita, non ha
# niente a che vedere con i limiti di giunto -- una configurazione puo'
# avere giunti ben centrati ed essere comunque vicina a una singolarita',
# es. gomito quasi disteso).
JOINT_LIMITS = {
    'arm_left_1_joint': (-0.524, 4.712),
    'arm_left_2_joint': (-2.443, 1.134),
    'arm_left_3_joint': (-2.618, 2.618),
    'arm_left_4_joint': (-2.443, 1.134),
    'arm_left_5_joint': (-3.665, 1.571),
    'arm_left_6_joint': (-1.885, 3.002),
    'arm_left_7_joint': (-2.443, 2.443),
}


def topic_slug(class_name):
    """'coke can' -> 'coke_can'. Deve restare identica a quella in center_computation_all.py."""
    return class_name.replace(' ', '_')


# --- Costruzione della catena KDL dall'URDF, fatta a mano ---
#
# Il pacchetto che normalmente farebbe questo lavoro (kdl_parser_py) non
# esiste per ROS2 Humble (verificato: solo la libreria C++ e' pubblicata,
# ros-humble-kdl-parser, senza binding Python) -- quindi costruiamo la
# catena da soli, con la stessa matematica standard che userebbe quel
# pacchetto: non e' un'invenzione, e' la conversione URDF -> KDL descritta
# nella documentazione di Orocos KDL.
#
# Per ogni joint lungo il percorso: l'origine URDF (xyz, rpy) diventa una
# PyKDL.Frame ("F_parent_jnt", la posa del giunto rispetto al link
# genitore). Il PyKDL.Joint si costruisce con quella stessa origine e con
# l'asse di rotazione/traslazione (ruotato nell'orientamento di
# F_parent_jnt) -- e la STESSA F_parent_jnt va anche passata come "tip
# frame" del Segment: puo' sembrare un doppio conteggio, ma non lo e' --
# PyKDL.Joint(origin, axis, ...) definisce il moto del giunto come una
# rotazione/traslazione attorno a un asse che passa per "origin" (non
# attorno all'origine del segmento), quindi la composizione
# joint.pose(q) * F_parent_jnt si semplifica esattamente in
# "F_parent_jnt seguita da una rotazione/traslazione pura attorno
# all'asse locale" -- cioe' la cinematica vera descritta dall'URDF.
def _kdl_frame_from_urdf_origin(origin):
    if origin is None:
        return PyKDL.Frame.Identity()
    xyz = origin.xyz if origin.xyz is not None else [0.0, 0.0, 0.0]
    rpy = origin.rpy if origin.rpy is not None else [0.0, 0.0, 0.0]
    return PyKDL.Frame(PyKDL.Rotation.RPY(*rpy), PyKDL.Vector(*xyz))


def _trova_catena_joint_urdf(robot, base_link, tip_link):
    """Risale da tip_link a base_link seguendo i genitori (l'URDF e' un
    albero: ogni link non-radice ha esattamente un joint genitore), e
    restituisce i joint attraversati in ordine base -> tip."""
    child_to_joint = {joint.child: joint for joint in robot.joints}
    joints_al_contrario = []
    corrente = tip_link
    while corrente != base_link:
        joint = child_to_joint.get(corrente)
        if joint is None:
            raise RuntimeError(
                f"Risalendo da '{tip_link}' non si arriva a '{base_link}': "
                f"nessun joint genitore per '{corrente}' -- controlla "
                f"KDL_BASE_LINK/KDL_TIP_LINK (nomi presi da tiago_pro.urdf.xacro)."
            )
        joints_al_contrario.append(joint)
        corrente = joint.parent
    joints_al_contrario.reverse()
    return joints_al_contrario


def costruisci_catena_kdl(robot_description_xml, base_link, tip_link):
    """
    Restituisce (chain, nomi_giunti_attuati): la catena PyKDL da base_link a
    tip_link e i nomi dei suoi giunti NON fissi, nello stesso ordine con cui
    la catena li vuole nel JntArray -- da usare per rimappare
    (joint_names, positions) di MoveIt (che potrebbero avere un ordine
    diverso) prima di calcolare lo Jacobiano.
    """
    robot = URDF.from_xml_string(robot_description_xml)
    chain = PyKDL.Chain()
    nomi_giunti_attuati = []

    for joint in _trova_catena_joint_urdf(robot, base_link, tip_link):
        f_parent_jnt = _kdl_frame_from_urdf_origin(joint.origin)

        if joint.type == 'fixed':
            kdl_joint = PyKDL.Joint(joint.name, PyKDL.Joint.Fixed)
        else:
            axis_xyz = joint.axis if joint.axis is not None else [0.0, 0.0, 1.0]
            asse_nel_genitore = f_parent_jnt.M * PyKDL.Vector(*axis_xyz)
            if joint.type in ('revolute', 'continuous'):
                kdl_joint = PyKDL.Joint(joint.name, f_parent_jnt.p, asse_nel_genitore, PyKDL.Joint.RotAxis)
            elif joint.type == 'prismatic':
                kdl_joint = PyKDL.Joint(joint.name, f_parent_jnt.p, asse_nel_genitore, PyKDL.Joint.TransAxis)
            else:
                raise RuntimeError(f"Tipo di joint non gestito: '{joint.type}' ({joint.name})")
            nomi_giunti_attuati.append(joint.name)

        chain.addSegment(PyKDL.Segment(joint.child, kdl_joint, f_parent_jnt))

    return chain, nomi_giunti_attuati


class MoveGroupClient(Node):
    def __init__(self):
        super().__init__('manipulability_optimizer')
        self._client = ActionClient(self, MoveGroup, '/move_action')
        self._scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self._candidates_marker_pub = self.create_publisher(
            MarkerArray, '/grasp_candidates_marker', 10
        )
        # Marker di debug per TUTTI i nostri ostacoli (tavolo, pringles,
        # biscotti) -- stesse pose/dimensioni dei CollisionObject inviati a
        # MoveIt, ma su un topic tutto nostro: PlanningScene di RViz mischia
        # i nostri ostacoli con l'octomap della percezione (es. il pavimento,
        # enorme) senza modo di separarli, quindi qui vediamo solo i nostri.
        self._obstacles_marker_pub = self.create_publisher(
            Marker, '/obstacles_debug_marker', 10
        )
        self._obstacle_markers = {}
        self.create_timer(1.0, self._republish_obstacle_markers)

        # --- Catena KDL per lo Jacobiano (vedi calcola_manipolabilita) ---
        # /robot_description e' pubblicato da robot_state_publisher con QoS
        # transient_local (il messaggio resta "agganciato" per chi si
        # iscrive dopo, non serve arrivare prima della pubblicazione).
        self.robot_description_xml = None
        qos_robot_description = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            String, '/robot_description', self._robot_description_callback,
            qos_robot_description,
        )
        self._kdl_chain = None
        self._kdl_chain_joint_names = None
        self._kdl_jac_solver = None

        # --- Sottoscrizioni ai centri pubblicati da detection_and_ranging ---
        self.latest_coke_center = None
        self.coke_center_sub = self.create_subscription(
            PointStamped, COKE_CENTER_TOPIC, self._coke_center_callback, 10
        )

        self.latest_centers_all = {}
        self._centers_all_subs = {}
        for class_name in TRACKED_CLASSES:
            slug = topic_slug(class_name)
            topic = f'centers_all/{slug}_center_base_footprint'
            self._centers_all_subs[class_name] = self.create_subscription(
                PointStamped, topic,
                lambda msg, name=class_name: self._centers_all_callback(msg, name),
                10,
            )

        self.get_logger().info(
            f"In ascolto su '{COKE_CENTER_TOPIC}' e su centers_all/* per: "
            f"{', '.join(TRACKED_CLASSES)}"
        )

    def _robot_description_callback(self, msg: String):
        if self.robot_description_xml is None:
            self.robot_description_xml = msg.data
            self.get_logger().info("URDF ricevuto da /robot_description.")

    def build_kdl_chain(self):
        """
        Costruisce la catena KDL_BASE_LINK -> KDL_TIP_LINK dall'URDF vero
        del robot (non da un URDF a parte tenuto a mano: se il robot cambia,
        questo si aggiorna da solo) e il solver dello Jacobiano -- una volta
        sola, poi riusata per ogni candidato.
        """
        if self._kdl_chain is not None:
            return  # gia' costruita

        chain, chain_joint_names = costruisci_catena_kdl(
            self.robot_description_xml, KDL_BASE_LINK, KDL_TIP_LINK
        )
        if chain.getNrOfJoints() == 0:
            raise RuntimeError(
                f"Catena KDL {KDL_BASE_LINK} -> {KDL_TIP_LINK} vuota -- "
                f"controlla i nomi dei link (vedi tiago_pro.urdf.xacro)."
            )

        self._kdl_chain = chain
        self._kdl_chain_joint_names = chain_joint_names
        self._kdl_jac_solver = PyKDL.ChainJntToJacSolver(chain)
        self.get_logger().info(
            f"Catena KDL costruita: {KDL_BASE_LINK} -> {KDL_TIP_LINK}, "
            f"{len(chain_joint_names)} giunti: {chain_joint_names}"
        )

    def calcola_manipolabilita(self, joint_names, positions):
        """
        Indice di manipolabilita' di Yoshikawa: w(q) = sqrt(det(J(q) J(q)^T)).
        Vale 0 esattamente in una singolarita' cinematica, piu' alto quanto
        piu' la configurazione e' "ben condizionata" (capace di muoversi bene
        in ogni direzione cartesiana). PIU' ALTO = MEGLIO, come
        calcola_distanza_limiti in joint_margin_optimizer -- stessa
        convenzione, criterio diverso.
        """
        self.build_kdl_chain()

        nome_a_valore = dict(zip(joint_names, positions))
        try:
            valori = [nome_a_valore[nome] for nome in self._kdl_chain_joint_names]
        except KeyError as errore:
            raise RuntimeError(
                f"Il giunto {errore} della catena KDL non e' nella traiettoria "
                f"restituita da MoveIt -- controlla KDL_BASE_LINK/KDL_TIP_LINK."
            )

        q = PyKDL.JntArray(len(valori))
        for i, v in enumerate(valori):
            q[i] = v

        jacobiano = PyKDL.Jacobian(self._kdl_chain.getNrOfJoints())
        self._kdl_jac_solver.JntToJac(q, jacobiano)

        J = np.array([[jacobiano[r, c] for c in range(jacobiano.columns())]
                      for r in range(jacobiano.rows())])
        determinante = np.linalg.det(J @ J.T)
        # max(...,0.0): vicino a una singolarita' il determinante puo' uscire
        # leggermente negativo per puro errore numerico di arrotondamento,
        # non ha senso fisico un indice negativo -- si tronca a zero.
        return float(np.sqrt(max(determinante, 0.0)))

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
        """
        Box pieno da terra (z=0) alla superficie del tavolo, ancorato a
        TABLE_FRAME = 'base_footprint'. La posizione si ricostruisce
        confrontando dove il robot vede ORA (in base_footprint) gli oggetti
        tracciati con dove quegli stessi oggetti si trovano per certo nel
        mondo Gazebo -- vedi _fit_rigid_transform_2d.
        """
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
        for marker in self._obstacle_markers.values():
            marker.header.stamp = self.get_clock().now().to_msg()
            self._obstacles_marker_pub.publish(marker)

    def wait_for_topics(self, attribute_getters, timeout_sec=10.0):
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            if all(getter(self) is not None for getter in attribute_getters):
                return True
            rclpy.spin_once(self, timeout_sec=0.2)
        return all(getter(self) is not None for getter in attribute_getters)

    def add_object_obstacle(self, object_id, center: PointStamped, radius, height):
        obj = CollisionObject()
        obj.header.frame_id = center.header.frame_id
        obj.id = object_id

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.CYLINDER
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
        marker.scale.x = marker.scale.y = radius * 2.0
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
        colori = {
            'in_attesa': (0.6, 0.6, 0.6, 0.6),
            'ok':        (0.0, 1.0, 0.0, 0.8),
            'fallito':   (1.0, 0.0, 0.0, 0.8),
            'migliore':  (1.0, 0.84, 0.0, 1.0),
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
        primitive.dimensions = [0.005]
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
    R_world_grasp = R.from_euler('z', yaw) * R.from_euler('x', np.pi)
    return R_world_grasp.as_quat()


def costruisci_target(position, yaw, raggio=0.05):
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
    Campiona diversi valori di yaw, prova l'IK per ciascuno (con KDL, solo
    pianificazione -- il braccio non si muove qui), e restituisce la
    soluzione con la manipolabilita' piu' alta (vedi
    MoveGroupClient.calcola_manipolabilita) -- criterio diverso da
    joint_margin_optimizer (che sceglie in base ai limiti di giunto), stesso
    identico meccanismo di sweep.
    """
    migliore = None
    migliore_costo = -float('inf')  # la manipolabilita' e' sempre >= 0, ma -inf resta la scelta piu' sicura
    migliore_indice = None
    candidati_riusciti = []

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

            costo = node.calcola_manipolabilita(traj.joint_names, positions)

            print(f"yaw={yaw:.2f} rad -> SUCCESSO, manipolabilita'={costo:.5f}")

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

    plot_candidati_manipolabilita(candidati_riusciti, migliore_indice)
    plot_candidati_giunti(candidati_riusciti, migliore_indice)
    if migliore is not None:
        plot_ellissoide_manipolabilita(node, migliore[1], migliore[2])

    return migliore


def plot_candidati_giunti(candidati_riusciti, migliore_indice_globale, output_path=None):
    """
    Come plot_candidati_manipolabilita, ma i 7 angoli di giunto di ogni
    candidato riuscito (stesso stile del grafico di joint_margin_optimizer)
    invece dell'indice di manipolabilita' -- utile per correlare visivamente
    una bassa manipolabilita' con giunti specifici vicini a configurazioni
    "estese" (es. gomito quasi disteso, il caso classico di singolarita').
    I limiti (JOINT_LIMITS) sono disegnati solo come riferimento: qui NON
    sono il criterio di scelta (quello resta la manipolabilita').
    """
    if not candidati_riusciti:
        return

    if output_path is None:
        os.makedirs(MANIPULABILITY_RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(MANIPULABILITY_RESULTS_DIR, 'manipulability_joint_values.png')

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
    fig.suptitle('Angoli di giunto per candidato riuscito (solo riferimento -- il criterio qui e\' la manipolabilita\')')
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Grafico giunti-vs-candidati salvato in {output_path}")


def plot_ellissoide_manipolabilita(node, joint_names, positions, output_path=None):
    """
    Disegna l'ellissoide di manipolabilita' (solo la parte lineare/
    traslazionale, le 3 righe in alto dello Jacobiano -- la parte
    rotazionale esisterebbe anche lei ma non e' facilmente visualizzabile
    insieme a quella lineare in un solo disegno 3D) del candidato SCELTO
    (non di tutti i candidati -- serve solo per quello finale).

    Costruzione standard: gli assi dell'ellissoide sono gli autovettori di
    Jv @ Jv^T (Jv = parte lineare dello Jacobiano), le loro lunghezze sono
    le radici quadrate dei rispettivi autovalori -- direzioni in cui
    l'end effector puo' muoversi con piu' o meno "facilita'" a parita' di
    velocita' di giunto. Centrato sulla posizione VERA dell'end effector in
    quella configurazione, calcolata con la cinematica diretta (KDL) --
    non la posizione target comandata, quella raggiunta per davvero.
    """
    node.build_kdl_chain()

    nome_a_valore = dict(zip(joint_names, positions))
    valori = [nome_a_valore[nome] for nome in node._kdl_chain_joint_names]
    q = PyKDL.JntArray(len(valori))
    for i, v in enumerate(valori):
        q[i] = v

    jacobiano = PyKDL.Jacobian(node._kdl_chain.getNrOfJoints())
    node._kdl_jac_solver.JntToJac(q, jacobiano)
    J = np.array([[jacobiano[r, c] for c in range(jacobiano.columns())]
                  for r in range(jacobiano.rows())])
    Jv = J[:3, :]  # parte lineare (traslazionale) dello Jacobiano

    fk_solver = PyKDL.ChainFkSolverPos_recursive(node._kdl_chain)
    frame_end_effector = PyKDL.Frame()
    fk_solver.JntToCart(q, frame_end_effector)
    centro = np.array([frame_end_effector.p.x(), frame_end_effector.p.y(), frame_end_effector.p.z()])

    autovalori, autovettori = np.linalg.eigh(Jv @ Jv.T)  # simmetrica, eigh e' la scelta giusta e stabile
    autovalori = np.clip(autovalori, 0.0, None)  # arrotondamenti numerici potrebbero dare valori leggermente < 0
    semiassi = np.sqrt(autovalori)

    if output_path is None:
        os.makedirs(MANIPULABILITY_RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(MANIPULABILITY_RESULTS_DIR, 'manipulability_ellipsoid.png')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registra la proiezione 3d

    # Sfera unitaria campionata, poi deformata negli assi/lunghezze
    # dell'ellissoide (autovettori/autovalori) e traslata sul vero
    # end effector.
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 30)
    sfera_x = np.outer(np.cos(u), np.sin(v))
    sfera_y = np.outer(np.sin(u), np.sin(v))
    sfera_z = np.outer(np.ones_like(u), np.cos(v))
    punti_sfera = np.stack([sfera_x.ravel(), sfera_y.ravel(), sfera_z.ravel()])
    punti_ellissoide = autovettori @ np.diag(semiassi) @ punti_sfera
    ell_x = (punti_ellissoide[0] + centro[0]).reshape(sfera_x.shape)
    ell_y = (punti_ellissoide[1] + centro[1]).reshape(sfera_y.shape)
    ell_z = (punti_ellissoide[2] + centro[2]).reshape(sfera_z.shape)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(ell_x, ell_y, ell_z, color='tab:blue', alpha=0.5, edgecolor='none')
    ax.scatter(*centro, color='gold', s=60, zorder=5, label='end effector (candidato scelto)')

    raggio_max = max(semiassi.max(), 1e-3) * 1.3
    ax.set_xlim(centro[0] - raggio_max, centro[0] + raggio_max)
    ax.set_ylim(centro[1] - raggio_max, centro[1] + raggio_max)
    ax.set_zlim(centro[2] - raggio_max, centro[2] + raggio_max)
    try:
        ax.set_box_aspect([1, 1, 1])  # assi in scala 1:1:1, altrimenti l'ellissoide appare distorto
    except AttributeError:
        pass  # versioni vecchie di matplotlib senza set_box_aspect -- non blocca il resto

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(
        "Ellissoide di manipolabilita' (parte lineare) -- candidato scelto\n"
        f"semiassi: {semiassi[0]:.4f}, {semiassi[1]:.4f}, {semiassi[2]:.4f} m"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Ellissoide di manipolabilita' salvato in {output_path}")


def plot_candidati_manipolabilita(candidati_riusciti, migliore_indice_globale,
                                   output_path=None):
    """
    Un solo grafico (a differenza di joint_margin_optimizer, che ne fa 7 --
    uno per giunto: qui il criterio e' un unico numero per candidato, non un
    valore per giunto): manipolabilita' vs yaw, per ogni candidato RIUSCITO,
    con il candidato scelto evidenziato in oro.
    """
    if not candidati_riusciti:
        print("Nessun candidato riuscito, niente da plottare.")
        return

    if output_path is None:
        os.makedirs(MANIPULABILITY_RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(MANIPULABILITY_RESULTS_DIR, 'manipulability_candidates.png')

    import matplotlib
    matplotlib.use('Agg')  # nodo headless, nessun display
    import matplotlib.pyplot as plt

    xs = [c['yaw'] for c in candidati_riusciti]
    ys = [c['costo'] for c in candidati_riusciti]
    colors = ['gold' if c['indice_globale'] == migliore_indice_globale else 'tab:blue'
              for c in candidati_riusciti]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(xs, ys, c=colors, zorder=3, edgecolors='black', linewidths=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, label='singolarita\' (w=0)')
    ax.set_xlabel('yaw del candidato (rad)')
    ax.set_ylabel("indice di manipolabilita' w(q)")
    ax.set_title("Manipolabilita' per candidato riuscito (oro = scelto)")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Grafico manipolabilita'-vs-candidati salvato in {output_path}")


def main():
    rclpy.init()
    node = MoveGroupClient()

    print("Aspetto l'URDF su /robot_description...")
    if not node.wait_for_topics(
        [lambda n: n.robot_description_xml], timeout_sec=real_seconds_for(15.0),
    ):
        print("URDF non ricevuto in tempo, esco.")
        node.destroy_node()
        rclpy.shutdown()
        return
    node.build_kdl_chain()

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
        else:
            codice = risultato.result.error_code.val if risultato else "nessuna risposta"
            print(f"Esecuzione fallita (error_code={codice}).")
    else:
        print("Nessuna soluzione trovata per nessun valore di yaw.")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
