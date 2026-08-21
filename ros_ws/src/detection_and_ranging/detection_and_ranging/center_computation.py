#!/usr/bin/env python3
"""
Center computation node.

Riceve il punto sulla superficie tangente della lattina (yolo/coke_can_position,
il centro del bounding box proiettato con la depth) e ne stima il centro
dell'asse verticale del cilindro.

IMPORTANTE: la correzione (spostare il punto di un raggio noto lungo la
componente ORIZZONTALE del raggio camera->punto) e' un ragionamento valido
solo in un frame dove l'asse Z e' davvero verticale (allineato con la
gravita'), tipo TARGET_FRAME ('base_footprint'). Il frame della camera NON
va bene: la camera e' inclinata (guarda il tavolo dall'alto), quindi il suo
piano x,y locale non e' affatto il piano orizzontale vero. Per questo il
punto viene prima trasformato in TARGET_FRAME via TF, e SOLO DOPO si applica
la correzione -- farlo nell'ordine inverso (come nella prima versione di
questo nodo) sposta il punto nella direzione sbagliata.

  - x, y: corretti spostando il punto (gia' in TARGET_FRAME) di un raggio
    noto (CAN_RADIUS) lungo la componente orizzontale della direzione
    camera->punto. La posizione della camera in TARGET_FRAME si ottiene
    gratis dalla traduzione della trasformazione TF stessa.
  - z: lasciata invariata. Lo spostamento orizzontale non cambia l'altezza, e
    la z misurata e' gia' un punto valido sull'asse vero -- niente bisogno di
    conoscere l'altezza del tavolo o altra informazione specifica della scena.

CAN_RADIUS e' un prior sull'OGGETTO (non cambia se cambia la scena), quindi
resta valido su tavoli/mondi diversi -- a differenza di un'altezza del tavolo
hardcoded, che sarebbe un prior sulla SCENA e non generalizzerebbe.

Pubblica il centro corretto (gia' in TARGET_FRAME) su cokecan_center, e un
marker RViz che disegna un segmento verticale passante per quel centro
(invece del solo pallino).
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registra il supporto a PointStamped per do_transform_point)
from tf2_geometry_msgs import do_transform_point


CAN_RADIUS = 0.04    # raggio noto della lattina (prior sull'oggetto)
AXIS_MARKER_LENGTH = 0.40   # lunghezza del segmento disegnato in RViz (solo visualizzazione,
                              # volutamente piu' lungo della lattina vera per essere ben visibile)
TARGET_FRAME = 'base_footprint'   # frame con asse Z verticale (gravita'), per disegnare l'asse correttamente


class CenterComputationNode(Node):
    def __init__(self):
        super().__init__('center_computation_node')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.center_pub = self.create_publisher(PointStamped, 'cokecan_center', 10)
        self.axis_marker_pub = self.create_publisher(Marker, 'cokecan_center_axis', 10)

        self.position_sub = self.create_subscription(
            PointStamped, 'yolo/coke_can_position', self.position_callback, 10
        )

        self.get_logger().info('Center computation node started.')

    def position_callback(self, msg):
        # Passo 1: porta il punto grezzo in TARGET_FRAME (asse Z verticale
        # vero) PRIMA di ragionare in termini di "piano orizzontale".
        try:
            transform = self.tf_buffer.lookup_transform(
                TARGET_FRAME, msg.header.frame_id, rclpy.time.Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            self.get_logger().warn(
                f'TF {TARGET_FRAME} <- {msg.header.frame_id} non disponibile: {error}',
                throttle_duration_sec=2.0
            )
            return

        point_in_target = do_transform_point(msg, transform)
        px = point_in_target.point.x
        py = point_in_target.point.y
        pz = point_in_target.point.z

        # Posizione della camera in TARGET_FRAME: e' la traslazione della
        # trasformazione stessa (l'origine del frame camera, espressa in
        # TARGET_FRAME).
        cam_x = transform.transform.translation.x
        cam_y = transform.transform.translation.y

        # Passo 2: ORA il ragionamento sul piano orizzontale e' corretto,
        # perche' siamo in un frame dove z e' davvero verticale.
        dx = px - cam_x
        dy = py - cam_y
        horizontal_norm = math.hypot(dx, dy)
        if horizontal_norm < 1e-6:
            self.get_logger().warn(
                "Punto troppo vicino alla proiezione verticale della camera, salto la correzione."
            )
            return

        ux = dx / horizontal_norm
        uy = dy / horizontal_norm

        center = PointStamped()
        center.header.frame_id = TARGET_FRAME
        center.header.stamp = msg.header.stamp
        center.point.x = px + CAN_RADIUS * ux
        center.point.y = py + CAN_RADIUS * uy
        center.point.z = pz

        self.center_pub.publish(center)
        self.publish_axis_marker(center)

    def publish_axis_marker(self, center: PointStamped):
        cx = center.point.x
        cy = center.point.y
        cz = center.point.z

        marker = Marker()
        marker.header.frame_id = TARGET_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'cokecan_center_axis'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01  # spessore della linea (m)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.points = [
            Point(x=cx, y=cy, z=cz - AXIS_MARKER_LENGTH / 2.0),
            Point(x=cx, y=cy, z=cz + AXIS_MARKER_LENGTH / 2.0),
        ]

        self.axis_marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = CenterComputationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
