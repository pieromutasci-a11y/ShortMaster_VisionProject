#!/usr/bin/env python3
"""
Center computation node.

Riceve il punto sulla superficie tangente della lattina (yolo/coke_can_position,
il centro del bounding box proiettato con la depth) e ne stima il centro
dell'asse verticale del cilindro:

  - x, y: corretti spostando il punto misurato di un raggio noto (CAN_RADIUS)
    lungo la componente ORIZZONTALE del raggio ottico camera->punto. La camera
    e' l'origine nel proprio frame, quindi quella direzione e' semplicemente
    (x, y) del punto stesso normalizzato in 2D (z esclusa: il problema di
    trovare l'asse di un cilindro verticale e' geometricamente un problema 2D,
    la componente z del raggio non vi entra).
  - z: lasciata invariata. Lo spostamento orizzontale non cambia l'altezza, e
    la z misurata e' gia' un punto valido sull'asse vero -- niente bisogno di
    conoscere l'altezza del tavolo o altra informazione specifica della scena.

CAN_RADIUS e' un prior sull'OGGETTO (non cambia se cambia la scena), quindi
resta valido su tavoli/mondi diversi -- a differenza di un'altezza del tavolo
hardcoded, che sarebbe un prior sulla SCENA e non generalizzerebbe.

Pubblica il centro corretto (yolo/coke_can_position) su cokecan_center, e un
marker RViz che disegna un segmento verticale passante per quel centro (invece
del solo pallino) -- per essere davvero verticale (allineato con la gravita',
non con l'asse Z del frame camera, che e' inclinato) il marker viene disegnato
in TARGET_FRAME dopo una trasformazione TF del punto.
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
CAN_HEIGHT = 0.15     # altezza nota della lattina, usata solo per la lunghezza del marker
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
        x, y, z = msg.point.x, msg.point.y, msg.point.z

        horizontal_norm = math.hypot(x, y)
        if horizontal_norm < 1e-6:
            self.get_logger().warn(
                "Punto troppo vicino all'asse ottico della camera, salto la correzione."
            )
            return

        # Versore orizzontale (2D, solo x/y) della direzione camera->punto.
        ux = x / horizontal_norm
        uy = y / horizontal_norm

        center = PointStamped()
        center.header = msg.header
        center.point.x = x + CAN_RADIUS * ux
        center.point.y = y + CAN_RADIUS * uy
        center.point.z = z

        self.center_pub.publish(center)
        self.publish_axis_marker(center)

    def publish_axis_marker(self, center: PointStamped):
        try:
            transform = self.tf_buffer.lookup_transform(
                TARGET_FRAME, center.header.frame_id, rclpy.time.Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            self.get_logger().warn(
                f'TF {TARGET_FRAME} <- {center.header.frame_id} non disponibile: {error}',
                throttle_duration_sec=2.0
            )
            return

        center_in_target = do_transform_point(center, transform)

        cx = center_in_target.point.x
        cy = center_in_target.point.y
        cz = center_in_target.point.z

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
            Point(x=cx, y=cy, z=cz - CAN_HEIGHT / 2.0),
            Point(x=cx, y=cy, z=cz + CAN_HEIGHT / 2.0),
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
