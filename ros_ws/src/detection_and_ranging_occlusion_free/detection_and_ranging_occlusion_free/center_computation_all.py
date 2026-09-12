#!/usr/bin/env python3
"""
Nodo di center computation multi-oggetto: per ogni classe in RADIUS_BY_CLASS,
riceve il punto di superficie rilevato e ne stima il centro dell'asse
verticale spostandolo del raggio noto dell'oggetto lungo la componente
orizzontale del raggio camera->punto.

La correzione e' valida solo in un frame con asse Z verticale (TARGET_FRAME,
'base_footprint'): il punto viene prima trasformato li' via TF, e solo dopo
corretto -- l'ordine inverso sposterebbe il punto nella direzione sbagliata.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker

import tf2_ros
import tf2_geometry_msgs  # noqa: F401
from tf2_geometry_msgs import do_transform_point


RADIUS_BY_CLASS = {
    'coke can':      0.04,
    'pringles can':  0.04,
    'biscuits pack': 0.03,
}

COLOR_BY_CLASS = {
    'coke can':      (0.80, 0.16, 0.80),
    'pringles can':  (0.16, 0.50, 0.80),
    'biscuits pack': (1.00, 0.67, 0.00),
}
DEFAULT_AXIS_COLOR = (0.0, 1.0, 0.0)

AXIS_MARKER_LENGTH = 0.40
TARGET_FRAME = 'base_footprint'


def topic_slug(class_name):
    """'coke can' -> 'coke_can'. Deve restare identica a quella in rt_object_detection_node_all.py."""
    return class_name.replace(' ', '_')


class CenterComputationAllNode(Node):
    def __init__(self):
        super().__init__('center_computation_all_node')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.center_pubs = {}
        self.center_base_footprint_pubs = {}
        self.axis_marker_pubs = {}
        self.subscriptions_by_class = {}

        self.last_stamp = {}
        self.instance_counter = {}

        for class_name in RADIUS_BY_CLASS:
            slug = topic_slug(class_name)

            self.center_pubs[class_name] = self.create_publisher(
                PointStamped, f'centers_all/{slug}_center', 10
            )
            self.center_base_footprint_pubs[class_name] = self.create_publisher(
                PointStamped, f'centers_all/{slug}_center_base_footprint', 10
            )
            self.axis_marker_pubs[class_name] = self.create_publisher(
                Marker, f'centers_all/{slug}_axis', 10
            )

            self.subscriptions_by_class[class_name] = self.create_subscription(
                PointStamped,
                f'yolo_all/{slug}_position',
                lambda msg, name=class_name: self.position_callback(msg, name),
                10,
            )

            self.last_stamp[class_name] = None
            self.instance_counter[class_name] = 0

        self.get_logger().info(
            f'Center computation multi-oggetto avviata sulle classi: {", ".join(RADIUS_BY_CLASS)}'
        )

    def position_callback(self, msg, class_name):
        radius = RADIUS_BY_CLASS[class_name]
        marker_id = self.next_marker_id(class_name, msg.header.stamp)

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

        cam_x = transform.transform.translation.x
        cam_y = transform.transform.translation.y

        dx = px - cam_x
        dy = py - cam_y
        horizontal_norm = math.hypot(dx, dy)
        if horizontal_norm < 1e-6:
            self.get_logger().warn(
                f'[{class_name}] Punto troppo vicino alla proiezione verticale '
                'della camera, salto la correzione.',
                throttle_duration_sec=2.0
            )
            return

        ux = dx / horizontal_norm
        uy = dy / horizontal_norm

        center_in_target = PointStamped()
        center_in_target.header.frame_id = TARGET_FRAME
        center_in_target.header.stamp = msg.header.stamp
        center_in_target.point.x = px + radius * ux
        center_in_target.point.y = py + radius * uy
        center_in_target.point.z = pz

        try:
            inverse_transform = self.tf_buffer.lookup_transform(
                msg.header.frame_id, TARGET_FRAME, rclpy.time.Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            self.get_logger().warn(
                f'TF {msg.header.frame_id} <- {TARGET_FRAME} non disponibile: {error}',
                throttle_duration_sec=2.0
            )
            return

        center_in_source = do_transform_point(center_in_target, inverse_transform)

        self.center_pubs[class_name].publish(center_in_source)
        self.center_base_footprint_pubs[class_name].publish(center_in_target)
        self.publish_axis_marker(center_in_target, class_name, marker_id)

    def next_marker_id(self, class_name, stamp):
        """Id progressivo dell'istanza dentro il frame corrente (riparte da 0 quando cambia header.stamp)."""
        stamp_key = (stamp.sec, stamp.nanosec)
        if self.last_stamp[class_name] != stamp_key:
            self.last_stamp[class_name] = stamp_key
            self.instance_counter[class_name] = 0
        else:
            self.instance_counter[class_name] += 1
        return self.instance_counter[class_name]

    def publish_axis_marker(self, center: PointStamped, class_name, marker_id):
        cx = center.point.x
        cy = center.point.y
        cz = center.point.z

        marker = Marker()
        marker.header.frame_id = TARGET_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f'{topic_slug(class_name)}_center_axis'
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        red, green, blue = COLOR_BY_CLASS.get(class_name, DEFAULT_AXIS_COLOR)
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 1.0
        marker.points = [
            Point(x=cx, y=cy, z=cz - AXIS_MARKER_LENGTH / 2.0),
            Point(x=cx, y=cy, z=cz + AXIS_MARKER_LENGTH / 2.0),
        ]

        self.axis_marker_pubs[class_name].publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = CenterComputationAllNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
