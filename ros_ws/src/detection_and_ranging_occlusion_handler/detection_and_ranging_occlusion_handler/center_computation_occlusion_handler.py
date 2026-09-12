#!/usr/bin/env python3
"""
Nodo di center computation per lo stack con gestione occlusioni: riceve, per
ogni oggetto tracciato, i 3 punti di superficie campionati da
rt_object_detection_node_occlusion_handler.py e ne ricava il centro
dell'asse verticale fittando la circonferenza passante per quei 3 punti
(circocentro, soluzione esatta), invece di assumere un raggio noto a priori.

Il fit avviene nel piano XY di TARGET_FRAME ('base_footprint', asse Z
verticale), non nel frame camera (inclinato): i punti vengono prima
trasformati via TF. Se i 3 punti sono troppo allineati (area del triangolo
o raggio misurato fuori soglia, tipico di un oggetto molto occluso), il
frame viene scartato invece di pubblicare un centro inaffidabile.

Topic identici a center_computation_all.py, cosi' pose_optimizer non
richiede modifiche per passare da uno stack all'altro.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker

import tf2_ros
import tf2_geometry_msgs  # noqa: F401
from tf2_geometry_msgs import do_transform_point

from tiago_vision_msgs.msg import TrackedObjectsArray


# Raggio nominale per classe, usato solo come riferimento per il sanity-check sul raggio misurato dal fit.
RADIUS_BY_CLASS = {
    'coke can':      0.04,
    'pringles can':  0.04,
    'biscuits pack': 0.029,
}

# Fattore di tolleranza tra raggio misurato e nominale prima di scartare il frame come inaffidabile.
RADIUS_SANITY_FACTOR = 2.0

# Area minima (m^2) del triangolo dei 3 punti perche' il fit sia considerato affidabile.
MIN_TRIANGLE_AREA_M2 = 1e-5

COLOR_BY_CLASS = {
    'coke can':      (0.80, 0.16, 0.80),
    'pringles can':  (0.16, 0.50, 0.80),
    'biscuits pack': (1.00, 0.67, 0.00),
}
DEFAULT_AXIS_COLOR = (0.0, 1.0, 0.0)

AXIS_MARKER_LENGTH = 0.40
TARGET_FRAME = 'base_footprint'


def topic_slug(class_name):
    """'coke can' -> 'coke_can'. Deve restare identica a quella usata nello stack _all."""
    return class_name.replace(' ', '_')


def fit_circle_2d(p1, p2, p3):
    """Centro della circonferenza passante per 3 punti (x, y) nel piano (circocentro, formula chiusa).
    Ritorna sempre (cx, cy, area_triangolo): il chiamante scarta il risultato se l'area e' troppo piccola."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    area = abs(d) / 4.0

    if abs(d) < 1e-12:
        return None, None, area

    sq1 = x1 * x1 + y1 * y1
    sq2 = x2 * x2 + y2 * y2
    sq3 = x3 * x3 + y3 * y3

    cx = (sq1 * (y2 - y3) + sq2 * (y3 - y1) + sq3 * (y1 - y2)) / d
    cy = (sq1 * (x3 - x2) + sq2 * (x1 - x3) + sq3 * (x2 - x1)) / d

    return cx, cy, area


class CenterComputationOcclusionHandlerNode(Node):
    def __init__(self):
        super().__init__('center_computation_occlusion_handler_node')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.center_pubs = {}
        self.center_base_footprint_pubs = {}
        self.axis_marker_pubs = {}

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

        self.tracked_objects_sub = self.create_subscription(
            TrackedObjectsArray,
            'yolo/tracked_objects_points',
            self.tracked_objects_callback,
            10,
        )

        self.get_logger().info(
            f'Center computation (occlusion handler) avviata sulle classi: {", ".join(RADIUS_BY_CLASS)}'
        )

    def tracked_objects_callback(self, msg):
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

        for entry in msg.objects:
            self.process_object(entry, msg.header, transform, inverse_transform)

    def process_object(self, entry, header, transform, inverse_transform):
        class_name = entry.class_name
        if class_name not in RADIUS_BY_CLASS:
            return

        if len(entry.points) != 3:
            self.get_logger().warn(
                f'[{class_name}] Attesi 3 punti di superficie, ricevuti {len(entry.points)} -- salto.',
                throttle_duration_sec=5.0
            )
            return

        points_in_target = []
        for point in entry.points:
            point_stamped = PointStamped()
            point_stamped.header = header
            point_stamped.point = point
            transformed = do_transform_point(point_stamped, transform)
            points_in_target.append(transformed.point)

        xy_points = [(p.x, p.y) for p in points_in_target]
        cx, cy, area = fit_circle_2d(*xy_points)

        if cx is None or area < MIN_TRIANGLE_AREA_M2:
            self.get_logger().warn(
                f'[{class_name}] Punti di superficie troppo allineati/coincidenti '
                f'(area triangolo {area:.6f} m^2) -- centro non affidabile, salto il frame.',
                throttle_duration_sec=5.0
            )
            return

        cz = sum(p.z for p in points_in_target) / 3.0

        measured_radius = math.hypot(xy_points[0][0] - cx, xy_points[0][1] - cy)
        nominal_radius = RADIUS_BY_CLASS[class_name]
        if not (nominal_radius / RADIUS_SANITY_FACTOR <= measured_radius <= nominal_radius * RADIUS_SANITY_FACTOR):
            self.get_logger().warn(
                f'[{class_name}] Raggio misurato ({measured_radius:.3f} m) molto diverso '
                f'dal nominale ({nominal_radius:.3f} m) -- fit inaffidabile (probabile arco '
                f'visibile troppo stretto), salto il frame.',
                throttle_duration_sec=5.0
            )
            return

        center_in_target = PointStamped()
        center_in_target.header.frame_id = TARGET_FRAME
        center_in_target.header.stamp = header.stamp
        center_in_target.point.x = cx
        center_in_target.point.y = cy
        center_in_target.point.z = cz

        center_in_source = do_transform_point(center_in_target, inverse_transform)

        self.center_pubs[class_name].publish(center_in_source)
        self.center_base_footprint_pubs[class_name].publish(center_in_target)
        self.publish_axis_marker(center_in_target, class_name)

    def publish_axis_marker(self, center: PointStamped, class_name):
        cx = center.point.x
        cy = center.point.y
        cz = center.point.z

        marker = Marker()
        marker.header.frame_id = TARGET_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f'{topic_slug(class_name)}_center_axis'
        marker.id = 0
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
    node = CenterComputationOcclusionHandlerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
