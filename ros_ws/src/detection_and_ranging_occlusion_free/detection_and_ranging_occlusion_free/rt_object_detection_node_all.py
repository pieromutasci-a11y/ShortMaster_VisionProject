"""
Nodo di detection multi-oggetto in tempo reale: rileva tutte le istanze di
tutte le classi in TRACKED_CLASSES e ne pubblica la posizione 3D, una per
istanza, su un topic per classe (yolo_all/<classe>_position).

La depth di ogni oggetto e' stimata come mediana di una patch centrale del
bounding box, in isolamento dagli altri oggetti rilevati: questo nodo
assume scene senza occlusioni reciproche fra gli oggetti tracciati.
"""

import threading
import time
import traceback
from collections import deque
from threading import Lock
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor

from message_filters import Subscriber, ApproximateTimeSynchronizer

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO


COKE_CLASS_NAME = 'coke can'
PRINGLES_CLASS_NAME = 'pringles can'
BISCUITS_CLASS_NAME = 'biscuits pack'
DINNER_TABLE_CLASS_NAME = 'dinner table'

TRACKED_CLASSES = [
    COKE_CLASS_NAME,
    PRINGLES_CLASS_NAME,
    BISCUITS_CLASS_NAME,
    DINNER_TABLE_CLASS_NAME,
]

DETECTION_CONFIDENCE_THRESHOLD = 0.6

MIN_RANGE = 0.3
MAX_RANGE = 3.0

DEPTH_PATCH_FRACTION = 0.3

MODEL_WEIGHTS_PATH = "/home/user/ros_workspace/src/vision_pipeline/models/wandb/runs/sqkfh2ka/training/xl1874f6/weights/best.pt"
# MODEL_WEIGHTS_PATH = "/home/user/ros_workspace/src/vision_pipeline/models/small_omogeneous_dataset_model_best.pt"
# MODEL_WEIGHTS_PATH = "/home/user/ros_workspace/src/vision_pipeline/models/small_eterogeneous_dataset_model_best.pt"
# MODEL_WEIGHTS_PATH = "/home/user/ros_workspace/src/vision_pipeline/models/segmentation_model_best.pt"


def topic_slug(class_name):
    """'coke can' -> 'coke_can'. Deve restare identica a quella in center_computation_all.py."""
    return class_name.replace(' ', '_')


def normalize_yolo_class_name(raw_class_name):
    """Uniforma il nome classe letto da model.names prima del confronto con TRACKED_CLASSES (underscore -> spazio, spazi multipli -> uno solo)."""
    return ' '.join(raw_class_name.replace('_', ' ').split())


class BoundingBoxPixels:
    """Bounding box in coordinate pixel della depth image, con la depth stimata."""

    def __init__(self, x1, y1, x2, y2, class_name, confidence):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.class_name = class_name
        self.confidence = confidence
        self.depth = None


class RtObjectDetectionAllNode(Node):
    """
    Subscribes to synchronised RGB and depth images, runs YOLOv8 detection on
    the RGB frame, and publishes:
      - an annotated RGB frame, for visualisation/debugging
      - the estimated 3D position of every detected object of every tracked
        class, one PointStamped per instance, on a per-class topic
    """

    def __init__(self):
        super().__init__('rt_object_detection_all_node')

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.frame_buffer = deque(maxlen=1)
        self.buffer_lock = Lock()

        self.cv_bridge = CvBridge()

        self.detection_model = YOLO(MODEL_WEIGHTS_PATH)

        self.focal_length_x = None
        self.focal_length_y = None
        self.principal_point_x = None
        self.principal_point_y = None
        self.intrinsics_lock = Lock()

        self.color_image_sub = Subscriber(
            self, Image, '/head_front_camera/color/image_raw', qos_profile=camera_qos
        )
        self.depth_image_sub = Subscriber(
            self, Image, '/head_front_camera/depth/image_rect_raw', qos_profile=camera_qos
        )

        self.time_synchronizer = ApproximateTimeSynchronizer(
            [self.color_image_sub, self.depth_image_sub],
            queue_size=10,
            slop=0.01
        )
        self.time_synchronizer.registerCallback(self.synchronized_frames_callback)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/head_front_camera/depth/camera_info',
            self.camera_info_callback,
            qos_profile=camera_qos
        )

        self.annotated_frame_pub = self.create_publisher(Image, 'yolo_all/annotated_frame', 10)

        self.position_pubs = {
            class_name: self.create_publisher(
                PointStamped, f'yolo_all/{topic_slug(class_name)}_position', 10
            )
            for class_name in TRACKED_CLASSES
        }

        self.worker_running = True
        self.worker_thread = threading.Thread(target=self.detection_worker_loop, daemon=True)
        self.worker_thread.start()

        self.get_logger().info(
            f'Detection multi-oggetto avviata sulle classi: {", ".join(TRACKED_CLASSES)}'
        )

    def camera_info_callback(self, camera_info_msg):
        """Store the camera intrinsics extracted from the CameraInfo K matrix."""
        with self.intrinsics_lock:
            self.focal_length_x = camera_info_msg.k[0]
            self.focal_length_y = camera_info_msg.k[4]
            self.principal_point_x = camera_info_msg.k[2]
            self.principal_point_y = camera_info_msg.k[5]

    def synchronized_frames_callback(self, rgb_msg, depth_msg):
        """Store the latest synchronised RGB/depth pair for the worker thread to process."""
        with self.buffer_lock:
            self.frame_buffer.append((rgb_msg, depth_msg))

    def detection_worker_loop(self):
        """
        Main detection loop, run on a background thread. Continuously pulls the
        latest RGB/depth pair from the buffer, runs YOLO detection, and publishes
        the annotated frame plus the 3D position of every detected object.
        """
        while rclpy.ok() and self.worker_running:
            frame_pair = None
            try:
                with self.buffer_lock:
                    if self.frame_buffer:
                        frame_pair = self.frame_buffer.pop()

                if frame_pair is None:
                    time.sleep(0.01)
                    continue

                rgb_msg, depth_msg = frame_pair
                self.process_frame_pair(rgb_msg, depth_msg)

            except Exception as error:
                self.get_logger().error(
                    f"Error while processing frame: {type(error).__name__}: {error}\n"
                    f"{traceback.format_exc()}"
                )

    def process_frame_pair(self, rgb_msg, depth_msg):
        """Run detection on one RGB/depth pair and publish every detected object."""

        rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

        with self.intrinsics_lock:
            focal_length_x = self.focal_length_x
            focal_length_y = self.focal_length_y
            principal_point_x = self.principal_point_x
            principal_point_y = self.principal_point_y

        if None in (focal_length_x, focal_length_y, principal_point_x, principal_point_y):
            self.get_logger().warn(
                "Camera intrinsics not yet received, skipping frame.",
                throttle_duration_sec=5.0
            )
            return

        detection_results = self.detection_model(rgb_image, verbose=False)[0]

        annotated_image = detection_results.plot()
        annotated_image_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, encoding='passthrough')
        annotated_image_msg.encoding = 'bgr8'
        annotated_image_msg.header = rgb_msg.header
        self.annotated_frame_pub.publish(annotated_image_msg)

        rgb_height, rgb_width = rgb_image.shape[:2]
        depth_height, depth_width = depth_image.shape[:2]
        scale_x = depth_width / rgb_width
        scale_y = depth_height / rgb_height

        boxes = []
        for bounding_box in detection_results.boxes:
            class_id = int(bounding_box.cls[0])
            class_name = normalize_yolo_class_name(self.detection_model.names[class_id])
            confidence_score = float(bounding_box.conf[0])

            if class_name not in TRACKED_CLASSES or confidence_score < DETECTION_CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = bounding_box.xyxy[0]
            boxes.append(BoundingBoxPixels(
                x1=self.clamp(int(float(x1) * scale_x), 0, depth_width - 1),
                y1=self.clamp(int(float(y1) * scale_y), 0, depth_height - 1),
                x2=self.clamp(int(float(x2) * scale_x), 0, depth_width - 1),
                y2=self.clamp(int(float(y2) * scale_y), 0, depth_height - 1),
                class_name=class_name,
                confidence=confidence_score,
            ))

        if not boxes:
            return

        for box in boxes:
            box.depth = self.estimate_box_depth(box, depth_image)
            if box.depth is None:
                continue

            position = self.depth_to_3d_point(
                box, focal_length_x, focal_length_y,
                principal_point_x, principal_point_y
            )
            if position is None:
                continue

            point_x, point_y, point_z = position
            self.publish_object_position(
                box.class_name, point_x, point_y, point_z, rgb_msg.header
            )

    def valid_depth_mask(self, depth_region):
        """Elimina 0, NaN, Inf e valori fuori dal range operativo della camera depth."""
        mask = np.isfinite(depth_region)
        mask &= (depth_region >= MIN_RANGE)
        mask &= (depth_region <= MAX_RANGE)
        return mask

    def estimate_box_depth(self, box, depth_image):
        """Mediana dei valori di depth validi in una patch centrale del box.
        None se non resta nessun pixel valido."""
        patch = self.central_patch(box)
        if patch is None:
            return None

        px1, py1, px2, py2 = patch
        depth_region = depth_image[py1:py2, px1:px2]
        if depth_region.size == 0:
            return None

        mask = self.valid_depth_mask(depth_region)
        valid_values = depth_region[mask]
        if valid_values.size == 0:
            return self.center_pixel_depth(box, depth_image)

        return float(np.median(valid_values))

    def central_patch(self, box):
        """Rettangolo centrale del bounding box (DEPTH_PATCH_FRACTION dei suoi lati)."""
        width = box.x2 - box.x1
        height = box.y2 - box.y1
        if width <= 0 or height <= 0:
            return None

        half_patch_w = max(1, int(width * DEPTH_PATCH_FRACTION / 2))
        half_patch_h = max(1, int(height * DEPTH_PATCH_FRACTION / 2))

        center_x = (box.x1 + box.x2) // 2
        center_y = (box.y1 + box.y2) // 2

        return (
            max(box.x1, center_x - half_patch_w),
            max(box.y1, center_y - half_patch_h),
            min(box.x2, center_x + half_patch_w) + 1,
            min(box.y2, center_y + half_patch_h) + 1,
        )

    def center_pixel_depth(self, box, depth_image):
        """Depth del solo pixel centrale del box (fallback), o None se non valida."""
        center_x = (box.x1 + box.x2) // 2
        center_y = (box.y1 + box.y2) // 2
        depth_value = float(depth_image[center_y, center_x])

        if depth_value <= 0.0 or depth_value != depth_value:
            return None
        return depth_value

    def depth_to_3d_point(self, box, focal_length_x, focal_length_y,
                          principal_point_x, principal_point_y):
        """Converte centro del box + depth in una posizione 3D nel frame ottico della camera (modello pinhole)."""
        if box.depth is None:
            return None

        pixel_x = (box.x1 + box.x2) // 2
        pixel_y = (box.y1 + box.y2) // 2

        point_z = box.depth
        point_x = (pixel_x - principal_point_x) * point_z / focal_length_x
        point_y = (pixel_y - principal_point_y) * point_z / focal_length_y

        return point_x, point_y, point_z

    def publish_object_position(self, class_name, point_x, point_y, point_z, header):
        """Publish one detected object's 3D position on its class's topic."""
        position_msg = PointStamped()
        position_msg.header = header
        position_msg.point.x = point_x
        position_msg.point.y = point_y
        position_msg.point.z = point_z
        self.position_pubs[class_name].publish(position_msg)

    @staticmethod
    def clamp(value, lower, upper):
        return min(max(value, lower), upper)

    def destroy_node(self):
        """Ensure the worker thread stops cleanly before the node is destroyed."""
        self.worker_running = False
        self.worker_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RtObjectDetectionAllNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
