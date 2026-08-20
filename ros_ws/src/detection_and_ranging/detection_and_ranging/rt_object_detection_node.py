"""
Real-time object detection node.

Detects a target object (e.g. a Coca-Cola can) in an RGB image stream using
YOLOv8, fuses the detection with a synchronised depth image to estimate the
object's 3D position, and publishes both an annotated frame (for debugging)
and the object's position as a geometry_msgs/PointStamped message.


ROBA DA FARE:
Controlla la risoluzione dell'immagine di YOLO
"""

# Standard library imports
import threading
import time
from collections import deque
from threading import Lock
import numpy as np

# Third-party / ROS imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor

from message_filters import Subscriber, ApproximateTimeSynchronizer

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO


# Name of the class (as defined in the YOLO model's names dict) that we want to track (MODIFICATO)
COKE_CLASS_NAME = 'coke_cane'
BISCUITS_CLASS_NAME = 'biscuits_pack'
PRINGLES_CLASS_NAME = 'pringles_can'
INTERESTED_OBJ = [COKE_CLASS_NAME, BISCUITS_CLASS_NAME, PRINGLES_CLASS_NAME]


TARGET_CLASS_NAME = COKE_CLASS_NAME
DETECTION_CONFIDENCE_THRESHOLD = 0.6

# Valori di depth della Intel RealSense D435 montata sul Tiago Pro
MIN_RANGE = 0.3
MAX_RANGE = 3.0

# Best model dello sweep W&B (run xl1874f6, vedi vision_pipeline/vision_pipeline/yolo/evaluate.py)
MODEL_WEIGHTS_PATH = "/home/user/ros_workspace/src/vision_pipeline/models/wandb/runs/sqkfh2ka/training/xl1874f6/weights/best.pt"


class RtObjectDetectionNode(Node):
    """
    Subscribes to synchronised RGB and depth images, runs YOLOv8 detection on
    the RGB frame, and publishes:
      - an annotated RGB frame, for visualisation/debugging
      - the estimated 3D position of the target object, once found
    """

    def __init__(self):
        super().__init__('rt_object_detection_node')

        # QoS profile matching typical camera driver settings (best effort,
        # shallow history — we don't need every historical frame).
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Buffer holding only the most recent synchronised RGB/depth pair.
        # Using maxlen=1 means older, unprocessed frames are automatically
        # dropped, which keeps the pipeline running in real time rather than
        # building up a backlog if detection is briefly slower than the camera.
        self.frame_buffer = deque(maxlen=1)
        self.buffer_lock = Lock()

        # Bridge used to convert between ROS Image messages and OpenCV/NumPy arrays
        self.cv_bridge = CvBridge()

        # Load the YOLOv8 model once at start-up
        self.detection_model = YOLO(MODEL_WEIGHTS_PATH)

        # Camera intrinsics, populated once a CameraInfo message arrives.
        # Left as None initially so we can skip processing until they're available.
        self.focal_length_x = None
        self.focal_length_y = None
        self.principal_point_x = None
        self.principal_point_y = None
        self.intrinsics_lock = Lock()

        # --- Subscribers ---

        # RGB and depth image subscribers, synchronised by (approximate) timestamp
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

        # CameraInfo carries the intrinsic matrix K, needed to convert pixel
        # coordinates + depth into a 3D point. Usually published at a low rate.
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/head_front_camera/depth/camera_info',
            self.camera_info_callback,
            qos_profile=camera_qos
        )

        # --- Publishers ---

        self.annotated_frame_pub = self.create_publisher(Image, 'yolo/annotated_frame', 10)
        self.object_position_pub = self.create_publisher(PointStamped, 'yolo/coke_can_position', 10)

        # --- Worker thread ---

        # Detection runs on a separate thread so that YOLO inference (which can
        # be relatively slow) never blocks the ROS executor's subscription callbacks.
        self.worker_running = True
        self.worker_thread = threading.Thread(target=self.detection_worker_loop, daemon=True)
        self.worker_thread.start()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def detection_worker_loop(self):
        """
        Main detection loop, run on a background thread. Continuously pulls the
        latest RGB/depth pair from the buffer, runs YOLO detection, and publishes
        the annotated frame plus the target object's 3D position, if found.
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
                self.get_logger().error(f"Error while processing frame: {error}")

    def process_frame_pair(self, rgb_msg, depth_msg):
        """Run detection on one RGB/depth pair and publish the results."""

        rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

        # NOTE: this assumes the depth driver publishes float32 metres (32FC1).
        # Some drivers instead publish uint16 millimetres (16UC1) — if depth
        # values look implausible, check the topic's actual encoding and adjust
        # accordingly (e.g. request '16UC1' and divide by 1000.0).
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

        # Publish the annotated frame for visualisation/debugging
        annotated_image = detection_results.plot()
        annotated_image_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        annotated_image_msg.header = rgb_msg.header
        self.annotated_frame_pub.publish(annotated_image_msg)

        image_height, image_width = depth_image.shape[:2]

        for bounding_box in detection_results.boxes:
            class_id = int(bounding_box.cls[0])
            class_name = self.detection_model.names[class_id]
            confidence_score = float(bounding_box.conf[0])

            if class_name != TARGET_CLASS_NAME or confidence_score < DETECTION_CONFIDENCE_THRESHOLD:
                continue

            position = self.estimate_object_position(
                bounding_box, depth_image, image_width, image_height,
                focal_length_x, focal_length_y, principal_point_x, principal_point_y
            )

            if position is None:
                continue

            point_x, point_y, point_z = position
            self.publish_object_position(point_x, point_y, point_z, rgb_msg.header)

            # Only handle the first valid detection above the confidence threshold
            break

    # NUOVA FUNZIONE
    def valid_depth_mask(self, depth_region):
        """
            Funzione che elimina gli 0, NaN, Inf e valori fuori dal range operativo della camera depth del tiago pro
        """
        mask = np.isfinite(depth_region)
        mask &= (depth_region >= MIN_RANGE)
        mask &= (depth_region <= MAX_RANGE)
        return mask

    def bbox_intersection(self, box_a, box_b):
        x1 = max(box_a.x1, box_b.x1)
        y1 = max(box_a.y1, box_b.y1)
        x2 = min(box_a.x2, box_b.x2)
        y2 = min(box_a.y2, box_b.y2)
        if x1 < x2 and y1 < y2:
            return (x1, y1, x2, y2)  # si sovrappongono, questo è il rettangolo di sovrapposizione
        else:
            return None  # non si sovrappongono

    # VECCHIE FUNZIONI
    def estimate_object_position(self, bounding_box, depth_image, image_width, image_height,
                                  focal_length_x, focal_length_y,
                                  principal_point_x, principal_point_y):
        """
        Convert a 2D bounding box + depth image into a 3D position (X, Y, Z),
        expressed in the depth camera's optical frame, using the pinhole camera model.
        Returns None if the depth value at the box centre is invalid.
        """
        x1, y1, x2, y2 = bounding_box.xyxy[0]

        pixel_x = int((x1 + x2) / 2)
        pixel_y = int((y1 + y2) / 2)

        # Clamp to image bounds in case the box centre falls slightly outside
        # (can happen for objects partially cropped at the image edge)
        pixel_x = min(max(pixel_x, 0), image_width - 1)
        pixel_y = min(max(pixel_y, 0), image_height - 1)

        depth_value = float(depth_image[pixel_y, pixel_x])

        # Reject invalid depth readings — zero or NaN are common at object
        # edges or on reflective surfaces (a metal can is a good example)
        if depth_value <= 0.0 or depth_value != depth_value:
            return None

        point_z = depth_value
        point_x = (pixel_x - principal_point_x) * point_z / focal_length_x
        point_y = (pixel_y - principal_point_y) * point_z / focal_length_y

        return point_x, point_y, point_z

    def publish_object_position(self, point_x, point_y, point_z, header):
        """Publish the estimated 3D position of the target object as a PointStamped message."""
        position_msg = PointStamped()
        position_msg.header = header
        position_msg.point.x = point_x
        position_msg.point.y = point_y
        position_msg.point.z = point_z
        self.object_position_pub.publish(position_msg)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def destroy_node(self):
        """Ensure the worker thread stops cleanly before the node is destroyed."""
        self.worker_running = False
        self.worker_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RtObjectDetectionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
