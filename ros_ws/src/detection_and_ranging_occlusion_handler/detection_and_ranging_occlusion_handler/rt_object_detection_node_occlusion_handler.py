"""
Real-time object detection node.

Detects a target object (e.g. a Coca-Cola can) in an RGB image stream using
YOLOv8, fuses the detection with a synchronised depth image to estimate the
object's 3D position, and publishes both an annotated frame (for debugging)
and the object's position as a geometry_msgs/PointStamped message.

Two position-estimation strategies are available, selected via the
'use_occlusion_handling' ROS 2 parameter:

  - Robust mode (default): accounts for occlusion between the tracked
    objects (coke can, pringles, biscuits) and for background pixels
    leaking into the bounding box. Slower, more accurate when objects can
    occlude each other.
  - Light mode: takes the bounding box centre directly, no occlusion
    handling. Only appropriate when the target is known to never be
    occluded by another tracked object in the scene.

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
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from ultralytics import YOLO

# Custom messages for the multi-point, multi-object output (see the
# tiago_vision_msgs package): class name + known shape + up to 3 points on
# the visible surface, so a downstream grasp-planning node can fit a circle
# through them and recover a cylindrical object's central axis.
from tiago_vision_msgs.msg import TrackedObjectPoints, TrackedObjectsArray

# Name of the classese (as defined in the YOLO model's names dict) that we want to track 
COKE_CLASS_NAME = 'coke can'
BISCUITS_CLASS_NAME = 'biscuits pack'
PRINGLES_CLASS_NAME = 'pringles can'
INTERESTED_OBJ = [COKE_CLASS_NAME, BISCUITS_CLASS_NAME, PRINGLES_CLASS_NAME]


TARGET_CLASS_NAME = COKE_CLASS_NAME

DETECTION_CONFIDENCE_THRESHOLD = 0.6

# Valori di depth della Intel RealSense D435 montata sul Tiago Pro
MIN_RANGE = 0.3
MAX_RANGE = 3.0

# Absolute minimum number of clean pixels required for an object's depth
# estimate to be trusted when establishing the near/far ordering between
# tracked objects (Step 3). Below this, the median is statistically too
# noisy to trust for ordering purposes — distinct from MIN_VALID_PIXELS
# below, which gates the final published position, not the ordering step.
# TODO: calibrate empirically
MIN_PIXELS_FOR_ORDERING = 20

# Minimum fraction of an object's own bounding box area that must remain
# "clean" for a pair involving the target to be trusted enough to decide
# who occludes whom. If either object in a target-involving, overlapping
# pair falls below this, the frame is rejected rather than risking a wrong
# grasp (see Step 3.5).
# TODO: calibrate empirically
MIN_ORDERING_COVERAGE_RATIO = 0.20

# Small tolerance (metres) used only to avoid flip-flopping between "A is
# closer" and "B is closer" when two depths are numerically almost
# identical (floating-point / render noise). NOT a physical size threshold:
# any genuine depth difference, however small, means real occlusion — a
# nearer opaque surface always blocks a farther one, there is no minimum
# physical gap required (see conversation for why a can-radius-based
# threshold was the wrong basis for this check).
# TODO: verify actual simulator depth noise and recalibrate this value
DEPTH_ORDER_EPSILON_M = 0.0005

# Tolerance band (metres) used only in Step 8 to pick a small, spatially
# coherent cluster of pixels near the median depth, instead of a single
# pixel via argmin. A single-pixel argmin is a discontinuous choice — a
# sub-millimetre change in the data can flip which pixel "wins", causing
# the published point to jump between frames even on a perfectly static
# scene. Averaging over a small tolerance band is continuous: small input
# changes only shift the result by a proportionally small amount.
# TODO: calibrate empirically
NEAR_MEDIAN_TOLERANCE_M = 0.003

# Known geometric shape of each tracked class (English words, so a
# downstream consumer knows how to interpret the points). All three current
# classes are cylindrical — SHAPE_CUBE/SHAPE_SPHERE are unused placeholders
# for future object classes with different geometries.
SHAPE_CYLINDER = 'cylinder'
SHAPE_CUBE = 'cube'
SHAPE_SPHERE = 'sphere'

OBJECT_SHAPES = {
    COKE_CLASS_NAME: SHAPE_CYLINDER,
    PRINGLES_CLASS_NAME: SHAPE_CYLINDER,
    BISCUITS_CLASS_NAME: SHAPE_CYLINDER,
}

# Number of points sampled along a single, roughly constant-height row
# across the target's visible width, used downstream to fit a circle and
# recover the cylinder's central axis for grasp planning.
NUM_GRASP_CIRCLE_POINTS = 3

# Minimum span (in columns) between the left and right genuine boundaries
# found by the row-walk below, before a row is trusted to host the
# NUM_GRASP_CIRCLE_POINTS points.
# TODO: calibrate empirically
MIN_VALID_COLUMNS_FOR_ROW = 10

# How many rows above/below the starting height we're willing to search if
# that exact row doesn't have enough valid columns (e.g. an occluder
# happens to cross exactly at that height).
# TODO: calibrate empirically
MAX_ROW_SEARCH_OFFSET = 15

# Maximum plausible depth change between two ADJACENT columns on the same
# row, used to walk outward from the anchor point and stop at a genuine
# jump to a different object (rather than the object's own gradual,
# curvature-caused depth increase towards its edges). Unlike the
# depth-ORDERING check elsewhere (DEPTH_ORDER_EPSILON_M), a radius-sized
# threshold IS well justified here: it bounds how much THIS object's own
# depth can vary from its frontal point to its visible tangent edge.
# TODO: calibrate empirically
GRASP_ROW_MAX_DEPTH_STEP_M = 0.033

# Safety margin (in columns) backed off from each detected genuine boundary
# before placing a grasp-circle point there — avoids the single riskiest
# pixel (right at the jump/occlusion edge, most exposed to antialiasing or
# a slightly mistimed jump detection) while keeping nearly all of the
# spread that makes the circle fit numerically stable (see conversation).
# TODO: calibrate empirically
GRASP_POINT_BACKOFF_PIXELS = 3

# Numero minimo di pixel puliti (non occlusi, con depth valido) richiesti
# dentro la bounding box del target perché la stima di posizione sia
# considerata affidabile. 
# TODO: dare una motivazione a questo valore
MIN_VALID_PIXELS = 20

# Frazione minima dell'area originale della bounding box del target che deve
# rimanere dopo aver escluso i pixel occlusi/non validi.
MIN_COVERAGE_RATIO = 0.15

# TODO: parameterise this (e.g. via a ROS 2 node parameter or a package-relative
# path) so the node isn't tied to one machine's filesystem layout.
MODEL_WEIGHTS_PATH = "/home/user/exchange/models/weights/best2.pt"


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

        # Selects the position-estimation strategy at runtime (see module
        # docstring). Robust mode is the default; light mode can be enabled
        # for scenes known to be free of occlusion between tracked objects.
        self.declare_parameter('use_occlusion_handling', True)

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
        self.tracked_objects_pub = self.create_publisher(
            TrackedObjectsArray, 'yolo/tracked_objects_points', 10
        )
        self.grasp_points_marker_pub = self.create_publisher(
            Marker, 'yolo/grasp_circle_points', 10
        )
        # Same objects as yolo/tracked_objects_points, but with only the
        # single anchor point per object (not the full 3-point grasp
        # circle) — for consumers that just need one coordinate per object.
        self.object_anchor_points_pub = self.create_publisher(
            TrackedObjectsArray, 'yolo/tracked_objects_anchor_point', 10
        )
        # Blue sphere per object, one per anchor point (not 3 like
        # grasp_points_marker_pub) — published by BOTH modes, so the
        # anchor points are visible in RViz even in light mode, where no
        # Marker exists today.
        self.anchor_points_marker_pub = self.create_publisher(
            Marker, 'yolo/anchor_points_marker', 10
        )
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

        # TODO:
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

        # Choose the estimation strategy for this frame (see module docstring
        # and the 'use_occlusion_handling' parameter declared in __init__).
        use_occlusion_handling = self.get_parameter('use_occlusion_handling').value

        if use_occlusion_handling:
            self.estimate_position_with_occlusion_handling(
                detection_results, depth_image,
                focal_length_x, focal_length_y, principal_point_x, principal_point_y,
                rgb_msg.header
            )
        else:
            self.estimate_position_light_mode(
                detection_results, depth_image, image_width, image_height,
                focal_length_x, focal_length_y, principal_point_x, principal_point_y,
                rgb_msg.header
            )

    # ------------------------------------------------------------------
    # Light mode — bounding box centre, no occlusion handling
    # ------------------------------------------------------------------

    def estimate_position_light_mode(self, detection_results, depth_image, image_width, image_height,
                                      focal_length_x, focal_length_y,
                                      principal_point_x, principal_point_y, header):
        """
        Simple/fast position estimate for EVERY tracked object (not just the
        target): bounding-box centre + depth read, no occlusion handling —
        same technique used for the target, applied per detected class.

        Only appropriate when objects are known to never occlude each other
        in the scene. The target's point is also published on the legacy
        single-point topic, unchanged, for existing consumers.
        """
        self.get_logger().info("Modalità light", throttle_duration_sec=2.0)

        # One detection per tracked class (first one above threshold).
        best_detection_per_class = {}
        for bounding_box in detection_results.boxes:
            class_id = int(bounding_box.cls[0])
            class_name = self.detection_model.names[class_id]
            confidence_score = float(bounding_box.conf[0])
            self.get_logger().info(f"DEBUG light: {class_name} conf={confidence_score:.2f}", throttle_duration_sec=2.0)

            if class_name not in INTERESTED_OBJ or confidence_score < DETECTION_CONFIDENCE_THRESHOLD:
                continue
            if class_name not in best_detection_per_class:
                best_detection_per_class[class_name] = bounding_box

        if TARGET_CLASS_NAME not in best_detection_per_class:
            return  # nothing usable for the target this frame

        anchor_entries = []
        target_point = None

        for class_name, bounding_box in best_detection_per_class.items():
            position = self.estimate_object_position(
                bounding_box, depth_image, image_width, image_height,
                focal_length_x, focal_length_y, principal_point_x, principal_point_y
            )
            if position is None:
                self.get_logger().warn(
                    f"DEBUG light: depth non valido al centro bbox di {class_name}",
                    throttle_duration_sec=2.0
                )
                continue

            point_x, point_y, point_z = position
            anchor_entries.append(
                TrackedObjectPoints(
                    class_name=class_name,
                    shape=OBJECT_SHAPES.get(class_name, ''),
                    points=[Point(x=point_x, y=point_y, z=point_z)],
                )
            )
            if class_name == TARGET_CLASS_NAME:
                target_point = (point_x, point_y, point_z)

        if target_point is None:
            return  # target's own bbox-centre depth was invalid this frame

        self.publish_object_position(*target_point, header)

        anchor_msg = TrackedObjectsArray()
        anchor_msg.header = header
        anchor_msg.objects = anchor_entries
        self.object_anchor_points_pub.publish(anchor_msg)
        self.publish_anchor_points_marker(anchor_entries, header)

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

    # ------------------------------------------------------------------
    # Robust mode — occlusion-aware position estimation
    # ------------------------------------------------------------------

    def collect_tracked_detections(self, detection_results):
        """
        Reduce YOLO's raw detections to at most one bounding box per tracked
        class (coca-cola, pringles, biscuits), keeping only detections above
        the confidence threshold.

        Returns:
            A dict {class_name: (x1, y1, x2, y2)} with integer pixel
            coordinates for every tracked class detected above threshold, or
            None if the target class was seen but with confidence below
            threshold (signal to skip the frame entirely).
        """
        detected_boxes = {}

        for bounding_box in detection_results.boxes:
            class_id = int(bounding_box.cls[0])
            class_name = self.detection_model.names[class_id]
            confidence_score = float(bounding_box.conf[0])

            if confidence_score < DETECTION_CONFIDENCE_THRESHOLD:
                if class_name == TARGET_CLASS_NAME:
                    self.get_logger().warn(
                        f"Coca-cola detected but confidence too low ({confidence_score:.2f} < {DETECTION_CONFIDENCE_THRESHOLD}).",
                        throttle_duration_sec=5.0
                    )
                    return None  # Low-confidence target: nothing usable this frame
                continue

            if class_name in INTERESTED_OBJ:
                x1, y1, x2, y2 = bounding_box.xyxy[0]
                detected_boxes[class_name] = (int(x1), int(y1), int(x2), int(y2))

        return detected_boxes

    def bbox_intersection(self, box_a, box_b):
        """
        Compute the pixel-space intersection of two axis-aligned bounding boxes.

        Args:
            box_a, box_b: (x1, y1, x2, y2) integer pixel coordinates.

        Returns:
            (x1, y1, x2, y2) of the overlapping region, or None if the boxes
            don't overlap.
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        if x1 < x2 and y1 < y2:
            return (x1, y1, x2, y2)
        return None

    def valid_depth_mask(self, depth_region):
        """
        Boolean mask (same shape as depth_region): True where the depth
        reading is usable — i.e. finite (not zero/NaN/Inf) and inside the
        RealSense D435's operating range (see MIN_RANGE / MAX_RANGE).
        """
        mask = np.isfinite(depth_region)
        mask &= (depth_region >= MIN_RANGE)
        mask &= (depth_region <= MAX_RANGE)
        return mask

    def contiguous_valid_segment(self, valid_1d, index):
        """
        Given a 1D boolean array and an index known to be True, return the
        (lo, hi) bounds of the contiguous run of True values containing that
        index. The midpoint of any contiguous run is, by definition, also
        True — this is what guarantees the sequential centring in Step 8
        never lands on an excluded/invalid pixel, on any mask shape
        (including non-convex ones, e.g. an "L" left after occlusion).
        """
        lo = index
        while lo - 1 >= 0 and valid_1d[lo - 1]:
            lo -= 1
        hi = index
        while hi + 1 < valid_1d.size and valid_1d[hi + 1]:
            hi += 1
        return lo, hi

    def walk_to_genuine_boundary(self, target_depth_region, target_valid_mask, row_local, start_col, step):
        """
        Walk one column at a time from start_col in the given direction
        (step = +1 or -1), stopping as soon as either the next column is
        invalid or the depth change from the immediately previous column
        exceeds GRASP_ROW_MAX_DEPTH_STEP_M. A real object boundary (a
        different object, or background) shows up as a sharp jump between
        adjacent columns; the object's own curvature only ever increases
        depth gradually — this is what tells the two apart.

        Returns the last column reached before stopping (inclusive), which
        is guaranteed to be a genuine, on-object column.
        """
        width = target_valid_mask.shape[1]
        col = start_col
        prev_depth = target_depth_region[row_local, col]

        while True:
            next_col = col + step
            if next_col < 0 or next_col >= width:
                break
            if not target_valid_mask[row_local, next_col]:
                break
            next_depth = target_depth_region[row_local, next_col]
            if abs(next_depth - prev_depth) > GRASP_ROW_MAX_DEPTH_STEP_M:
                break
            col = next_col
            prev_depth = next_depth

        return col

    def select_grasp_circle_points(self, target_depth_region, target_valid_mask,
                                    start_row_local, start_col_local,
                                    target_x1, target_y1,
                                    focal_length_x, focal_length_y,
                                    principal_point_x, principal_point_y):
        """
        Sample NUM_GRASP_CIRCLE_POINTS points spread across the target's
        visible width at a single, roughly constant image row — used
        downstream to fit a circle through them and recover the can's
        central axis for grasp planning.

        Starts at (start_row_local, start_col_local) — the Step 8
        representative point, already guaranteed on-object — and walks
        outward left/right to the genuine boundaries (see
        walk_to_genuine_boundary), then backs off GRASP_POINT_BACKOFF_PIXELS
        from each boundary to avoid the single riskiest edge pixel while
        keeping most of the spread a stable circle fit needs. The middle
        point is the anchor itself. If that row's genuine span is too
        narrow (e.g. an occluder crosses exactly at that height), searches
        outward row by row up to MAX_ROW_SEARCH_OFFSET before giving up.

        Returns a list of NUM_GRASP_CIRCLE_POINTS geometry_msgs/Point in the
        camera's depth optical frame, or None if no row in range had a wide
        enough genuine span.
        """
        height, width = target_valid_mask.shape
        row_offsets = [0]
        for offset in range(1, MAX_ROW_SEARCH_OFFSET + 1):
            row_offsets.extend([-offset, offset])

        def pixel_to_point(row_local, col_local):
            depth_value = float(target_depth_region[row_local, col_local])
            pixel_x = col_local + target_x1
            pixel_y = row_local + target_y1
            point_x = (pixel_x - principal_point_x) * depth_value / focal_length_x
            point_y = (pixel_y - principal_point_y) * depth_value / focal_length_y
            return Point(x=float(point_x), y=float(point_y), z=depth_value)

        for offset in row_offsets:
            row_local = start_row_local + offset
            if row_local < 0 or row_local >= height:
                continue
            if not target_valid_mask[row_local, start_col_local]:
                continue  # anchor column itself isn't valid on this row

            right_boundary = self.walk_to_genuine_boundary(
                target_depth_region, target_valid_mask, row_local, start_col_local, +1
            )
            left_boundary = self.walk_to_genuine_boundary(
                target_depth_region, target_valid_mask, row_local, start_col_local, -1
            )

            if (right_boundary - left_boundary + 1) < MIN_VALID_COLUMNS_FOR_ROW:
                continue

            right_col = max(start_col_local, right_boundary - GRASP_POINT_BACKOFF_PIXELS)
            left_col = min(start_col_local, left_boundary + GRASP_POINT_BACKOFF_PIXELS)

            return [
                pixel_to_point(row_local, left_col),
                pixel_to_point(row_local, start_col_local),
                pixel_to_point(row_local, right_col),
            ]

        return None  # no row in range had a wide enough genuine span


    def compute_object_estimate(self, subject_class_name, detected_boxes, overlap_regions,
                                 clean_median_depth, clean_pixel_stats, ordered_by_depth,
                                 depth_image, focal_length_x, focal_length_y,
                                 principal_point_x, principal_point_y):
        """
        Run Steps 3.5 and 5-9 for a single tracked object (the "subject" —
        the target or an obstacle), instead of hard-coding them to the
        target only. Returns (point_x, point_y, point_z, grasp_circle_points)
        on success, or None if the subject's own depth couldn't be
        established, its ordering against an overlapping neighbour isn't
        trustworthy, or too few clean pixels remain after exclusion.

        Callers decide what a None means: for the target it should abort
        the whole frame; for an obstacle it should just skip that object.
        """
        if subject_class_name not in clean_median_depth:
            self.get_logger().warn(
                f"Not enough clean pixels to estimate {subject_class_name}'s depth for ordering.",
                throttle_duration_sec=5.0
            )
            return None

        # --- Step 3.5: refuse to guess when this subject's ordering is shaky ---
        subject_pixel_count, subject_coverage_ratio = clean_pixel_stats[subject_class_name]
        if subject_coverage_ratio < MIN_ORDERING_COVERAGE_RATIO:
            self.get_logger().warn(
                f"{subject_class_name}'s own clean-pixel coverage too low to trust any "
                f"occlusion ordering involving it ({subject_coverage_ratio:.0%}).",
                throttle_duration_sec=5.0
            )
            return None

        for other_class_name in detected_boxes:
            if other_class_name == subject_class_name:
                continue
            overlap_key = frozenset((other_class_name, subject_class_name))
            if overlap_key not in overlap_regions:
                continue  # doesn't overlap the subject, ordering with it is irrelevant

            other_pixel_count, other_coverage_ratio = clean_pixel_stats[other_class_name]
            if other_pixel_count < MIN_PIXELS_FOR_ORDERING or other_coverage_ratio < MIN_ORDERING_COVERAGE_RATIO:
                self.get_logger().warn(
                    f"Ordering between {subject_class_name} and {other_class_name} is too "
                    f"unreliable to trust ({other_pixel_count} px, "
                    f"{other_coverage_ratio:.0%} coverage) — skipping this object "
                    f"rather than risking a bad estimate.",
                    throttle_duration_sec=5.0
                )
                return None

        # --- Step 5: exclude pixels occluded by objects genuinely closer than the subject ---
        subject_depth = clean_median_depth[subject_class_name]
        subject_x1, subject_y1, subject_x2, subject_y2 = detected_boxes[subject_class_name]

        exclusion_mask = np.zeros((subject_y2 - subject_y1, subject_x2 - subject_x1), dtype=bool)
        for other_class_name, other_depth_value in ordered_by_depth:
            if other_class_name == subject_class_name:
                break  # remaining objects in the ordering are farther, they can't occlude the subject

            if (subject_depth - other_depth_value) <= DEPTH_ORDER_EPSILON_M:
                continue

            overlap_key = frozenset((other_class_name, subject_class_name))
            if overlap_key not in overlap_regions:
                continue  # closer, but its box doesn't actually overlap the subject's

            ox1, oy1, ox2, oy2 = overlap_regions[overlap_key]
            exclusion_mask[oy1 - subject_y1:oy2 - subject_y1, ox1 - subject_x1:ox2 - subject_x1] = True

        # --- Step 6: combine occlusion exclusion with the depth-validity filter ---
        subject_depth_region = depth_image[subject_y1:subject_y2, subject_x1:subject_x2]
        keep_mask = ~exclusion_mask
        subject_valid_mask = keep_mask & self.valid_depth_mask(subject_depth_region)

        rows_local, cols_local = np.where(subject_valid_mask)
        clean_depth_values = subject_depth_region[subject_valid_mask]

        # --- Step 7: sanity check — enough clean pixels left to trust the estimate? ---
        original_area = (subject_x2 - subject_x1) * (subject_y2 - subject_y1)
        coverage_ratio = clean_depth_values.size / original_area if original_area > 0 else 0.0

        if clean_depth_values.size < MIN_VALID_PIXELS or coverage_ratio < MIN_COVERAGE_RATIO:
            self.get_logger().warn(
                f"{subject_class_name} too occluded to trust the position estimate "
                f"({clean_depth_values.size} px, {coverage_ratio:.0%} coverage).",
                throttle_duration_sec=5.0
            )
            return None

        # --- Step 8: average a small, spatially coherent cluster near the median ---
        median_depth = np.median(clean_depth_values)
        near_median_mask = np.abs(clean_depth_values - median_depth) <= NEAR_MEDIAN_TOLERANCE_M

        if np.any(near_median_mask):
            centre_row = rows_local[near_median_mask].mean()
            centre_col = cols_local[near_median_mask].mean()
            candidate_rows = rows_local[near_median_mask]
            candidate_cols = cols_local[near_median_mask]
            nearest_idx = np.argmin(
                (candidate_rows - centre_row) ** 2 + (candidate_cols - centre_col) ** 2
            )
            start_row = int(candidate_rows[nearest_idx])
            start_col = int(candidate_cols[nearest_idx])

            col_lo, col_hi = self.contiguous_valid_segment(subject_valid_mask[start_row, :], start_col)
            centred_col = round((col_lo + col_hi) / 2)

            row_lo, row_hi = self.contiguous_valid_segment(subject_valid_mask[:, centred_col], start_row)
            centred_row = round((row_lo + row_hi) / 2)

            final_depth = float(subject_depth_region[centred_row, centred_col])
            final_pixel_x = centred_col + subject_x1
            final_pixel_y = centred_row + subject_y1
        else:
            closest_idx = np.argmin(np.abs(clean_depth_values - median_depth))
            final_depth = float(clean_depth_values[closest_idx])
            final_pixel_x = int(cols_local[closest_idx]) + subject_x1
            final_pixel_y = int(rows_local[closest_idx]) + subject_y1
            centred_col = final_pixel_x - subject_x1
            centred_row = final_pixel_y - subject_y1

        # --- Back-projection (pinhole model) ---
        point_z = final_depth
        point_x = (final_pixel_x - principal_point_x) * point_z / focal_length_x
        point_y = (final_pixel_y - principal_point_y) * point_z / focal_length_y

        # --- Step 9: sample NUM_GRASP_CIRCLE_POINTS points at ~constant height ---
        start_row_local = int(round(final_pixel_y)) - subject_y1
        start_col_local = centred_col
        grasp_circle_points = self.select_grasp_circle_points(
            subject_depth_region, subject_valid_mask, start_row_local, start_col_local,
            subject_x1, subject_y1,
            focal_length_x, focal_length_y, principal_point_x, principal_point_y
        )

        if grasp_circle_points is None:
            grasp_circle_points = [Point(x=point_x, y=point_y, z=point_z)] * NUM_GRASP_CIRCLE_POINTS

        return point_x, point_y, point_z, grasp_circle_points



    def estimate_position_with_occlusion_handling(self, detection_results, depth_image,
                                                   focal_length_x, focal_length_y,
                                                   principal_point_x, principal_point_y,
                                                   header):
        """
        Robust position estimation for the target object (coca-cola) that
        accounts for occlusion by the other tracked objects (pringles,
        biscuits) and for background/other-object pixels leaking into the
        bounding box.

        Steps:
          1. Keep one bounding box per tracked class above the confidence
             threshold.
          2. For every pair of tracked objects whose boxes overlap in the
             image, mark the overlapping region as "ambiguous" for both —
             pure 2D geometry, no depth involved yet.
          3. Compute each object's depth using only its non-ambiguous pixels,
             so occluder and occluded object never contaminate each other's
             estimate (this is what lets us tell "A occludes B" apart from
             "B occludes A" even when one of the two is heavily covered).
        3.5. Refuse to trust the ordering for any target-involving,
             overlapping pair if either side has too few clean pixels
             (see MIN_PIXELS_FOR_ORDERING / MIN_ORDERING_COVERAGE_RATIO) —
             skip the frame entirely rather than risk excluding the wrong
             pixels.
          4. Order the objects by this clean depth (nearest to farthest).
          5. For the target, exclude only the ambiguous pixels shared with
             objects that are genuinely closer (beyond a tiny numerical
             tolerance, DEPTH_ORDER_EPSILON_M, to avoid frame-to-frame
             flip-flopping on near-identical depths) — an object that is
             farther than the target does not occlude it, so its overlap is
             left untouched.
          6. Discard invalid depth readings (out of sensor range, NaN/Inf).
          7. Sanity-check how many clean pixels remain; if too few, treat the
             estimate as unreliable and skip this frame instead of
             publishing a misleading position.
          8. Restrict to the pixels within NEAR_MEDIAN_TOLERANCE_M of the
             median depth (a small, spatially coherent cluster on the
             object's surface, not the independent medians of u, v and z
             combined, which could describe a pixel that never actually
             existed). Find the real cluster member closest to its centroid
             as a starting point, then push it away from the occlusion
             boundary with two sequential contiguous-segment cuts (row,
             then the new column) — guaranteed to stay on a valid pixel on
             any mask shape, including non-convex ones. Falls back to the
             single closest pixel if the tolerance band is empty.
          9. Starting from the Step 8 point, walk left/right along its row
             to the genuine object boundaries (stopping at a depth jump
             that signals a different object, not the can's own gradual
             curvature — GRASP_ROW_MAX_DEPTH_STEP_M), back off
             GRASP_POINT_BACKOFF_PIXELS from each to avoid the single
             riskiest edge pixel, and publish these two plus the anchor as
             NUM_GRASP_CIRCLE_POINTS points for a downstream node to fit a
             circle through and recover the can's central axis.

        Publishes a PointStamped in the camera's depth optical frame if a
        reliable estimate is found; otherwise logs a warning and does
        nothing.
        """
        image_height, image_width = depth_image.shape[:2]

        # --- Step 1: one bounding box per tracked class ---
        detected_boxes = self.collect_tracked_detections(detection_results)
        if not detected_boxes:
            return  # nothing tracked in this frame (or target below threshold)
        if TARGET_CLASS_NAME not in detected_boxes:
            return  # target not visible in this frame

        # --- Step 2: pairwise bounding box overlaps (pure 2D geometry) ---
        overlap_regions = {}  # frozenset({class_a, class_b}) -> (x1, y1, x2, y2)
        ambiguous_mask = {
            class_name: np.zeros((image_height, image_width), dtype=bool)
            for class_name in detected_boxes
        }

        tracked_classes = list(detected_boxes.keys())
        for i in range(len(tracked_classes)):
            for j in range(i + 1, len(tracked_classes)):
                class_a, class_b = tracked_classes[i], tracked_classes[j]
                overlap = self.bbox_intersection(detected_boxes[class_a], detected_boxes[class_b])
                if overlap is None:
                    continue
                ox1, oy1, ox2, oy2 = overlap
                overlap_regions[frozenset((class_a, class_b))] = overlap
                ambiguous_mask[class_a][oy1:oy2, ox1:ox2] = True
                ambiguous_mask[class_b][oy1:oy2, ox1:ox2] = True

        # --- Step 3: "clean" depth per object (own pixels only, overlaps excluded) ---
        clean_median_depth = {}
        # Stats for EVERY detected object, even ones that fail the ordering
        # threshold below — Step 3.5 needs these to judge whether a
        # target-involving pair's ordering can be trusted at all.
        clean_pixel_stats = {}  # class_name -> (pixel_count, coverage_ratio)

        for class_name, (x1, y1, x2, y2) in detected_boxes.items():
            region = depth_image[y1:y2, x1:x2]
            clean_pixels_mask = ~ambiguous_mask[class_name][y1:y2, x1:x2]
            clean_values = region[clean_pixels_mask]
            clean_values = clean_values[self.valid_depth_mask(clean_values)]

            original_area = (x2 - x1) * (y2 - y1)
            coverage_ratio = clean_values.size / original_area if original_area > 0 else 0.0
            clean_pixel_stats[class_name] = (clean_values.size, coverage_ratio)

            if clean_values.size < MIN_PIXELS_FOR_ORDERING:
                # Too few clean samples for a trustworthy median — excluded
                # from the ordering below (generalises the old "size == 0"
                # check to any too-small sample, not just zero).
                continue
            clean_median_depth[class_name] = np.median(clean_values)

        # --- Step 4: order tracked objects from nearest to farthest ---
        ordered_by_depth = sorted(clean_median_depth.items(), key=lambda item: item[1])

        # --- Steps 3.5 + 5-9, run once for the target (mandatory) ---
        target_result = self.compute_object_estimate(
            TARGET_CLASS_NAME, detected_boxes, overlap_regions,
            clean_median_depth, clean_pixel_stats, ordered_by_depth,
            depth_image, focal_length_x, focal_length_y, principal_point_x, principal_point_y
        )
        if target_result is None:
            return  # target unreliable this frame — publish nothing at all

        point_x, point_y, point_z, target_grasp_points = target_result

        # Legacy single-point topic — target only, unchanged for existing consumers.
        self.publish_object_position(point_x, point_y, point_z, header)

        object_entries = [
            TrackedObjectPoints(
                class_name=TARGET_CLASS_NAME,
                shape=OBJECT_SHAPES.get(TARGET_CLASS_NAME, ''),
                points=target_grasp_points,
            )
        ]
        # Same objects, but with just the single anchor point each — the
        # same point already used as the middle of the 3 and as the legacy
        # PointStamped for the target.
        anchor_entries = [
            TrackedObjectPoints(
                class_name=TARGET_CLASS_NAME,
                shape=OBJECT_SHAPES.get(TARGET_CLASS_NAME, ''),
                points=[Point(x=point_x, y=point_y, z=point_z)],
            )
        ]

        # --- Steps 3.5 + 5-9, run once per obstacle (optional — skip on failure) ---
        for other_class_name in detected_boxes:
            if other_class_name == TARGET_CLASS_NAME:
                continue

            other_result = self.compute_object_estimate(
                other_class_name, detected_boxes, overlap_regions,
                clean_median_depth, clean_pixel_stats, ordered_by_depth,
                depth_image, focal_length_x, focal_length_y, principal_point_x, principal_point_y
            )
            if other_result is None:
                continue  # skip only this obstacle, target/other obstacles unaffected

            other_point_x, other_point_y, other_point_z, other_grasp_points = other_result
            object_entries.append(
                TrackedObjectPoints(
                    class_name=other_class_name,
                    shape=OBJECT_SHAPES.get(other_class_name, ''),
                    points=other_grasp_points,
                )
            )
            anchor_entries.append(
                TrackedObjectPoints(
                    class_name=other_class_name,
                    shape=OBJECT_SHAPES.get(other_class_name, ''),
                    points=[Point(x=other_point_x, y=other_point_y, z=other_point_z)],
                )
            )

        tracked_objects_msg = TrackedObjectsArray()
        tracked_objects_msg.header = header
        tracked_objects_msg.objects = object_entries
        self.tracked_objects_pub.publish(tracked_objects_msg)

        anchor_points_msg = TrackedObjectsArray()
        anchor_points_msg.header = header
        anchor_points_msg.objects = anchor_entries
        self.object_anchor_points_pub.publish(anchor_points_msg)
        self.publish_anchor_points_marker(anchor_entries, header)

        # --- RViz visualisation: every object's points, concatenated, as one sphere list ---
        all_points = [point for entry in object_entries for point in entry.points]

        marker_msg = Marker()
        marker_msg.header = header
        marker_msg.ns = 'grasp_circle_points'
        marker_msg.id = 0
        marker_msg.type = Marker.SPHERE_LIST
        marker_msg.action = Marker.ADD
        marker_msg.scale.x = 0.01
        marker_msg.scale.y = 0.01
        marker_msg.scale.z = 0.01
        marker_msg.color.r = 1.0
        marker_msg.color.g = 0.0
        marker_msg.color.b = 0.0
        marker_msg.color.a = 1.0
        marker_msg.points = all_points
        self.grasp_points_marker_pub.publish(marker_msg)


    def publish_object_position(self, point_x, point_y, point_z, header):
        """Publish the estimated 3D position of the target object as a PointStamped message."""
        position_msg = PointStamped()
        position_msg.header = header
        position_msg.point.x = point_x
        position_msg.point.y = point_y
        position_msg.point.z = point_z
        self.object_position_pub.publish(position_msg)

    def publish_anchor_points_marker(self, anchor_entries, header):
        """
        Publish every object's single anchor point as a small blue sphere
        list, so it's visible in RViz regardless of which mode produced it
        — unlike grasp_points_marker_pub (robust mode only, 3 points per
        object, red), this one point per object marker is shared by both
        light and robust mode.
        """
        marker_msg = Marker()
        marker_msg.header = header
        marker_msg.ns = 'anchor_points'
        marker_msg.id = 0
        marker_msg.type = Marker.SPHERE_LIST
        marker_msg.action = Marker.ADD
        marker_msg.scale.x = 0.015
        marker_msg.scale.y = 0.015
        marker_msg.scale.z = 0.015
        marker_msg.color.r = 0.0
        marker_msg.color.g = 0.4
        marker_msg.color.b = 1.0
        marker_msg.color.a = 1.0
        marker_msg.points = [entry.points[0] for entry in anchor_entries]
        self.anchor_points_marker_pub.publish(marker_msg)

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