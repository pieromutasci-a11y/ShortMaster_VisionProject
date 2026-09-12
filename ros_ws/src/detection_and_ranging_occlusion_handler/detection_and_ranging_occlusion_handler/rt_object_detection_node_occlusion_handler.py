"""
Nodo di detection in tempo reale con gestione delle occlusioni tra oggetti
tracciati (coke can, pringles, biscuits). Rileva gli oggetti con YOLOv8,
fonde la detection con la depth per stimarne la posizione 3D, e pubblica sia
il frame annotato che le posizioni stimate.

Due modalita', selezionate dal parametro ROS 'use_occlusion_handling':
  - Robust (default): esclude i pixel condivisi tra oggetti sovrapposti
    prima di stimare la depth, piu' accurata in presenza di occlusioni.
  - Light: centro bounding box + depth diretta, nessuna gestione occlusioni.
"""

import threading
import time
from collections import deque
from threading import Lock
import numpy as np

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

from tiago_vision_msgs.msg import TrackedObjectPoints, TrackedObjectsArray

COKE_CLASS_NAME = 'coke can'
BISCUITS_CLASS_NAME = 'biscuits pack'
PRINGLES_CLASS_NAME = 'pringles can'
INTERESTED_OBJ = [COKE_CLASS_NAME, BISCUITS_CLASS_NAME, PRINGLES_CLASS_NAME]


TARGET_CLASS_NAME = COKE_CLASS_NAME

DETECTION_CONFIDENCE_THRESHOLD = 0.6

MIN_RANGE = 0.3
MAX_RANGE = 3.0

# Numero minimo di pixel puliti richiesto per fidarsi dell'ordinamento vicino/lontano tra oggetti.
MIN_PIXELS_FOR_ORDERING = 20

# Frazione minima di area "pulita" richiesta per fidarsi dell'ordinamento di un oggetto.
MIN_ORDERING_COVERAGE_RATIO = 0.20

# Tolleranza (metri) per evitare flip-flop tra "A piu' vicino" e "B piu' vicino" su depth quasi identiche.
DEPTH_ORDER_EPSILON_M = 0.0005

# Banda di tolleranza (metri) attorno alla mediana per scegliere un cluster di pixel stabile invece di un singolo argmin.
NEAR_MEDIAN_TOLERANCE_M = 0.003

SHAPE_CYLINDER = 'cylinder'
SHAPE_CUBE = 'cube'
SHAPE_SPHERE = 'sphere'

OBJECT_SHAPES = {
    COKE_CLASS_NAME: SHAPE_CYLINDER,
    PRINGLES_CLASS_NAME: SHAPE_CYLINDER,
    BISCUITS_CLASS_NAME: SHAPE_CYLINDER,
}

# Punti campionati lungo una riga a larghezza visibile costante, per il fit del cerchio a valle.
NUM_GRASP_CIRCLE_POINTS = 3

# Span minimo (colonne) tra i due bordi genuini prima di fidarsi di una riga per i punti del cerchio.
MIN_VALID_COLUMNS_FOR_ROW = 10

# Righe sopra/sotto da provare se quella di partenza non ha abbastanza colonne valide.
MAX_ROW_SEARCH_OFFSET = 15

# Massima variazione di depth plausibile tra due colonne adiacenti sulla stessa riga (bordo genuino vs curvatura propria dell'oggetto).
GRASP_ROW_MAX_DEPTH_STEP_M = 0.033

# Margine (colonne) di sicurezza dal bordo genuino individuato, prima di posizionare un punto del cerchio.
GRASP_POINT_BACKOFF_PIXELS = 3

# Numero minimo di pixel puliti richiesti nella bounding box del target per considerare la stima affidabile.
MIN_VALID_PIXELS = 20

# Frazione minima dell'area originale della bounding box che deve restare dopo l'esclusione dei pixel occlusi/non validi.
MIN_COVERAGE_RATIO = 0.15

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

        self.declare_parameter('use_occlusion_handling', True)

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

        self.annotated_frame_pub = self.create_publisher(Image, 'yolo/annotated_frame', 10)
        self.object_position_pub = self.create_publisher(PointStamped, 'yolo/coke_can_position', 10)
        self.tracked_objects_pub = self.create_publisher(
            TrackedObjectsArray, 'yolo/tracked_objects_points', 10
        )
        self.grasp_points_marker_pub = self.create_publisher(
            Marker, 'yolo/grasp_circle_points', 10
        )
        self.object_anchor_points_pub = self.create_publisher(
            TrackedObjectsArray, 'yolo/tracked_objects_anchor_point', 10
        )
        self.anchor_points_marker_pub = self.create_publisher(
            Marker, 'yolo/anchor_points_marker', 10
        )

        self.worker_running = True
        self.worker_thread = threading.Thread(target=self.detection_worker_loop, daemon=True)
        self.worker_thread.start()

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
        annotated_image_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        annotated_image_msg.header = rgb_msg.header
        self.annotated_frame_pub.publish(annotated_image_msg)

        image_height, image_width = depth_image.shape[:2]

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

    def estimate_position_light_mode(self, detection_results, depth_image, image_width, image_height,
                                      focal_length_x, focal_length_y,
                                      principal_point_x, principal_point_y, header):
        """Stima veloce per ogni oggetto tracciato: centro bounding box + depth, nessuna gestione occlusioni."""
        self.get_logger().info("Modalità light", throttle_duration_sec=2.0)

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
            return

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
            return

        self.publish_object_position(*target_point, header)

        anchor_msg = TrackedObjectsArray()
        anchor_msg.header = header
        anchor_msg.objects = anchor_entries
        self.object_anchor_points_pub.publish(anchor_msg)
        self.publish_anchor_points_marker(anchor_entries, header)

    def estimate_object_position(self, bounding_box, depth_image, image_width, image_height,
                                  focal_length_x, focal_length_y,
                                  principal_point_x, principal_point_y):
        """Converte bounding box + depth in una posizione 3D (modello pinhole). None se la depth al centro non e' valida."""
        x1, y1, x2, y2 = bounding_box.xyxy[0]

        pixel_x = int((x1 + x2) / 2)
        pixel_y = int((y1 + y2) / 2)

        pixel_x = min(max(pixel_x, 0), image_width - 1)
        pixel_y = min(max(pixel_y, 0), image_height - 1)

        depth_value = float(depth_image[pixel_y, pixel_x])

        if depth_value <= 0.0 or depth_value != depth_value:
            return None

        point_z = depth_value
        point_x = (pixel_x - principal_point_x) * point_z / focal_length_x
        point_y = (pixel_y - principal_point_y) * point_z / focal_length_y

        return point_x, point_y, point_z

    def collect_tracked_detections(self, detection_results):
        """Riduce le detection YOLO ad al massimo un box per classe tracciata sopra soglia.
        Ritorna {class_name: (x1, y1, x2, y2)}, o None se il target ha confidenza troppo bassa."""
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
                    return None
                continue

            if class_name in INTERESTED_OBJ:
                x1, y1, x2, y2 = bounding_box.xyxy[0]
                detected_boxes[class_name] = (int(x1), int(y1), int(x2), int(y2))

        return detected_boxes

    def bbox_intersection(self, box_a, box_b):
        """Intersezione pixel di due bounding box, o None se non si sovrappongono."""
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
        """Maschera booleana: True dove la depth e' finita e nel range operativo (MIN_RANGE/MAX_RANGE)."""
        mask = np.isfinite(depth_region)
        mask &= (depth_region >= MIN_RANGE)
        mask &= (depth_region <= MAX_RANGE)
        return mask

    def contiguous_valid_segment(self, valid_1d, index):
        """Bounds (lo, hi) del segmento contiguo di True in valid_1d che contiene index."""
        lo = index
        while lo - 1 >= 0 and valid_1d[lo - 1]:
            lo -= 1
        hi = index
        while hi + 1 < valid_1d.size and valid_1d[hi + 1]:
            hi += 1
        return lo, hi

    def walk_to_genuine_boundary(self, target_depth_region, target_valid_mask, row_local, start_col, step):
        """Cammina colonna per colonna da start_col (step=+1/-1) fino a un salto di depth
        oltre GRASP_ROW_MAX_DEPTH_STEP_M (bordo genuino) o un pixel non valido.
        Ritorna l'ultima colonna genuina raggiunta."""
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
        """Campiona NUM_GRASP_CIRCLE_POINTS punti sulla larghezza visibile del target a
        un'altezza ~costante, per il fit del cerchio a valle (asse del cilindro).

        Parte dal punto rappresentativo (start_row_local, start_col_local) e cammina
        verso i bordi genuini (walk_to_genuine_boundary), poi arretra di
        GRASP_POINT_BACKOFF_PIXELS da ciascun bordo (adattivo: mai piu' del margine
        disponibile, cosi' i 3 punti restano distinti anche con poco arco visibile).
        Se la riga di partenza ha uno span troppo stretto, cerca riga per riga fino a
        MAX_ROW_SEARCH_OFFSET. None se nessuna riga ha uno span sufficiente.
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
                continue

            right_boundary = self.walk_to_genuine_boundary(
                target_depth_region, target_valid_mask, row_local, start_col_local, +1
            )
            left_boundary = self.walk_to_genuine_boundary(
                target_depth_region, target_valid_mask, row_local, start_col_local, -1
            )

            if (right_boundary - left_boundary + 1) < MIN_VALID_COLUMNS_FOR_ROW:
                continue

            right_margin = right_boundary - start_col_local
            left_margin = start_col_local - left_boundary
            right_backoff = min(GRASP_POINT_BACKOFF_PIXELS, max(0, right_margin - 1))
            left_backoff = min(GRASP_POINT_BACKOFF_PIXELS, max(0, left_margin - 1))
            right_col = right_boundary - right_backoff
            left_col = left_boundary + left_backoff

            return [
                pixel_to_point(row_local, left_col),
                pixel_to_point(row_local, start_col_local),
                pixel_to_point(row_local, right_col),
            ]

        return None

    def compute_object_estimate(self, subject_class_name, detected_boxes, overlap_regions,
                                 clean_median_depth, clean_pixel_stats, ordered_by_depth,
                                 depth_image, focal_length_x, focal_length_y,
                                 principal_point_x, principal_point_y):
        """Stima posizione 3D e punti del cerchio di presa per un oggetto tracciato
        (target o ostacolo), escludendo i pixel occlusi da oggetti piu' vicini.
        None se la depth non e' stabilibile, l'ordinamento non e' affidabile, o
        restano troppo pochi pixel puliti."""
        if subject_class_name not in clean_median_depth:
            self.get_logger().warn(
                f"Not enough clean pixels to estimate {subject_class_name}'s depth for ordering.",
                throttle_duration_sec=5.0
            )
            return None

        # Step 3.5: rifiuta la stima se l'ordinamento del subject non e' affidabile
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
                continue

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

        # Step 5: esclude i pixel occlusi da oggetti genuinamente piu' vicini
        subject_depth = clean_median_depth[subject_class_name]
        subject_x1, subject_y1, subject_x2, subject_y2 = detected_boxes[subject_class_name]

        exclusion_mask = np.zeros((subject_y2 - subject_y1, subject_x2 - subject_x1), dtype=bool)
        for other_class_name, other_depth_value in ordered_by_depth:
            if other_class_name == subject_class_name:
                break

            if (subject_depth - other_depth_value) <= DEPTH_ORDER_EPSILON_M:
                continue

            overlap_key = frozenset((other_class_name, subject_class_name))
            if overlap_key not in overlap_regions:
                continue

            ox1, oy1, ox2, oy2 = overlap_regions[overlap_key]
            exclusion_mask[oy1 - subject_y1:oy2 - subject_y1, ox1 - subject_x1:ox2 - subject_x1] = True

        # Step 6: combina l'esclusione occlusioni con il filtro di validita' depth
        subject_depth_region = depth_image[subject_y1:subject_y2, subject_x1:subject_x2]
        keep_mask = ~exclusion_mask
        subject_valid_mask = keep_mask & self.valid_depth_mask(subject_depth_region)

        rows_local, cols_local = np.where(subject_valid_mask)
        clean_depth_values = subject_depth_region[subject_valid_mask]

        # Step 7: abbastanza pixel puliti per fidarsi della stima?
        original_area = (subject_x2 - subject_x1) * (subject_y2 - subject_y1)
        coverage_ratio = clean_depth_values.size / original_area if original_area > 0 else 0.0

        if clean_depth_values.size < MIN_VALID_PIXELS or coverage_ratio < MIN_COVERAGE_RATIO:
            self.get_logger().warn(
                f"{subject_class_name} too occluded to trust the position estimate "
                f"({clean_depth_values.size} px, {coverage_ratio:.0%} coverage).",
                throttle_duration_sec=5.0
            )
            return None

        # Step 8: media su un piccolo cluster spazialmente coerente vicino alla mediana
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

        point_z = final_depth
        point_x = (final_pixel_x - principal_point_x) * point_z / focal_length_x
        point_y = (final_pixel_y - principal_point_y) * point_z / focal_length_y

        # Step 9: campiona NUM_GRASP_CIRCLE_POINTS punti ad altezza ~costante
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
        """Stima robusta della posizione del target, tenendo conto dell'occlusione da
        parte degli altri oggetti tracciati e dei pixel di sfondo nel bounding box.

        Algoritmo: individua un box per classe (1), marca come "ambigue" le regioni
        di overlap tra coppie di box (2), stima la depth "pulita" di ogni oggetto sui
        soli pixel non ambigui (3) rifiutando ordinamenti non affidabili (3.5), ordina
        gli oggetti per depth (4), esclude dal target solo i pixel condivisi con
        oggetti genuinamente piu' vicini (5), filtra le depth non valide (6), verifica
        che restino abbastanza pixel puliti (7), individua un punto rappresentativo
        stabile vicino alla mediana (8) e campiona i punti del cerchio di presa (9).

        Pubblica un PointStamped se trova una stima affidabile; altrimenti non pubblica nulla.
        """
        image_height, image_width = depth_image.shape[:2]

        detected_boxes = self.collect_tracked_detections(detection_results)
        if not detected_boxes:
            return
        if TARGET_CLASS_NAME not in detected_boxes:
            return

        overlap_regions = {}
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

        clean_median_depth = {}
        clean_pixel_stats = {}

        for class_name, (x1, y1, x2, y2) in detected_boxes.items():
            region = depth_image[y1:y2, x1:x2]
            clean_pixels_mask = ~ambiguous_mask[class_name][y1:y2, x1:x2]
            clean_values = region[clean_pixels_mask]
            clean_values = clean_values[self.valid_depth_mask(clean_values)]

            original_area = (x2 - x1) * (y2 - y1)
            coverage_ratio = clean_values.size / original_area if original_area > 0 else 0.0
            clean_pixel_stats[class_name] = (clean_values.size, coverage_ratio)

            if clean_values.size < MIN_PIXELS_FOR_ORDERING:
                continue
            clean_median_depth[class_name] = np.median(clean_values)

        # Step 4: ordina gli oggetti tracciati dal piu' vicino al piu' lontano
        ordered_by_depth = sorted(clean_median_depth.items(), key=lambda item: item[1])

        target_result = self.compute_object_estimate(
            TARGET_CLASS_NAME, detected_boxes, overlap_regions,
            clean_median_depth, clean_pixel_stats, ordered_by_depth,
            depth_image, focal_length_x, focal_length_y, principal_point_x, principal_point_y
        )
        if target_result is None:
            return

        point_x, point_y, point_z, target_grasp_points = target_result

        self.publish_object_position(point_x, point_y, point_z, header)

        object_entries = [
            TrackedObjectPoints(
                class_name=TARGET_CLASS_NAME,
                shape=OBJECT_SHAPES.get(TARGET_CLASS_NAME, ''),
                points=target_grasp_points,
            )
        ]
        anchor_entries = [
            TrackedObjectPoints(
                class_name=TARGET_CLASS_NAME,
                shape=OBJECT_SHAPES.get(TARGET_CLASS_NAME, ''),
                points=[Point(x=point_x, y=point_y, z=point_z)],
            )
        ]

        for other_class_name in detected_boxes:
            if other_class_name == TARGET_CLASS_NAME:
                continue

            other_result = self.compute_object_estimate(
                other_class_name, detected_boxes, overlap_regions,
                clean_median_depth, clean_pixel_stats, ordered_by_depth,
                depth_image, focal_length_x, focal_length_y, principal_point_x, principal_point_y
            )
            if other_result is None:
                continue

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
        """Pubblica il punto ancora di ogni oggetto come sfera blu, comune a entrambe le modalita'."""
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