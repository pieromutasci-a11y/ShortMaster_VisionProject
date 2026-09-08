"""
Real-time MULTI-object detection node.

Versione multi-oggetto di rt_object_detection_node.py: invece di inseguire una
sola classe ('coke can') e fermarsi alla prima detection valida, rileva TUTTE
le istanze di TUTTE le classi in TRACKED_CLASSES e ne pubblica la posizione 3D.

Come si trasporta piu' di un oggetto restando su geometry_msgs/PointStamped
(che porta un punto solo e nessuna etichetta):

  - la CLASSE sta nel nome del topic: un topic per classe,
    yolo_all/<classe>_position (vedi topic_slug()). Chi ascolta sa la classe
    da quale subscription ha ricevuto il messaggio.
  - le ISTANZE MULTIPLE sono piu' messaggi sullo stesso topic, tutti con lo
    stesso header.stamp (quello del frame RGB da cui provengono): tre lattine
    in scena = tre PointStamped per frame su yolo_all/coke_can_position.

I topic sono volutamente distinti da quelli del nodo single-object
(yolo/...), cosi' i due nodi possono girare insieme senza collidere.

Rispetto al nodo single-object la stima della depth e' piu' robusta, perche'
con piu' oggetti in scena il singolo pixel al centro del bbox non basta piu':
si usa la mediana di una patch centrale mascherata (valid_depth_mask), da cui
sono esclusi i pixel coperti dal bbox di un oggetto piu' vicino
(bbox_intersection).
"""

# Standard library imports
import threading
import time
import traceback
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


# Classi da rilevare. I nomi devono corrispondere esattamente a quelli in
# vision_pipeline/data/training_dataset.yolov8/data.yaml.
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

# Valori di depth della Intel RealSense D435 montata sul Tiago Pro
MIN_RANGE = 0.3
MAX_RANGE = 3.0

# Frazione centrale del bounding box usata per campionare la depth: prendere
# solo il cuore del box evita i bordi dell'oggetto, dove la depth e' rumorosa
# o appartiene gia' allo sfondo.
DEPTH_PATCH_FRACTION = 0.3

# Confronto tra modelli in corso (vedi vision_pipeline/models/runs/): al
# momento in uso il modello di SEGMENTAZIONE (YOLOv8-seg, non detection) --
# per tornare a un altro modello commentare/scommentare la riga giusta tra
# queste. Un modello -seg pubblica anche le maschere per istanza oltre alle
# bounding box, ma qui contano solo queste ultime (detection_results.boxes)
# -- stesso codice, nessuna modifica necessaria per usarlo. detection_results.plot()
# in piu' disegnera' anche i contorni della maschera sul frame annotato.
# MODEL_WEIGHTS_PATH = "/home/user/ros_workspace/src/vision_pipeline/models/wandb/runs/sqkfh2ka/training/xl1874f6/weights/best.pt"
# MODEL_WEIGHTS_PATH = "/home/user/ros_workspace/src/vision_pipeline/models/runs/small_omogeneous_dataset_model_best.pt"
# MODEL_WEIGHTS_PATH = "/home/user/ros_workspace/src/vision_pipeline/models/runs/small_eterogeneous_dataset_model_best.pt"
MODEL_WEIGHTS_PATH = "/home/user/ros_workspace/src/vision_pipeline/models/runs/segmentation_model_best.pt"


def topic_slug(class_name):
    """'coke can' -> 'coke_can'. Deve restare identica a quella in center_computation_all.py."""
    return class_name.replace(' ', '_')


def normalize_yolo_class_name(raw_class_name):
    """
    Normalizza il nome classe grezzo letto da model.names PRIMA di
    confrontarlo con TRACKED_CLASSES.

    Emerso confrontando modelli diversi in models/runs/ per lo stesso
    confronto: run di training diverse etichettano leggermente diverso
    anche per le STESSE classi (es. il modello di segmentazione ha
    'coke_can' con underscore invece di uno spazio, e 'pringles  can' con
    due spazi invece di uno) -- un confronto con uguaglianza esatta le
    scarta silenziosamente (nessun errore, nessun centro pubblicato, il
    sintomo e' "non succede niente"). Underscore -> spazio, poi qualsiasi
    sequenza di spazi ridotta a uno solo: rende il confronto robusto a
    queste variazioni senza bisogno di un caso speciale per ogni modello.
    """
    return ' '.join(raw_class_name.replace('_', ' ').split())


class BoundingBoxPixels:
    """
    Bounding box in coordinate pixel della DEPTH image, con la depth stimata.

    Serve come struttura di appoggio fra le due fasi del processing: prima si
    stima la depth di ogni detection, poi (ordinando per depth crescente) si
    ri-stimano quelle piu' lontane escludendo i pixel occlusi da quelle piu'
    vicine. Ha gli attributi x1/y1/x2/y2 attesi da bbox_intersection().
    """

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

        self.annotated_frame_pub = self.create_publisher(Image, 'yolo_all/annotated_frame', 10)

        # Un publisher PointStamped per classe: e' il nome del topic a dire di
        # che oggetto si tratta, dato che PointStamped non ha campi per la classe.
        self.position_pubs = {
            class_name: self.create_publisher(
                PointStamped, f'yolo_all/{topic_slug(class_name)}_position', 10
            )
            for class_name in TRACKED_CLASSES
        }

        # --- Worker thread ---

        # Detection runs on a separate thread so that YOLO inference (which can
        # be relatively slow) never blocks the ROS executor's subscription callbacks.
        self.worker_running = True
        self.worker_thread = threading.Thread(target=self.detection_worker_loop, daemon=True)
        self.worker_thread.start()

        self.get_logger().info(
            f'Detection multi-oggetto avviata sulle classi: {", ".join(TRACKED_CLASSES)}'
        )

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
                # Tipo + traceback, non solo str(error): con un KeyError il
                # messaggio e' la sola chiave (es. "16"), che da sola non dice
                # ne' dove ne' in quale dizionario e rende impossibile capire.
                self.get_logger().error(
                    f"Error while processing frame: {type(error).__name__}: {error}\n"
                    f"{traceback.format_exc()}"
                )

    def process_frame_pair(self, rgb_msg, depth_msg):
        """Run detection on one RGB/depth pair and publish every detected object."""

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

        # Publish the annotated frame for visualisation/debugging.
        #
        # NOTA: NON usare cv2_to_imgmsg(..., encoding='bgr8'). Quel ramo di
        # cv_bridge fa `self.cvtype_to_name[self.encoding_to_cvtype2(encoding)]`,
        # e cvtype_to_name viene costruito con getattr(cv2, 'CV_8UC3'): nelle
        # versioni recenti di OpenCV quelle costanti non sono piu' interi, il
        # dizionario finisce indicizzato con oggetti diversi e la ricerca per
        # intero esplode con `KeyError: 16` (16 = CV_8UC3) a ogni frame.
        # Con 'passthrough' cv_bridge salta quel controllo; l'encoding lo
        # scriviamo noi dopo, ed e' corretto perche' Results.plot() restituisce
        # sempre un array uint8 HxWx3 in ordine BGR, cioe' esattamente 'bgr8'.
        annotated_image = detection_results.plot()
        annotated_image_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, encoding='passthrough')
        annotated_image_msg.encoding = 'bgr8'
        annotated_image_msg.header = rgb_msg.header
        self.annotated_frame_pub.publish(annotated_image_msg)

        # I bounding box di YOLO sono in pixel dell'immagine RGB, ma la depth va
        # indicizzata in pixel dell'immagine di depth: se le due risoluzioni
        # differiscono serve un fattore di scala (no-op quando coincidono).
        rgb_height, rgb_width = rgb_image.shape[:2]
        depth_height, depth_width = depth_image.shape[:2]
        scale_x = depth_width / rgb_width
        scale_y = depth_height / rgb_height

        # --- Fase 1: raccogli le detection valide, in pixel depth ---
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

        # --- Fase 2: prima stima della depth di ognuno, senza occlusioni ---
        # Serve solo a sapere chi sta davanti a chi.
        for box in boxes:
            box.depth = self.estimate_box_depth(box, depth_image, occluders=[])

        boxes = [box for box in boxes if box.depth is not None]
        boxes.sort(key=lambda b: b.depth)

        # --- Fase 3: ri-stima escludendo i pixel degli oggetti piu' vicini ---
        # Un oggetto davanti a un altro "buca" il bbox di quello dietro: se non
        # si escludono quei pixel, la depth dell'oggetto occluso viene tirata
        # verso quella dell'occlusore.
        for index, box in enumerate(boxes):
            occluders = boxes[:index]   # gia' ordinati per depth crescente
            refined_depth = self.estimate_box_depth(box, depth_image, occluders=occluders)
            if refined_depth is not None:
                box.depth = refined_depth

            position = self.depth_to_3d_point(
                box, focal_length_x, focal_length_y,
                principal_point_x, principal_point_y
            )
            if position is None:
                continue

            point_x, point_y, point_z = position
            # NOTA: si usa l'header del frame RGB, come nel nodo single-object.
            # Formalmente il punto e' ricostruito con le intrinseche della camera
            # depth e starebbe nel frame ottico depth, ma sul TIAGo Pro la depth
            # e' registrata sul color e il nodo esistente funziona cosi': non
            # cambiamo il frame per non introdurre una regressione silenziosa.
            self.publish_object_position(
                box.class_name, point_x, point_y, point_z, rgb_msg.header
            )

    # ------------------------------------------------------------------
    # Stima della depth
    # ------------------------------------------------------------------

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

    def estimate_box_depth(self, box, depth_image, occluders):
        """
        Stima la depth di un oggetto come MEDIANA dei valori validi in una patch
        centrale del suo bounding box, escludendo i pixel coperti dal bbox di un
        oggetto piu' vicino (gli 'occluders').

        La mediana e' molto piu' stabile del singolo pixel centrale usato dal
        nodo single-object: resiste ai buchi della depth (0/NaN sulle superfici
        riflettenti, tipico di una lattina) e a qualche pixel di sfondo che
        finisce dentro il box.

        Ritorna None se non resta nessun pixel valido.
        """
        patch = self.central_patch(box)
        if patch is None:
            return None

        px1, py1, px2, py2 = patch
        depth_region = depth_image[py1:py2, px1:px2]
        if depth_region.size == 0:
            return None

        mask = self.valid_depth_mask(depth_region)

        # Scarta i pixel che appartengono a un oggetto piu' vicino, che qui
        # coprirebbe l'oggetto in esame.
        patch_box = BoundingBoxPixels(px1, py1, px2, py2, box.class_name, box.confidence)
        for occluder in occluders:
            # Un box che CONTIENE interamente quello in esame non e' un
            # occlusore ma un contenitore: e' il caso del tavolo, il cui bbox
            # racchiude tutti gli oggetti che ci stanno sopra. Trattarlo come
            # occlusore cancellerebbe l'intera patch e ci lascerebbe senza
            # nessun pixel valido.
            if self.contains(occluder, box):
                continue
            overlap = self.bbox_intersection(patch_box, occluder)
            if overlap is None:
                continue
            ox1, oy1, ox2, oy2 = overlap
            # coordinate dell'overlap relative alla patch
            mask[oy1 - py1:oy2 - py1, ox1 - px1:ox2 - px1] = False

        valid_values = depth_region[mask]
        if valid_values.size == 0:
            # Fallback: il singolo pixel al centro del box, come nel nodo
            # single-object. Meglio una stima grezza che nessuna stima.
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

        # Reject invalid depth readings — zero or NaN are common at object
        # edges or on reflective surfaces (a metal can is a good example)
        if depth_value <= 0.0 or depth_value != depth_value:
            return None
        return depth_value

    def depth_to_3d_point(self, box, focal_length_x, focal_length_y,
                          principal_point_x, principal_point_y):
        """
        Convert a bounding box centre + its estimated depth into a 3D position
        (X, Y, Z), expressed in the camera's optical frame, using the pinhole
        camera model. Returns None if the box has no valid depth.
        """
        if box.depth is None:
            return None

        pixel_x = (box.x1 + box.x2) // 2
        pixel_y = (box.y1 + box.y2) // 2

        point_z = box.depth
        point_x = (pixel_x - principal_point_x) * point_z / focal_length_x
        point_y = (pixel_y - principal_point_y) * point_z / focal_length_y

        return point_x, point_y, point_z

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_object_position(self, class_name, point_x, point_y, point_z, header):
        """Publish one detected object's 3D position on its class's topic."""
        position_msg = PointStamped()
        position_msg.header = header
        position_msg.point.x = point_x
        position_msg.point.y = point_y
        position_msg.point.z = point_z
        self.position_pubs[class_name].publish(position_msg)

    @staticmethod
    def contains(outer, inner):
        """True se il box 'outer' racchiude interamente 'inner'."""
        return (outer.x1 <= inner.x1 and outer.y1 <= inner.y1
                and outer.x2 >= inner.x2 and outer.y2 >= inner.y2)

    @staticmethod
    def clamp(value, lower, upper):
        return min(max(value, lower), upper)

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
