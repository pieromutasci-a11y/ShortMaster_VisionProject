#!/usr/bin/env python3
"""
Center computation node per lo stack CON gestione occlusioni.

Riceve, per ogni oggetto tracciato, fino a 3 punti sulla sua superficie
visibile (yolo/tracked_objects_points, tiago_vision_msgs/TrackedObjectsArray
-- pubblicato da rt_object_detection_node_occlusion_handler.py in modalita'
robust) e ne ricava il centro dell'asse verticale fittando la circonferenza
che passa per quei 3 punti -- invece di assumere un raggio noto a priori
come fa center_computation_all.py (stack senza gestione occlusioni).

Perche' e' meglio di un raggio noto, quando i 3 punti sono buoni:
  - il raggio non e' piu' un prior fisso uguale per tutte le istanze della
    stessa classe, ma una misura vera, ricavata dai pixel di QUESTO
    oggetto in QUESTO frame -- si adatta anche a un raggio leggermente
    diverso da quello nominale del modello Gazebo.
  - non serve nessuna tabella RADIUS_BY_CLASS per farlo funzionare (resta
    comunque definita, vedi sotto, ma solo come riferimento per il log di
    sanity-check, non come input del calcolo).

--- Perche' il fit avviene nel piano XY di base_footprint, non nello spazio 3D ---

I 3 punti sono presi lungo una riga dell'immagine a depth ~costante, nel
frame ottico della camera -- quel frame e' inclinato (la camera guarda il
tavolo dall'alto), quindi "il piano dei 3 punti" li' non e' affatto il
piano orizzontale vero. Esattamente lo stesso problema gia' risolto in
center_computation_all.py: i punti vanno prima trasformati in
TARGET_FRAME ('base_footprint', asse Z davvero verticale) via TF, e SOLO
DOPO si puo' ragionare in termini di "cerchio orizzontale". Fittare un
cerchio generico per 3 punti nello spazio (senza questo vincolo) sarebbe
piu' debole: ignorerebbe un'informazione vera che gia' abbiamo (l'oggetto
e' un cilindro ad asse verticale) e sarebbe piu' sensibile al rumore sulla
z dei singoli punti.

Con 3 punti nel piano, il centro della circonferenza che passa per tutti e
3 e' la soluzione ESATTA di un sistema lineare (circocentro, formula
chiusa via determinante) -- non serve un fit ai minimi quadrati, che avrebbe
senso solo con piu' di 3 punti.

--- Il caso degenere ---

Quando i 3 punti sono troppo allineati (determinante ~0), il centro non e'
calcolabile in modo affidabile -- un piccolo rumore sposterebbe il
risultato di molto. Succede in almeno due casi:
  1. rt_object_detection_node_occlusion_handler.py non ha trovato nessuna
     riga abbastanza larga (select_grasp_circle_points -> None) e ha
     ripiegato pubblicando lo STESSO punto ripetuto 3 volte -- 3 punti
     coincidenti sono il caso degenere per definizione.
  2. Un oggetto molto occluso da un altro piu' vicino (test: biscotti
     davanti alla coca al ~60%) lascia visibile solo un arco stretto della
     circonferenza -- i 3 punti campionati su quell'arco sono vicini e
     quasi allineati anche quando presi correttamente (non e' un bug nella
     scelta dei punti, e' la geometria: un arco stretto NON contiene
     abbastanza informazione per stimare in modo stabile un cerchio molto
     piu' grande dell'arco stesso).

NOTA: 'biscuits pack' in QUESTO modello Gazebo e' in realta' un cilindro
(radius 0.029), non una scatola come inizialmente assunto -- il fit
funziona per questa classe quanto per le altre due, non e' un caso
degenere strutturale.

Il controllo sull'area del triangolo (MIN_TRIANGLE_AREA_M2) intercetta
solo il caso 1 (area davvero ~0). Nel caso 2 l'area puo' essere piccola ma
non nulla -- il triangolo passa il controllo, ma il centro calcolato puo'
comunque essere molto lontano da quello vero (fit mal condizionato: punti
quasi allineati amplificano il rumore). Per questo c'e' un secondo
controllo, sul RAGGIO misurato (vedi RADIUS_SANITY_FACTOR): un raggio
molto diverso dal nominale della classe e' il sintomo di un fit
inaffidabile anche quando l'area non lo segnala.

Scelta fatta (vedi conversazione): niente fallback a raggio noto -- il
frame viene scartato in entrambi i casi, coerente con la filosofia gia'
in uso altrove nel progetto (MIN_VALID_PIXELS, MIN_ORDERING_COVERAGE_RATIO,
...): meglio nessun dato che un dato inventato. Conseguenza pratica da
tenere a mente: un oggetto molto occluso pubblichera' un centro solo nei
frame in cui l'arco visibile e' abbastanza ampio da dare un fit stabile.

--- Topic: identici a center_computation_all.py, di proposito ---

Cosi' pose_optimizer non richiede NESSUNA modifica: i due stack (con e
senza gestione occlusioni) sono intercambiabili solo scegliendo quale
center_computation lanciare insieme a rt_object_detection_node_occlusion_handler
o rt_object_detection_node_all.py.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registra il supporto a PointStamped per do_transform_point)
from tf2_geometry_msgs import do_transform_point

from tiago_vision_msgs.msg import TrackedObjectsArray


# Raggio nominale per classe (stesso valore di center_computation_all.py),
# usato QUI solo come riferimento per il sanity-check sul raggio misurato
# (vedi RADIUS_SANITY_FACTOR: scarta il frame se troppo diverso), non come
# input del calcolo del centro -- a differenza dello stack senza gestione
# occlusioni.
RADIUS_BY_CLASS = {
    'coke can':      0.04,
    'pringles can':  0.04,
    'biscuits pack': 0.029,   # cilindro anche questo in questo modello Gazebo, vedi sopra
}

# Quanto puo' discostarsi il raggio MISURATO (dal fit) da quello nominale
# prima di scartare il frame come inaffidabile. Fattore moltiplicativo
# (es. 2.0 = scarta se il raggio misurato e' piu' del doppio o meno della
# meta' del nominale). Intercetta il fit mal condizionato (punti quasi
# allineati su un arco stretto, tipicamente per occlusione severa) che
# MIN_TRIANGLE_AREA_M2 da solo non cattura -- vedi il caso 2 sopra.
# TODO: calibrare empiricamente.
RADIUS_SANITY_FACTOR = 2.0

# Area minima (m^2) del triangolo formato dai 3 punti (nel piano XY di
# TARGET_FRAME) perche' il fit sia considerato affidabile. Sotto questa
# soglia i punti sono troppo allineati/coincidenti -- il centro verrebbe
# calcolato ma sarebbe numericamente instabile (piccolo rumore -> grande
# spostamento del centro). Il frame viene scartato piuttosto che
# pubblicare un centro inventato.
# TODO: calibrare empiricamente (verificato solo che 3 punti coincidenti,
# area 0 esatta, vengono correttamente scartati).
MIN_TRIANGLE_AREA_M2 = 1e-5

COLOR_BY_CLASS = {
    'coke can':      (0.80, 0.16, 0.80),   # magenta
    'pringles can':  (0.16, 0.50, 0.80),   # blu
    'biscuits pack': (1.00, 0.67, 0.00),   # arancione
}
DEFAULT_AXIS_COLOR = (0.0, 1.0, 0.0)

AXIS_MARKER_LENGTH = 0.40
TARGET_FRAME = 'base_footprint'   # frame con asse Z verticale (gravita')


def topic_slug(class_name):
    """'coke can' -> 'coke_can'. Deve restare identica a quella usata nello stack _all."""
    return class_name.replace(' ', '_')


def fit_circle_2d(p1, p2, p3):
    """
    Centro della circonferenza passante per 3 punti (x, y) nel piano.

    Soluzione esatta (circocentro) via formula chiusa -- valida solo se i
    3 punti non sono allineati. Ritorna (cx, cy, area_triangolo) sempre:
    il chiamante decide, guardando l'area, se il risultato e' affidabile
    (area troppo piccola -> quasi allineati -> risultato numericamente
    instabile, va scartato).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    area = abs(d) / 4.0  # d = 4 * area con segno del triangolo

    if abs(d) < 1e-12:
        return None, None, area  # esattamente allineati/coincidenti, evita la divisione per zero

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

        # Un solo oggetto per classe alla volta (rt_object_detection_node_occlusion_handler.py
        # tiene una sola bounding box per classe, non supporta istanze
        # multiple) -- l'id marker resta sempre 0, niente contatore di
        # istanze come in center_computation_all.py.
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
        # Un solo lookup TF per messaggio: tutti gli oggetti del messaggio
        # condividono lo stesso frame/istante (stesso frame camera).
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
            return  # classe non tracciata da questo stack (es. dinner table -- comunque mai presente qui)

        if len(entry.points) != 3:
            self.get_logger().warn(
                f'[{class_name}] Attesi 3 punti di superficie, ricevuti {len(entry.points)} -- salto.',
                throttle_duration_sec=5.0
            )
            return

        # Passo 1: porta i 3 punti in TARGET_FRAME (asse Z davvero
        # verticale) PRIMA di ragionare in termini di "cerchio orizzontale"
        # -- stesso motivo, stesso ordine, di center_computation_all.py.
        points_in_target = []
        for point in entry.points:
            point_stamped = PointStamped()
            point_stamped.header = header
            point_stamped.point = point
            transformed = do_transform_point(point_stamped, transform)
            points_in_target.append(transformed.point)

        # Passo 2: fit del cerchio nel piano XY (ORA il piano e' davvero
        # orizzontale).
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

        # Secondo controllo di affidabilita' (oltre all'area del
        # triangolo): il raggio misurato quanto si discosta dal nominale
        # della classe? Un'area non nulla non basta a garantire un fit
        # buono -- su un arco stretto (oggetto molto occluso) i 3 punti
        # possono passare il controllo sull'area e comunque dare un centro
        # molto lontano da quello vero (fit mal condizionato). Se il
        # raggio misurato e' implausibile, il frame viene scartato come
        # per il caso di area quasi nulla -- stessa filosofia.
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

        # Ri-trasforma il centro nel frame originale del messaggio, cosi'
        # e' confrontabile direttamente con i punti di superficie
        # (stesso frame) -- stessa convenzione di center_computation_all.py.
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
        marker.id = 0  # una sola istanza per classe in questo stack
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        red, green, blue = COLOR_BY_CLASS.get(class_name, DEFAULT_AXIS_COLOR)
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 1.0
        # Nessuna lifetime -- stesso motivo di center_computation_all.py:
        # resta a schermo finche' non arriva un aggiornamento, invece di
        # lampeggiare quando YOLO (su CPU) rallenta.
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
