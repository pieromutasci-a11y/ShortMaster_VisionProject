#!/usr/bin/env python3
"""
Center computation node, versione MULTI-oggetto.

Versione multi-oggetto di center_computation.py: invece di correggere il solo
punto della lattina di coca, riceve i punti di superficie di tutte le classi
in RADIUS_BY_CLASS e ne stima il centro dell'asse verticale, uno per istanza.

La geometria e' identica a quella del nodo single-object, ed e' quella la
parte delicata:

IMPORTANTE: la correzione (spostare il punto di un raggio noto lungo la
componente ORIZZONTALE del raggio camera->punto) e' un ragionamento valido
solo in un frame dove l'asse Z e' davvero verticale (allineato con la
gravita'), tipo TARGET_FRAME ('base_footprint'). Il frame della camera NON
va bene: la camera e' inclinata (guarda il tavolo dall'alto), quindi il suo
piano x,y locale non e' affatto il piano orizzontale vero. Per questo il
punto viene prima trasformato in TARGET_FRAME via TF, e SOLO DOPO si applica
la correzione -- farlo nell'ordine inverso sposta il punto nella direzione
sbagliata.

  - x, y: corretti spostando il punto (gia' in TARGET_FRAME) del raggio della
    sua classe lungo la componente orizzontale della direzione camera->punto.
    La posizione della camera in TARGET_FRAME si ottiene gratis dalla
    traduzione della trasformazione TF stessa.
  - z: lasciata invariata. Lo spostamento orizzontale non cambia l'altezza, e
    la z misurata e' gia' un punto valido sull'asse vero -- niente bisogno di
    conoscere l'altezza del tavolo o altra informazione specifica della scena.

Il raggio e' un prior sull'OGGETTO (non cambia se cambia la scena), quindi
resta valido su tavoli/mondi diversi -- a differenza di un'altezza del tavolo
hardcoded, che sarebbe un prior sulla SCENA e non generalizzerebbe.

Il calcolo avviene internamente in TARGET_FRAME (deve, per il motivo sopra),
ma il punto corretto viene ri-trasformato indietro nel frame originale del
messaggio in ingresso prima di pubblicarlo -- cosi' resta nello stesso
sistema di riferimento del punto di superficie, comodo per confrontare i due
direttamente. Il marker RViz invece va disegnato in TARGET_FRAME (deve essere
davvero verticale), quindi usa la versione del centro non ri-trasformata.

--- Cosa cambia rispetto al nodo single-object ---

Come in rt_object_detection_node_all.py, l'informazione che PointStamped e
Marker non sanno portare sta fuori dal payload:

  - la CLASSE sta nel nome del topic: una subscription per classe su
    yolo_all/<classe>_position, e publisher speculari su
    centers_all/<classe>_center, centers_all/<classe>_center_base_footprint
    e centers_all/<classe>_axis. La callback sa gia' quale raggio applicare
    perche' sa da quale topic e' stata chiamata.
  - come nel nodo single-object, il centro corretto va su DUE topic, stesso
    punto in due frame: centers_all/<classe>_center (ri-trasformato nel
    frame camera, confrontabile con yolo_all/<classe>_position) e
    centers_all/<classe>_center_base_footprint (gia' in TARGET_FRAME, pronto
    per chi lo usa nel frame del robot).
  - le ISTANZE MULTIPLE sono piu' messaggi consecutivi sullo stesso topic,
    riconoscibili perche' condividono header.stamp (vengono dallo stesso
    frame camera). Il contatore degli id marker si azzera al cambio di stamp,
    cosi' le lattine di uno stesso frame prendono id 0, 1, 2...
  - i marker hanno una lifetime breve: quando un oggetto esce di scena il suo
    marker scade da solo, senza dover tenere traccia di quali id cancellare.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registra il supporto a PointStamped per do_transform_point)
from tf2_geometry_msgs import do_transform_point


# Raggio noto per classe (prior sull'oggetto), in metri.
#
# ATTENZIONE: questa correzione modella un CILINDRO. Vale in pieno per le due
# lattine; per le altre due classi il valore e' una scelta consapevole:
#   - 'biscuits pack' e' una SCATOLA: mezzo spessore come raggio equivalente e'
#     un'approssimazione valida solo se la faccia inquadrata e' grossomodo
#     frontale alla camera, e peggiora man mano che la si guarda di sbieco.
#   - 'dinner table' non ha un "asse" con senso fisico: raggio 0.0, cioe'
#     nessuna correzione. Il centro pubblicato coincide col punto di
#     superficie -- il topic esiste ed e' utilizzabile come ingombro, ma non
#     inventiamo uno spostamento che non avrebbe significato.
RADIUS_BY_CLASS = {
    'coke can':      0.04,   # valore del nodo single-object, gia' verificato in simulazione
    'pringles can':  0.04,   # da tarare sul modello Gazebo effettivo
    'biscuits pack': 0.03,   # mezzo spessore (approssimazione, vedi sopra)
    'dinner table':  0.0,    # nessuna correzione (vedi sopra)
}

# Colore dell'asse per classe (r, g, b in 0..1), cosi' a colpo d'occhio si
# capisce quale oggetto e' quale. Sono le stesse tinte usate dalle sfere dei
# punti di superficie in complete_visualization_all.rviz: ogni classe ha un
# solo colore, e a distinguere superficie da centro e' la forma (sfera contro
# segmento), non il colore.
COLOR_BY_CLASS = {
    'coke can':      (0.80, 0.16, 0.80),   # magenta
    'pringles can':  (0.16, 0.50, 0.80),   # blu
    'biscuits pack': (1.00, 0.67, 0.00),   # arancione
    'dinner table':  (0.47, 0.47, 0.47),   # grigio
}
DEFAULT_AXIS_COLOR = (0.0, 1.0, 0.0)   # verde, per una classe senza colore assegnato

AXIS_MARKER_LENGTH = 0.40   # lunghezza del segmento disegnato in RViz (solo visualizzazione,
                            # volutamente piu' lungo dell'oggetto vero per essere ben visibile)
# Deve essere piu' lungo dell'intervallo fra due detection successive, altrimenti
# il marker scade prima di essere ripubblicato e si vede lampeggiare: con la
# simulazione rallentata e YOLO su CPU passa piu' di un secondo fra un frame
# elaborato e il successivo.
AXIS_MARKER_LIFETIME_SEC = 3.0
TARGET_FRAME = 'base_footprint'   # frame con asse Z verticale (gravita'), per disegnare l'asse correttamente


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

        # Stato per l'assegnazione degli id marker: per ogni classe teniamo lo
        # stamp dell'ultimo frame visto e quanti oggetti di quella classe sono
        # gia' arrivati in quel frame.
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

            # La lambda cattura class_name come default argument, altrimenti
            # tutte le callback userebbero l'ultimo valore del ciclo.
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

        # Ri-trasforma il centro corretto nel frame originale del messaggio,
        # cosi' il centro e il punto di superficie sono confrontabili
        # direttamente (stesso frame). La trasformazione verso il frame del
        # braccio, se serve, la fara' l'ottimizzatore.
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
        """
        Id progressivo dell'istanza dentro il frame corrente.

        Tutti i punti che il nodo di detection ricava dallo stesso frame camera
        condividono header.stamp: quando lo stamp cambia siamo su un frame
        nuovo e il contatore riparte da zero, cosi' la prima lattina del frame
        e' sempre l'id 0, la seconda l'id 1 e via cosi'.
        """
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
        marker.scale.x = 0.01  # spessore della linea (m)
        red, green, blue = COLOR_BY_CLASS.get(class_name, DEFAULT_AXIS_COLOR)
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 1.0
        # Senza lifetime i marker delle istanze sparite resterebbero appesi in
        # RViz: qui scadono da soli se non vengono ripubblicati.
        marker.lifetime = Duration(
            sec=int(AXIS_MARKER_LIFETIME_SEC),
            nanosec=int((AXIS_MARKER_LIFETIME_SEC % 1) * 1e9),
        )
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
