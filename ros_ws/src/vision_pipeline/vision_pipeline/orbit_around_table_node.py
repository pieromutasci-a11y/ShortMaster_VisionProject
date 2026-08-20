#!/usr/bin/env python3
"""
Fa percorrere al robot un'orbita circolare attorno al tavolo usando il movimento
LATERALE delle ruote mecanum/omnidirezionali (linear.x + linear.y nel frame del
robot), mantenendo l'orientamento sempre rivolto verso il centro del tavolo.

Vantaggio rispetto a un robot differenziale: il robot non deve ruotare per
allinearsi alla direzione di marcia, quindi la testa/camera resta sempre
puntata verso il tavolo durante tutta l'orbita, e il movimento e' molto piu'
stabile (niente rischio di "girare su se stesso" verso il tavolo).

Topic usati (verificati sul robot):
  /mobile_base_controller/odom       -> odometria (ruote, non serve lidar)
  /cmd_vel                           -> comandi di velocita' (linear.x, linear.y, angular.z)
  /head_controller/joint_trajectory  -> movimento testa
"""

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


# ---- Parametri configurabili ----
CENTER_X = 5.0          # centro del tavolo (da poliBaMaster.world)
CENTER_Y = 5.0
RADIUS = 1.0            # raggio dell'orbita (metri) - corrisponde alla distanza di spawn del robot dal tavolo

ORBIT_ANGULAR_SPEED = 0.2   # velocita' angolare di percorrenza dell'orbita (rad/s)
ORBIT_DIRECTION = -1         # 1 = antiorario, -1 = orario (verso sinistra del robot dato lo yaw di partenza)
ANGLE_STEP_DEG = 30          # ogni quanti gradi fermarsi per muovere la testa

RADIUS_GAIN = 0.2       # guadagno correzione raggio (mantiene la distanza dal tavolo)
YAW_GAIN = 1.5           # guadagno correzione orientamento (per restare rivolti al centro)

HEAD_PAN = 0.0
HEAD_TILT_UP = 0.3
HEAD_TILT_DOWN = -0.5
HEAD_MOVE_TIME = 1.5
PAUSE_AFTER_HEAD_SEC = 0.5

MAX_LINEAR_SPEED = 0.3   # limite di sicurezza sulla velocita' lineare (m/s)


class CircleAroundTableOmni(Node):
    def __init__(self):
        super().__init__('circle_around_table_omni')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.head_pub = self.create_publisher(
            JointTrajectory, '/head_controller/joint_trajectory', 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/mobile_base_controller/odom', self.odom_callback, 10
        )

        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.last_stop_step = None
        self.state = 'WAIT_ODOM'   # WAIT_ODOM -> DRIVING -> PAUSED -> DRIVING -> ...

        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        self.get_logger().info(
            f'Nodo avviato. Orbita omnidirezionale attorno a ({CENTER_X}, {CENTER_Y}), raggio {RADIUS} m.'
        )

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        if self.state == 'WAIT_ODOM':
            self.state = 'DRIVING'
            self.get_logger().info('Odometria ricevuta, inizio orbita.')

    def control_loop(self):
        if self.state in ('WAIT_ODOM', 'PAUSED'):
            return

        # --- Vettore dal centro al robot ---
        dx = self.current_x - CENTER_X
        dy = self.current_y - CENTER_Y
        r = math.sqrt(dx * dx + dy * dy)
        angle = math.atan2(dy, dx)  # angolo attuale sull'orbita

        angle_deg = math.degrees(angle) % 360
        step_index = int(angle_deg // ANGLE_STEP_DEG)

        if self.last_stop_step is None:
            self.last_stop_step = step_index

        if step_index != self.last_stop_step:
            self.last_stop_step = step_index
            self.stop_and_move_head()
            return

        # --- Direzione tangenziale (world frame), verso impostato da ORBIT_DIRECTION ---
        tangent_x = -math.sin(angle) * ORBIT_DIRECTION
        tangent_y = math.cos(angle) * ORBIT_DIRECTION

        # --- Correzione radiale per mantenere il raggio (world frame) ---
        # verso positivo = verso il centro se r > RADIUS, verso l'esterno se r < RADIUS
        radial_error = RADIUS - r
        radial_x = -dx / r if r > 1e-3 else 0.0
        radial_y = -dy / r if r > 1e-3 else 0.0

        # --- Velocita' desiderata nel frame mondo ---
        vx_world = (tangent_x * ORBIT_ANGULAR_SPEED * RADIUS) + (radial_x * RADIUS_GAIN * radial_error)
        vy_world = (tangent_y * ORBIT_ANGULAR_SPEED * RADIUS) + (radial_y * RADIUS_GAIN * radial_error)

        # --- Orientamento desiderato: robot sempre rivolto verso il centro tavolo ---
        yaw_desired = math.atan2(CENTER_Y - self.current_y, CENTER_X - self.current_x)
        yaw_error = self.normalize_angle(yaw_desired - self.current_yaw)

        # --- Trasformazione velocita' da frame mondo a frame robot (per mecanum) ---
        vx_body = vx_world * math.cos(self.current_yaw) + vy_world * math.sin(self.current_yaw)
        vy_body = -vx_world * math.sin(self.current_yaw) + vy_world * math.cos(self.current_yaw)

        # --- Limite di sicurezza ---
        vx_body = max(-MAX_LINEAR_SPEED, min(MAX_LINEAR_SPEED, vx_body))
        vy_body = max(-MAX_LINEAR_SPEED, min(MAX_LINEAR_SPEED, vy_body))

        msg = Twist()
        msg.linear.x = vx_body
        msg.linear.y = vy_body
        msg.angular.z = YAW_GAIN * yaw_error
        self.cmd_vel_pub.publish(msg)

    def stop_and_move_head(self):
        self.state = 'PAUSED'
        self.get_logger().info('Fermo il robot, muovo la testa su/giu...')

        self.cmd_vel_pub.publish(Twist())
        time.sleep(0.3)

        self.send_head(HEAD_PAN, HEAD_TILT_UP)
        time.sleep(HEAD_MOVE_TIME)

        self.send_head(HEAD_PAN, HEAD_TILT_DOWN)
        time.sleep(HEAD_MOVE_TIME)

        self.send_head(HEAD_PAN, 0.0)
        time.sleep(HEAD_MOVE_TIME)

        time.sleep(PAUSE_AFTER_HEAD_SEC)

        self.get_logger().info('Riprendo l\'orbita.')
        self.state = 'DRIVING'

    def send_head(self, pan, tilt):
        msg = JointTrajectory()
        msg.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.time_from_start = Duration(sec=1)
        msg.points = [point]
        self.head_pub.publish(msg)

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = CircleAroundTableOmni()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_vel_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
