#!/usr/bin/env python3
"""
Keyboard teleoperation node for PAL robots (e.g. TIAGo).

Controls:
  W / S           -> forward / backward (base)
  A / D           -> turn left / right (base)
  I / P           -> head: tilt up / down
  J / L           -> head: pan left / right
  Q / E           -> torso: up / down
  SPACE           -> immediate base stop
  ESC / CTRL+C    -> exit

Note: make sure the topic/joint names match your robot.
Verify with:
  ros2 topic list | grep cmd_vel
  ros2 topic list | grep head
  ros2 topic list | grep torso
  ros2 topic echo /joint_states --once
"""

import sys
import termios
import tty
import select
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


LINEAR_SPEED = 0.2
ANGULAR_SPEED = 0.5

HEAD_STEP = 0.1
HEAD_PAN_MIN, HEAD_PAN_MAX = -1.24, 1.24
HEAD_TILT_MIN, HEAD_TILT_MAX = -0.98, 0.79

TORSO_STEP = 0.02
TORSO_MIN, TORSO_MAX = 0.0, 0.35

MOVE_DURATION_SEC = 1


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.head_pub = self.create_publisher(
            JointTrajectory, '/head_controller/joint_trajectory', 10
        )
        self.head_pan = 0.0
        self.head_tilt = 0.0

        self.torso_pub = self.create_publisher(
            JointTrajectory, '/torso_controller/joint_trajectory', 10
        )
        self.torso_pos = 0.15

        self.get_logger().info('Teleop node started. Use WASD, arrows, Q/E. Ctrl+C to exit.')

    def send_cmd_vel(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_vel_pub.publish(msg)

    def stop_base(self):
        self.send_cmd_vel(0.0, 0.0)

    def send_head(self):
        self.head_pan = max(HEAD_PAN_MIN, min(HEAD_PAN_MAX, self.head_pan))
        self.head_tilt = max(HEAD_TILT_MIN, min(HEAD_TILT_MAX, self.head_tilt))

        msg = JointTrajectory()
        msg.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [self.head_pan, self.head_tilt]
        point.time_from_start = Duration(sec=MOVE_DURATION_SEC)
        msg.points = [point]

        self.head_pub.publish(msg)
        self.get_logger().info(f'Head -> pan={self.head_pan:.2f}, tilt={self.head_tilt:.2f}')

    def send_torso(self):
        self.torso_pos = max(TORSO_MIN, min(TORSO_MAX, self.torso_pos))

        msg = JointTrajectory()
        msg.joint_names = ['torso_lift_joint']
        point = JointTrajectoryPoint()
        point.positions = [self.torso_pos]
        point.time_from_start = Duration(sec=MOVE_DURATION_SEC)
        msg.points = [point]

        self.torso_pub.publish(msg)
        self.get_logger().info(f'Torso -> {self.torso_pos:.2f} m')


def get_key(settings, timeout=0.1):
    """Read a single key from stdin (including escape sequences for arrow keys)."""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        key = sys.stdin.read(1)
        if key == '\x1b':
            rlist2, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist2:
                key2 = sys.stdin.read(1)
                if key2 == '[':
                    key3 = sys.stdin.read(1)
                    key = '\x1b[' + key3
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()

    settings = termios.tcgetattr(sys.stdin)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        print(__doc__)
        while rclpy.ok():
            key = get_key(settings)

            if key == 'w':
                node.send_cmd_vel(LINEAR_SPEED, 0.0)
            elif key == 's':
                node.send_cmd_vel(-LINEAR_SPEED, 0.0)
            elif key == 'a':
                node.send_cmd_vel(0.0, ANGULAR_SPEED)
            elif key == 'd':
                node.send_cmd_vel(0.0, -ANGULAR_SPEED)
            elif key == ' ':
                node.stop_base()

            elif key == 'i':
                node.head_tilt += HEAD_STEP
                node.send_head()
            elif key == 'p':
                node.head_tilt -= HEAD_STEP
                node.send_head()
            elif key == 'j':
                node.head_pan -= HEAD_STEP
                node.send_head()
            elif key == 'l':
                node.head_pan += HEAD_STEP
                node.send_head()

            elif key == 'q':
                node.torso_pos += TORSO_STEP
                node.send_torso()
            elif key == 'e':
                node.torso_pos -= TORSO_STEP
                node.send_torso()

            elif key == '\x03' or key == '\x1b':
                break

    finally:
        node.stop_base()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
