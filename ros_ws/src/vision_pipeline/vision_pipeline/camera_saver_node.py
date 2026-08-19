#!/usr/bin/env python3
"""
Salva su disco un frame ogni N ricevuti dalla camera del robot, per costruire
un dataset di immagini da annotare (es. su Roboflow) e usare per il training YOLO.

La cartella di destinazione e' configurabile via parametro ROS 'save_dir', cosi'
da poter lanciare piu' sessioni di raccolta (es. una per ogni posa dell'oggetto)
senza modificare il codice:

  ros2 run vision_pipeline camera_saver --ros-args -p save_dir:=/home/user/ros_workspace/src/vision_pipeline/data/raw_captures/coke_nordest
"""
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import cv2


DEFAULT_SAVE_DIR = '/home/user/ros_workspace/src/vision_pipeline/data/raw_captures/new_capture'
DEFAULT_SAVE_EVERY_N = 5
DEFAULT_IMAGE_TOPIC = '/head_front_camera/color/image_raw'


class CameraSaver(Node):
    def __init__(self):
        super().__init__('camera_saver')

        self.declare_parameter('save_dir', DEFAULT_SAVE_DIR)
        self.declare_parameter('save_every_n', DEFAULT_SAVE_EVERY_N)
        self.declare_parameter('image_topic', DEFAULT_IMAGE_TOPIC)

        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        self.save_every_n = self.get_parameter('save_every_n').get_parameter_value().integer_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.counter = 0
        self.saved_count = 0

        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f'Node started. Saving 1 image every {self.save_every_n} frame in: {self.save_dir}'
        )

    def image_callback(self, msg):
        self.counter += 1

        if self.counter % self.save_every_n != 0:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        filename = os.path.join(self.save_dir, f'frame_{self.saved_count:06d}.png')
        cv2.imwrite(filename, cv_image)

        self.saved_count += 1
        self.get_logger().info(f'Saved image: {filename}')


def main(args=None):
    rclpy.init(args=args)
    node = CameraSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
