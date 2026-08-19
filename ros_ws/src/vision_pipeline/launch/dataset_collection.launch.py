"""Avvia insieme l'orbita attorno al tavolo e il salvataggio dei frame camera,
per raccogliere una sessione di immagini per il dataset YOLO.

Uso:
  ros2 launch vision_pipeline dataset_collection.launch.py \
      save_dir:=/home/user/ros_workspace/src/vision_pipeline/data/raw_captures/coke_nordest
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    save_dir_arg = DeclareLaunchArgument(
        'save_dir',
        default_value='/home/user/ros_workspace/src/vision_pipeline/data/raw_captures/new_capture',
        description='Cartella dove salvare i frame catturati dalla camera.',
    )
    save_every_n_arg = DeclareLaunchArgument(
        'save_every_n',
        default_value='5',
        description='Salva un frame ogni N ricevuti.',
    )

    camera_saver_node = Node(
        package='vision_pipeline',
        executable='camera_saver',
        name='camera_saver',
        output='screen',
        parameters=[{
            'save_dir': LaunchConfiguration('save_dir'),
            'save_every_n': LaunchConfiguration('save_every_n'),
        }],
    )

    orbit_node = Node(
        package='vision_pipeline',
        executable='orbit_around_table',
        name='circle_around_table_omni',
        output='screen',
    )

    return LaunchDescription([
        save_dir_arg,
        save_every_n_arg,
        camera_saver_node,
        orbit_node,
    ])
