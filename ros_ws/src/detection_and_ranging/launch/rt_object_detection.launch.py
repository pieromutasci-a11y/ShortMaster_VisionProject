"""Avvia insieme il nodo di detection+ranging e RViz gia' configurato per
visualizzare RGB annotato, depth (point cloud) e il marker della posizione
stimata dell'oggetto.

Uso:
  ros2 launch detection_and_ranging rt_object_detection.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Se true, avvia anche RViz con la vista pre-configurata.',
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            FindPackageShare('detection_and_ranging'),
            'rviz', 'complete_visualization.rviz',
        ]),
        description='File di configurazione RViz da usare.',
    )

    detection_node = Node(
        package='detection_and_ranging',
        executable='rt_object_detection',
        name='rt_object_detection_node',
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        rviz_arg,
        rviz_config_arg,
        detection_node,
        rviz_node,
    ])
