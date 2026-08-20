"""Avvia l'intero stack di detection in un colpo solo: simulazione Gazebo
del TIAGo Pro + nodo di detection/ranging + RViz.

Uso:
  ros2 launch detection_and_ranging full_stack.launch.py
  # se la simulazione e' gia' avviata altrove:
  ros2 launch detection_and_ranging full_stack.launch.py launch_simulation:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_simulation_arg = DeclareLaunchArgument(
        'launch_simulation',
        default_value='True',
        description='Se true, avvia anche la simulazione Gazebo del TIAGo Pro.',
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='True',
        description='Se true, avvia anche RViz con la vista pre-configurata.',
    )

    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('detection_and_ranging'), 'launch', 'simulation.launch.py',
        ])),
        condition=IfCondition(LaunchConfiguration('launch_simulation')),
    )

    detection_and_rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('detection_and_ranging'), 'launch', 'rt_object_detection.launch.py',
        ])),
        launch_arguments={'rviz': LaunchConfiguration('rviz')}.items(),
    )

    return LaunchDescription([
        launch_simulation_arg,
        rviz_arg,
        simulation_launch,
        detection_and_rviz_launch,
    ])
