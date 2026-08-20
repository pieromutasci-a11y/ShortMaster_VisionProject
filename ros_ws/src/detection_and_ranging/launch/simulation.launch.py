"""Avvia la simulazione Gazebo del TIAGo Pro, comoda da qui per chi lavora
sulla detection senza dover ricordare il nome del pacchetto vision_pipeline.

E' un semplice include del launch canonico in vision_pipeline (nessuna
duplicazione della configurazione della simulazione).

Uso:
  ros2 launch detection_and_ranging simulation.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world_name_arg = DeclareLaunchArgument('world_name', default_value='poliBaMaster')
    is_public_sim_arg = DeclareLaunchArgument('is_public_sim', default_value='True')
    slam_arg = DeclareLaunchArgument('slam', default_value='True')
    navigation_arg = DeclareLaunchArgument('navigation', default_value='True')

    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('vision_pipeline'), 'launch', 'simulation.launch.py',
        ])),
        launch_arguments={
            'world_name': LaunchConfiguration('world_name'),
            'is_public_sim': LaunchConfiguration('is_public_sim'),
            'slam': LaunchConfiguration('slam'),
            'navigation': LaunchConfiguration('navigation'),
        }.items(),
    )

    return LaunchDescription([
        world_name_arg,
        is_public_sim_arg,
        slam_arg,
        navigation_arg,
        simulation_launch,
    ])
