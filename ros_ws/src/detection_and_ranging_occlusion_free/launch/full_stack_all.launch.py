"""Avvia l'intero stack di detection MULTI-oggetto in un colpo solo:
simulazione Gazebo del TIAGo Pro + nodi di detection/ranging e center
computation su tutte le classi + RViz con la vista multi-oggetto.

Uso:
  ros2 launch detection_and_ranging_occlusion_free full_stack_all.launch.py
  # se la simulazione e' gia' avviata altrove:
  ros2 launch detection_and_ranging_occlusion_free full_stack_all.launch.py launch_simulation:=false
  # senza RViz:
  ros2 launch detection_and_ranging_occlusion_free full_stack_all.launch.py rviz:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
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
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            FindPackageShare('detection_and_ranging_occlusion_free'),
            'rviz', 'complete_visualization_all.rviz',
        ]),
        description='File di configurazione RViz da usare.',
    )

    # Simulazione: unica sorgente di verita' in vision_pipeline, incluso qui
    # senza duplicare la configurazione del mondo/robot.
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('vision_pipeline'), 'launch', 'simulation.launch.py',
        ])),
        condition=IfCondition(LaunchConfiguration('launch_simulation')),
    )

    detection_node = Node(
        package='detection_and_ranging_occlusion_free',
        executable='rt_object_detection_all',
        name='rt_object_detection_all_node',
        output='screen',
    )

    center_computation_node = Node(
        package='detection_and_ranging_occlusion_free',
        executable='center_computation_all',
        name='center_computation_all_node',
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
        launch_simulation_arg,
        rviz_arg,
        rviz_config_arg,
        simulation_launch,
        detection_node,
        center_computation_node,
        rviz_node,
    ])
