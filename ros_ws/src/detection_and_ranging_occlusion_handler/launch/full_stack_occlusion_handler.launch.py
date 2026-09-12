"""Avvia l'intero stack di detection CON gestione occlusioni in un colpo
solo: simulazione Gazebo del TIAGo Pro + nodi di detection/ranging e center
computation (fit di circonferenza sui 3 punti di superficie) + RViz con la
vista dedicata.

Uso:
  ros2 launch detection_and_ranging_occlusion_handler full_stack_occlusion_handler.launch.py
  # se la simulazione e' gia' avviata altrove:
  ros2 launch detection_and_ranging_occlusion_handler full_stack_occlusion_handler.launch.py launch_simulation:=false
  # senza RViz:
  ros2 launch detection_and_ranging_occlusion_handler full_stack_occlusion_handler.launch.py rviz:=false
  # modalita' light (niente gestione occlusioni, centro bbox + depth):
  ros2 launch detection_and_ranging_occlusion_handler full_stack_occlusion_handler.launch.py use_occlusion_handling:=false
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
            FindPackageShare('detection_and_ranging_occlusion_handler'),
            'rviz', 'complete_visualization_occlusion_handler.rviz',
        ]),
        description='File di configurazione RViz da usare.',
    )
    use_occlusion_handling_arg = DeclareLaunchArgument(
        'use_occlusion_handling',
        default_value='True',
        description=(
            'Modalita\' del nodo di detection: true (robust, default) esclude i '
            'pixel condivisi tra oggetti sovrapposti prima di stimare la depth e '
            'campiona i 3 punti di superficie usati dal fit di circonferenza; '
            'false (light) usa solo il centro bounding box, nessuna gestione '
            'occlusioni -- in questo caso center_computation_occlusion_handler '
            'non ricevera\' nulla su yolo/tracked_objects_points e non '
            'pubblichera\' nessun centro.'
        ),
    )

    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('vision_pipeline'), 'launch', 'simulation.launch.py',
        ])),
        condition=IfCondition(LaunchConfiguration('launch_simulation')),
    )

    detection_node = Node(
        package='detection_and_ranging_occlusion_handler',
        executable='rt_object_detection_occlusion_handler',
        name='rt_object_detection_occlusion_handler_node',
        output='screen',
        parameters=[{
            'use_occlusion_handling': LaunchConfiguration('use_occlusion_handling'),
        }],
    )

    center_computation_node = Node(
        package='detection_and_ranging_occlusion_handler',
        executable='center_computation_occlusion_handler',
        name='center_computation_occlusion_handler_node',
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
        use_occlusion_handling_arg,
        simulation_launch,
        detection_node,
        center_computation_node,
        rviz_node,
    ])
