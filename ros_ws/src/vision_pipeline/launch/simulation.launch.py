"""Avvia la simulazione Gazebo del TIAGo Pro (mondo, SLAM, navigazione),
punto di partenza per raccolta dataset, detection e pianificazione.

Uso:
  ros2 launch vision_pipeline simulation.launch.py
  ros2 launch vision_pipeline simulation.launch.py world_name:=altro_mondo slam:=False
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
    # Esplicito (non lasciato al default di tiago_pro_gazebo.launch.py, che
    # dipende da launch_pal e non è garantito): move_group deve essere su
    # perché pose_optimizer possa pianificare/eseguire.
    moveit_arg = DeclareLaunchArgument('moveit', default_value='True')

    tiago_pro_gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('tiago_pro_gazebo'), 'launch', 'tiago_pro_gazebo.launch.py',
        ])),
        launch_arguments={
            'world_name': LaunchConfiguration('world_name'),
            'is_public_sim': LaunchConfiguration('is_public_sim'),
            'slam': LaunchConfiguration('slam'),
            'navigation': LaunchConfiguration('navigation'),
            'moveit': LaunchConfiguration('moveit'),
        }.items(),
    )

    return LaunchDescription([
        world_name_arg,
        is_public_sim_arg,
        slam_arg,
        navigation_arg,
        moveit_arg,
        tiago_pro_gazebo_launch,
    ])
