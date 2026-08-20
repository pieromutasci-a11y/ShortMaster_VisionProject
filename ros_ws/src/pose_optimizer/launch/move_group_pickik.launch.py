# Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#FM: è la copia modificata del launch file originale di PAL (/opt/ros/humble/share/tiago_pro_moveit_config/launch/move_group.launch.py), utile
#per fare modifiche senza toccare il pacchetto di sistema.

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_pal.arg_utils import read_launch_argument
from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder
from launch_pal.arg_utils import LaunchArgumentsBase
from launch_pal.robot_arguments import CommonArgs
from tiago_pro_description.launch_arguments import TiagoProArgs
from tiago_pro_description.tiago_pro_launch_utils import get_tiago_pro_hw_suffix
from dataclasses import dataclass
from ament_index_python.packages import get_package_share_directory

#dichiarazione degli argomenti di launch, che possono essere passati da riga di comando o da un altro launch file

@dataclass(frozen=True)
class LaunchArguments(LaunchArgumentsBase):
    arm_type_right: DeclareLaunchArgument = TiagoProArgs.arm_type_right
    arm_type_left: DeclareLaunchArgument = TiagoProArgs.arm_type_left
    end_effector_right: DeclareLaunchArgument = TiagoProArgs.end_effector_right
    end_effector_left: DeclareLaunchArgument = TiagoProArgs.end_effector_left
    ft_sensor_right: DeclareLaunchArgument = TiagoProArgs.ft_sensor_right
    ft_sensor_left: DeclareLaunchArgument = TiagoProArgs.ft_sensor_left
    base_type: DeclareLaunchArgument = TiagoProArgs.base_type
    has_teleop_arms: DeclareLaunchArgument = TiagoProArgs.has_teleop_arms
    use_sim_time: DeclareLaunchArgument = CommonArgs.use_sim_time
    use_sensor_manager_arg: DeclareLaunchArgument = CommonArgs.use_sensor_manager
    ft_sensor_teleop_left: DeclareLaunchArgument = TiagoProArgs.ft_sensor_teleop_left
    ft_sensor_teleop_right: DeclareLaunchArgument = TiagoProArgs.ft_sensor_teleop_right

#aggiunge due nodi al launch file: uno per il move_group (costruito da start_move_group)e uno per aggiungere il piano al mondo di MoveIt
def declare_actions(launch_description: LaunchDescription, launch_args: LaunchArguments):

    launch_description.add_action(OpaqueFunction(function=start_move_group))

    add_plane_node = Node(
        package='tiago_pro_moveit_config',
        executable='add_ground_node.py',
        output='screen',
    )
    launch_description.add_action(add_plane_node)
    return

#dove effettivamente ci sono le modifiche risp all'originale.
def start_move_group(context, *args, **kwargs):

    arm_type_right = read_launch_argument('arm_type_right', context)
    arm_type_left = read_launch_argument('arm_type_left', context)
    end_effector_right = read_launch_argument('end_effector_right', context)
    end_effector_left = read_launch_argument('end_effector_left', context)
    use_sensor_manager = read_launch_argument('use_sensor_manager', context)

    hw_suffix = get_tiago_pro_hw_suffix(
        arm_right=arm_type_right,
        arm_left=arm_type_left,
        end_effector_right=end_effector_right,
        end_effector_left=end_effector_left)

    
    #viene dichiarato il path del file SRDF, che contiene le informazioni di configurazione del robot per MoveIt. Viene generato da 
    #un file xacro (tiago_pro.srdf.xacro) che permette di generare il file SRDF in base agli argomenti passati a launch (tipo di braccio, tipo di end effector, ecc)
    srdf_file_path = Path(
        os.path.join(
            get_package_share_directory("tiago_pro_moveit_config"),
            "config", "srdf",
            "tiago_pro.srdf.xacro",
        )
    )

    srdf_input_args = {
        'arm_type_right': read_launch_argument('arm_type_right', context),
        'arm_type_left': read_launch_argument('arm_type_left', context),
        'end_effector_right': read_launch_argument('end_effector_right', context),
        'end_effector_left': read_launch_argument('end_effector_left', context),
        'ft_sensor_right': read_launch_argument('ft_sensor_right', context),
        'ft_sensor_left': read_launch_argument('ft_sensor_left', context),
        "base_type": read_launch_argument("base_type", context),
        'has_teleop_arms': read_launch_argument('has_teleop_arms', context),
        'ft_sensor_teleop_left': read_launch_argument('ft_sensor_teleop_left', context),
        'ft_sensor_teleop_right': read_launch_argument('ft_sensor_teleop_right', context),
    }

    # Trajectory Execution Functionality
    moveit_simple_controllers_path = (
        f'config/controllers/controllers{hw_suffix}.yaml')

    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    # The robot description is read from the topic /robot_description if the parameter is empty
    #punto chiave, robot_description e robot_description_kinematics puntano a path diversi dall'originale 
    # Il primo contiene la descrizione del robot da tiago_pro_for_moveit.urdf,che è un file URDF modificato per MoveIt perche c'erano dei link mancanti.
    #Normalmente robot_description prende info dal topic /robot_description, che viene pubblicato dal nodo robot_state_publisher, ma in questo caso viene passato direttamente il file URDF modificato per motivi d
    #di simulazione e link mancanti. Il secondo contiene la descrizione della cinematica del robot, che è stata modificata per evitare problemi di pianificazione con il plugin KDL.
    #di base viene usato il plugin KDL per la cinematica, ma in questo caso è stato modificato per usare un altro plugin (pickik) in modo tale da utilizzare ottimizzazioni per risolvere la cinematica.
    #Infatti .robot_description_kinematics punta a kinematics_override/kinematics_pickik.yaml, che contiene la configurazione del plugin pickik per la cinematica inversa.
    moveit_config = (
        MoveItConfigsBuilder('tiago_pro')
        .robot_description_semantic(file_path=srdf_file_path, mappings=srdf_input_args)
        .robot_description_kinematics(file_path='/home/user/ros_workspace/kinematics_override/kinematics_pickik.yaml')
        .robot_description(file_path='/home/user/ros_workspace/tiago_pro_for_moveit.urdf')
        .trajectory_execution(moveit_simple_controllers_path)
        .joint_limits(file_path='/opt/ros/humble/share/tiago_pro_moveit_config/config/joint_limits.yaml')
        .planning_pipelines(pipelines=['ompl', 'chomp'], default_planning_pipeline='ompl')
        .planning_scene_monitor(planning_scene_monitor_parameters)
        .pilz_cartesian_limits(file_path='/opt/ros/humble/share/tiago_pro_moveit_config/config/pilz_cartesian_limits.yaml')
    )

    if use_sensor_manager == "True":
        # moveit_sensors path
        moveit_sensors_path = 'config/sensors_3d.yaml'
        moveit_config.sensors_3d(moveit_sensors_path)

    moveit_config.to_moveit_configs()

    move_group_configuration = {
        'use_sim_time': LaunchConfiguration('use_sim_time'),
        'publish_robot_description_semantic': True,
        'robot_description_timeout': 60.0,
        'capabilities': "move_group/ExecuteTaskSolutionCapability"
    }

    move_group_params = [
        moveit_config.to_dict(),
        move_group_configuration,
    ]

    # Start the actual move_group node/action server
    run_move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        emulate_tty=True,
        parameters=move_group_params,
    )

    return [run_move_group_node]


def generate_launch_description():

    # Create the launch description and populate
    ld = LaunchDescription()
    launch_arguments = LaunchArguments()

    launch_arguments.add_to_launch_description(ld)

    declare_actions(ld, launch_arguments)

    return ld
