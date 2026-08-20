import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint,
    CollisionObject, PlanningScene
)
from geometry_msgs.msg import PoseStamped, Pose
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
import numpy as np


class MoveGroupClient(Node):
    def __init__(self):
        super().__init__('move_group_client')
        self._client = ActionClient(self, MoveGroup, '/move_action')
        self._joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self._scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self._marker_pub = self.create_publisher(Marker, '/target_marker', 10)




    def add_table_obstacle(self, frame_id="base_footprint",
                            position=(0.8, 0.0, 0.375),
                            dimensions=(0.8, 0.8, 0.75)):
        obj = CollisionObject()
        obj.header.frame_id = frame_id
        obj.id = "table"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(dimensions)

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = position
        pose.orientation.w = 1.0

        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.world.collision_objects.append(obj)
        scene.is_diff = True

        # Pubblica più volte per garantire la ricezione
        for _ in range(5):
            self._scene_pub.publish(scene)
            rclpy.spin_once(self, timeout_sec=0.3)

    def publish_target_marker(self, pose_stamped: PoseStamped):
        m = Marker()
        m.header.frame_id = pose_stamped.header.frame_id
        m.ns = "target"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose = pose_stamped.pose
        m.scale.x = m.scale.y = m.scale.z = 0.04
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0

        for _ in range(10):
            m.header.stamp = self.get_clock().now().to_msg()
            self._marker_pub.publish(m)
            rclpy.spin_once(self, timeout_sec=0.3)

    def send_goal(self, target_pose: PoseStamped, group_name="arm_left", link_name="arm_left_tool_link"):
        self._client.wait_for_server()

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = group_name
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0

        pc = PositionConstraint()
        pc.header.frame_id = target_pose.header.frame_id
        pc.link_name = link_name
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.01]
        pc.constraint_region.primitives.append(primitive)
        pc.constraint_region.primitive_poses.append(target_pose.pose)
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = target_pose.header.frame_id
        oc.link_name = link_name
        oc.orientation = target_pose.pose.orientation
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 3.14
        oc.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)
        req.goal_constraints.append(constraints)

        goal_msg.request = req
        goal_msg.planning_options.plan_only = True

        future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal rifiutato")
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result()

    def publish_joint_state(self, joint_names, positions):
        msg = JointState()
        msg.name = joint_names
        msg.position = positions

        for _ in range(10):
            msg.header.stamp = self.get_clock().now().to_msg()
            self._joint_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.5)


def orientamento_tool_per_pinza_orizzontale(yaw=0.0):
        """
        Calcola l'orientamento di arm_left_tool_link necessario affinche'
        gripper_left_grasping_link abbia:
          - asse di approccio (z locale) verticale, verso il basso
          - piano delle dita (xy locale) parallelo al tavolo
        yaw: rotazione libera attorno all'asse verticale (rad)
        """
        # Trasformazione fissa nota dall'URDF: tool_link -> grasping_link
        # (gripper_left_base_joint e' identita', poi rpy="0 -1.57 0")
        R_tool_to_grasp = R.from_euler('y', -1.57)

        # Orientamento desiderato in world per grasping_link:
        # asse z locale -> verso il basso (-z world), piano xy -> orizzontale
        R_world_grasp = R.from_euler('z', yaw) * R.from_euler('x', np.pi)

        # Risolvi per l'orientamento di tool_link
        R_world_tool = R_world_grasp * R_tool_to_grasp.inv()

        return R_world_tool.as_quat()  # [x, y, z, w]



def main():
    rclpy.init()
    node = MoveGroupClient()

    # --- Ostacolo: tavolo ---
    #print("Aggiungo ostacolo (tavolo)...")
    #node.add_table_obstacle(
    #    position=(0.4, 0.0, 0.375),
    #    dimensions=(0.2, 0.4, 0.6)
    #)

    # --- Target ---
    target = PoseStamped()
    target.header.frame_id = "base_footprint"
    target.pose.position.x = 0.8
    target.pose.position.y = 0.0
    target.pose.position.z = 0.6
    quat = orientamento_tool_per_pinza_orizzontale(yaw=0.0)
    target.pose.orientation.x = quat[0]
    target.pose.orientation.y = quat[1]
    target.pose.orientation.z = quat[2]
    target.pose.orientation.w = quat[3]

    print("Pubblico marker del target...")
    node.publish_target_marker(target)

    result = node.send_goal(target)

    if result and result.result.error_code.val == 1:
        traj = result.result.planned_trajectory.joint_trajectory
        last = traj.points[-1]
        print("Configurazione finale (rad):")
        for name, pos in zip(traj.joint_names, last.positions):
            print(f"  {name}: {pos:.4f}")

        print("\nPubblico su /joint_states per la visualizzazione...")
        node.publish_joint_state(
            list(traj.joint_names) + ['torso_lift_joint'],
            list(last.positions) + [0.0]
        )
        print("Fatto. Controlla RViz.")
    else:
        print("Pianificazione fallita, error code:", result.result.error_code.val if result else "nessuna risposta")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
