import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory, Map

import numpy as np
from tf_transformations import euler_from_quaternion

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        # self.R_z = lambda theta: np.array([ [np.cos(theta), -np.sin(theta), 0],
        #                                      [np.sin(theta), np.cos(theta), 0],
        #                                      [0, 0, 1]
        #                                     ])
        
        self.occ_map = None

    def map_cb(self, msg):
        #height x width -> 1300 x 1730 = 2,249,000
        #msg.data len -> 2,249,000
        #msg.data vals -> -1, 0, or 100 (0 means free space)

        self.get_logger().info("starting")

        self.occ_map = Map(msg)



        # d = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # q = np.array([[0], [0], [0]]) #3x1 x,y,z

        # p = msg.info.origin.position
        # q = q - np.array([[p.x], [p.y], [p.z]])

        # orientation = msg.info.origin.orientation

        # _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        # q = ((self.R_z(-yaw) @ q) / msg.info.resolution)

        # self.get_logger().info(f'q {q}') #u, v

        # u, v = q[:2, :]

        # self.get_logger().info(f'v2 {d[round(v[0])][round(u[0])]}') #u, v

        return

    def pose_cb(self, pose):
        """
        New initial pose (PoseWithCovarianceStamped)
        """
        return

    def goal_cb(self, msg):
        """
        New goal pose (PoseStamped)
        """

        # u, v = 512, 963

        # x,y = self.occ_map.pixel_to_xy(u, v)
        # n_u,n_v = self.occ_map.xy_to_pixel(x, y)

        # self.get_logger().info(f'new u,v {(n_u,n_v)}') #u, v

        # assert (u, v) == (n_u, n_v)



        p = msg.pose.position
        self.get_logger().info(f'old x,y {(p.x,p.y)}') #u, v

        u,v = self.occ_map.xy_to_pixel(p.x, p.y)
        self.get_logger().info(f'u,v {(u,v)}') #u, v
        x,y = self.occ_map.pixel_to_xy(u , v)
        self.get_logger().info(f'new x,y {(x,y)}') #u, v

        # assert (x, y) == (p.x, p.y)
        return

    def plan_path(self, start_point, end_point, map):
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
