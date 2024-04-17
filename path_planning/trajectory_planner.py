import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory, Map

import numpy as np

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
        self.s = None
        self.t = None

    def map_cb(self, msg):
        #height x width -> 1300 x 1730 = 2,249,000
        #msg.data len -> 2,249,000
        #msg.data vals -> -1, 0, or 100 (0 means free space)
        self.get_logger().info("starting")
        self.occ_map = Map(msg)

    def pose_cb(self, pose):
        """
        New initial pose (PoseWithCovarianceStamped)
        """
        self.get_logger().info("Pose")
        self.s = pose.pose.pose.position
        if self.t is not None: 
            self.plan_path(self.s, self.t)

    def goal_cb(self, msg):
        """
        New goal pose (PoseStamped)
        """
        self.get_logger().info("Goal")
        self.t = msg.pose.position
        if self.s is not None: 
            self.plan_path(self.s, self.t)
        # p = msg.pose.position
        # self.get_logger().info(f'old x,y {(p.x,p.y)}') #u, v

        # u,v = self.occ_map.xy_to_pixel(p.x, p.y)
        # self.get_logger().info(f'u,v {(u,v)}') #u, v
        # x,y = self.occ_map.pixel_to_xy(u , v)
        # self.get_logger().info(f'new x,y {(x,y)}') #u, v

    def plan_path(self, start_point, end_point):
        """
        start_point: Ros2 Point
        end_point: Ros2 Point
        """
        self.trajectory.clear()
        
        s = (self.s.x, self.s.y)
        t = (self.t.x, self.t.y)

        #path = self.occ_map.bfs(s, t) #path start -> goal in tuples of x,y point nodes (float, float)
        path = self.occ_map.rrt(s, t)
        self.get_logger().info(str(path))
        for p in path:
            self.trajectory.addPoint(p)

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
