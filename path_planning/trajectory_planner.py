import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, PointStamped
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory, Map
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
# import cProfile

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")
        self.declare_parameter('clicked_point_topic',"default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.clicked_point_topic = self.get_parameter('clicked_point_topic').get_parameter_value().string_value

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

        self.point_sub = self.create_subscription(
            PointStamped,
            self.clicked_point_topic,
            self.update_path,
            10
        )

        self.tree_pub = self.create_publisher(MarkerArray, '/tree', 10)           ## REMOVE THIS

        #Line Trajectory class in utils.py
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        
        self.occ_map = None
        self.points = None
        self.s = None
        self.t = None

    def map_cb(self, msg):
        #height x width -> 1300 x 1730 = 2,249,000
        #msg.data len -> 2,249,000
        #msg.data vals -> -1, 0, or 100 (0 means free space)
        self.get_logger().info("starting")
        self.occ_map = Map(msg)

    def update_path(self,msg):
        if self.s is None:
            self.get_logger().info("Set initial pose first!")
            return

        pose = [msg.point.x, msg.point.y, None]

        dist = lambda p1,p2: ( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )**(1/2)

        self.points.append(pose)
        self.get_logger().info("Added point")

        # if len(self.points) > 2: # if points is not just (start_pose, clicked_point)
        #     self.points.append(pose)

    def pose_cb(self, pose):
        """
        New initial pose (PoseWithCovarianceStamped)
        """
        self.get_logger().info("Pose")
        orientation = euler_from_quaternion((
        pose.pose.pose.orientation.x,
        pose.pose.pose.orientation.y,
        pose.pose.pose.orientation.z,
        pose.pose.pose.orientation.w))
        self.s = [pose.pose.pose.position.x,pose.pose.pose.position.y,orientation[2]]
        # self.s_theta = orientation[2]

        # if self.points is None:
        #     self.points = [self.s]
        # else:
        #     self.points[0] = self.s
        self.points = [self.s]

        # if self.t is not None: 
        #     self.plan_path()
    
    def goal_cb(self, msg):
        """
        New goal pose (PoseStamped)
        """
        self.get_logger().info("Goal")
        # self.t = msg.pose.position
        orientation = euler_from_quaternion((
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w))
        # self.t_theta = orientation[2]
        self.t = [msg.pose.position.x,msg.pose.position.y,orientation[2]]
        total_path = []
        if self.s is not None:
            self.points.append(self.t)
            self.get_logger().info(f'Finding trajectory for {len(self.points)} points...')
            count = 2

            for idx in range(len(self.points)):
                s = self.points[idx]
                t = self.points[(idx + 1)%len(self.points)]
                if s[2] is None: #clicked point, so there will always be a point after
                    orientation = np.arctan2(t[1]-s[1], t[0]-s[0])
                    s[2] = orientation
                
                if t[2] is None:
                    next_ = self.points[(idx + 2)%len(self.points)]
                    orientation = np.arctan2(next_[1]-t[1], next_[0]-t[1])
                    t[2] = orientation
            
                path = self.plan_path(s,t)
                if path is not None:
                    total_path.extend(path)
                    self.get_logger().info(f'Found trajectory between first {count} points')
                    count+=1
                else:
                    return
                
            self.publish_path(total_path)
            # self.plan_path()

    def plan_path(self,start_point,end_point):
        """
        start_point s: Ros2 Point
        end_point t: Ros2 Point
        """
        self.trajectory.clear()

        ALG = "astar"
        try:
            search_dict = {"bfs": self.occ_map.bfs, "rrt": self.occ_map.rrt, "rrt_star": self.occ_map.rrt_star, "astar": self.occ_map.astar}
        except AttributeError:
            self.get_logger().info("Restart map!")
            return
        
        if ALG in ['bfs', 'astar', 'rrt']:
            s = (start_point[0],start_point[1])
            t = (end_point[0],end_point[1])
        else:
            s = start_point
            t = end_point

        nodes = None

        # profiler = cProfile.Profile()
        # profiler.enable()

        #path, nodes = self.occ_map.rrt_star(s, t)
        # profiler.disable()
        # profiler.print_stats(sort='time')

        #path = self.occ_map.rrt(s, t)

        # path = self.occ_map.astar(s, t)
        #path = self.occ_map.prune_path(path)
        

        # path = self.occ_map.rrt(s, t)
        path = search_dict[ALG](s, t) #path start -> goal in tuples of x,y point nodes (float, float)
        if isinstance(path, tuple):
            if path[0] is None:
                self.get_logger().info("no path")
                return
            else:
                path = path[0]
            # nodes = path[1]
            # path = path[0]

        if len(path) == 0 or path is None: 
            self.get_logger().info("No path found!")
            return
        
        return path

        # path = self.occ_map.prune_path(path)

    def publish_path(self,path):

        # if nodes is not None:
        #     x = []
        #     y = []
        #     # for path in paths:
        #     for point in nodes:
        #         x.append(point[0])
        #         y.append(point[1])

        #     self.get_logger().info("points generated")

        #     self.publish_marker_array(self.tree_pub, x, y)
        
        # if path is not None:
        self.trajectory.updatePoints(path)

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    def publish_marker_array(self, publisher, xa, ya, rgb=[1.0,0.0,0.0]):
        """
        converts the point array from car to global frame and publishes the markerarray
        """
        markers = []
        count = 0
        for x, y in zip(xa, ya):
            marker = self.to_marker((x, y), count, rgb=rgb)
            markers.append(marker)
            count+=1

        markerarray = MarkerArray()
        markerarray.markers = markers
        publisher.publish(markerarray)

    def to_marker(self,position,id = 1,rgb=[1.0,0.0,0.0],scale=0.25):
        marker = Marker()
        marker.header.frame_id = "/map"  # Set the frame id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "/followed_trajectory/trajectory"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.a = 1.0
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]

        return marker


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
