import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, Pose
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32

from tf_transformations import euler_from_quaternion
import tf2_ros

import numpy as np

from path_planning.utils import LineTrajectory
import math
import time

"""
TODO:
np argmin optimize for time
bug if goal pose is facing forward
xdot sometimes gives answer if initial trajectory is also ahead of goal

error plots
Make sure you mention your method for tuning the controller to closely track trajectories. 
(Hint: include error plots from rqt_plot)

y distance from closest segment
slope
lookahead
speed

ros2 launch path_planning sim_yeet.launch.xml
"""

class PID(Node):
    """ 
    Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__()
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        # self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        # self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.odom_topic = "/pf/pose/odom"
        self.drive_topic = "/vesc/input/navigation"
        # self.drive_topic = '/drive'

        self.follow_lane = None

        self.points = None
        self.current_pose = None
        self.intersections = None
        self.turning_markers = []
        self.goal = None

        self.last_points = None
        self.distance_check = False
        self.last_time = None
        self.last_error = None
        self.last_integral = None
        self.integral_count = 0
        self.last_point = None

        self._succeed = None
    
        
        self.MAX_TURN = 0.34

        self.trajectory = LineTrajectory(Node("followed_trajectory"))

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        
        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_callback, 1)

        self.transform = lambda theta: np.array([ [np.cos(theta), -np.sin(theta), 0],
                                             [np.sin(theta), np.cos(theta), 0],
                                             [0, 0, 1]
                                            ])
        
        self.get_logger().info('initialized')
        
        # self.relative_point_pub = self.create_publisher(MarkerArray,'/relative',1)
        # self.circle = self.create_publisher(MarkerArray, '/circle_marker', 1)
        self.intersection = self.create_publisher(Marker,'/intersection',1)
        # self.closest = self.create_publisher(Marker,'/closest',1)
        # self.segments = self.create_publisher(MarkerArray, '/segments', 1)
        # self.turning_points = self.create_publisher(MarkerArray, '/turning_points', 1)
        self.previous_angles = []

        self.previous_errors = []

        self.all_controls = []

        self.index = 0
        
        self.traveled_points = set()

    @property
    def success(self): return self._succeed

    def reset_success(self): 
        self._succeed = None
        self.index = 0

    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        
    def pose_callback(self, odometry_msg):
        """
        publishes pure pursuit upon odom callback
        """
        x = odometry_msg.pose.pose.position.x
        y = odometry_msg.pose.pose.position.y

        orientation = euler_from_quaternion((
        odometry_msg.pose.pose.orientation.x,
        odometry_msg.pose.pose.orientation.y,
        odometry_msg.pose.pose.orientation.z,
        odometry_msg.pose.pose.orientation.w))

        theta = orientation[2]
        R = self.transform(theta)
        self.current_pose = np.array([x,y,theta])  #car's coordinates in global frame


        drive_cmd = AckermannDriveStamped()
        if self.last_points is None or np.any(self.points != self.last_points):

            if self.points is not None:
                pt = self.points[self.index]
                closest_point = np.matmul(pt-self.current_pose,R)
                dist = self.distance(np.array([0,0,0]),closest_point)
                distance_to_goal = self.distance(np.array([0,0,0]), np.matmul(self.goal-self.current_pose, R)) 

                global_intersect = np.matmul(closest_point, np.linalg.inv(R)) + self.current_pose
                self.intersection.publish(self.to_marker(global_intersect, 0, [0.0, 1.0, 0.0], 0.5))

                if (self.follow_lane and distance_to_goal < 3.0) or self._succeed:
                    self.get_logger().info("Within radius...")
                    drive_cmd.drive.speed = 0.0
                    drive_cmd.drive.steering_angle = 0.0
                    self.drive_pub.publish(drive_cmd)
                    self.last_points = self.points
                    self.last_point = None
                    self._succeed = True
                elif distance_to_goal < 0.5 or self._succeed:
                    self.get_logger().info("Reached goal...")
                    drive_cmd.drive.speed = 0.0
                    drive_cmd.drive.steering_angle = 0.0
                    self.drive_pub.publish(drive_cmd)
                    self.last_points = self.points
                    self.last_point = None
                    self._succeed = True
                else:
                    if self.last_point is None:
                        self.last_point = closest_point
                        
                    if dist < 0.2 or closest_point[0]*self.last_point[0] < 0: #probs different irl
                        if self.index != len(self.points)-1:
                            self.index+=1
                            new_pt = self.points[self.index]
                            self.last_point = np.matmul(new_pt-self.current_pose,R)

                    speed = 1.5

                    orientation_error = np.arctan2( np.sin(pt[2]-theta), np.cos(pt[2]-theta) )

                    theta_xc = np.arctan2(closest_point[1], speed)
                    # self.get_logger().info(f'track: {closest_point[2]}, theta: {theta}')
                    # self.get_logger().info(f'heading: {orientation_error} cross track: {theta_xc}')

                    error = 0.8*orientation_error + 1.2*theta_xc

                    drive_cmd = AckermannDriveStamped()

                    #what worked for speed 3.0 - Kp 0.7, Kd = Kp/6.0, Ki = 0 (didnt test), turning angle += -0.04


                    ####### BOT TUNING DO NOT CHANGE #########
                    Kp = .25# previous Kp is 0.635
                    Kd = Kp / 6.0
                    Ki = 0#-Kp / 2.0
                    # # self.previous_errors.append(error)

                    ############# SIM PARAMS ##########
                    # Kp = .25
                    # Kd = Kp/6.0
                    # Ki = 0


                    P = Kp* ( error )
                    if self.last_time is not None:
                        dt = self.get_clock().now().nanoseconds/1e9 - self.last_time
                        I = self.last_integral + Ki*error*dt
                        self.integral_count += 1
                        D = Kd * (error-self.last_error)/dt
                    else:
                        I = 0
                        D = 0

                    control = P + I + D
                    # self.get_logger().info(f'{error}')
                    # self.get_logger().info(f'P: {round(P,3)} I: {round(I,3)} D: {round(D,3)}')
                    self.previous_errors.append(error)
                    self.all_controls.append((P,I,D))

                    turning_angle = control
                    # turning_angle += -0.045
                    turning_angle += -0.03

                    # self.get_logger().info(f'{turning_angle}')
                                
                    if abs(turning_angle) > self.MAX_TURN:
                        turning_angle = self.MAX_TURN if turning_angle > 0 else -self.MAX_TURN


                    # if closest_point[0] < 0:
                    #     speed = -2.0
                    #     turning_angle = turning_angle

                    drive_cmd.drive.speed = speed
                    drive_cmd.drive.steering_angle = turning_angle

                    self.drive_pub.publish(drive_cmd)
                    self.last_time = self.get_clock().now().nanoseconds/1e9
                    self.last_error = error
                    self.last_integral = I
                    if self.integral_count == 15:
                        self.last_integral = 0
                        self.integral_count = 0


                    self.drive_pub.publish(drive_cmd)
        else:
            # self.get_logger().info('what')
            pass
        
    def trajectory_callback(self, msg):
        """
        msg: PoseArray
        geometry_msgs/msg/Pose[] poses
            geometry_msgs/msg/Point position
            geometry_msgs/msg/Quaternion orientation
        """
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        # self.points = np.array([(i.position.x,i.position.y,position) for i in msg.poses]) #no theta 
        # self.get_intersections()
        # self.goal = self.points[-1]

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.points = np.array(self.trajectory.points)
        if self.points.shape[-1] != 3:
            self.get_logger().info('hi')
            # self.points = self.reset_distances()
            # self.update
            self.trajectory.updatePoints(self.points)
            self.points = np.array(self.trajectory.points)
        # self.index = 0
        # self.goal = self.points[-1]
        # self._succeed = None
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True

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
    follower = PID()
    try:
        rclpy.spin(follower)
    except KeyboardInterrupt:
        drive_cmd = AckermannDriveStamped()
        drive_cmd.drive.speed = 0.0
        drive_cmd.drive.steering_angle = 0.0
        follower.drive_pub.publish(drive_cmd)
    rclpy.shutdown()



    