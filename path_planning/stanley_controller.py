import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

from tf_transformations import euler_from_quaternion
import numpy as np

from localization.motion_model import MotionModel

from .utils import LineTrajectory
from yasmin_ros.yasmin_node import YasminNode

class StanleyController(YasminNode):

    def __init__(self):
        super().__init__()
        self.declare_parameter('odom_topic', "/pf/pose/odom")
        self.declare_parameter('real_odom_topic', "/odom")
        self.declare_parameter('drive_topic', "/vesc/input/navigation")

        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        real_odom_topic = self.get_parameter("real_odom_topic").get_parameter_value().string_value   
        drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value  
        self.get_logger().info(f"Controller drive topic: {drive_topic}")
        self.get_logger().info(f"Controller odom topic: {real_odom_topic}")      
        
        self.trajectory = LineTrajectory("/followed_trajectory")
        self.goal = None
        self.follow_lane = None
        self._succeed = None

        # Subscribers and publishers
        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)

        self.odom_sub = self.create_subscription(Odometry, real_odom_topic,
                                                 self.odom_callback,
                                                 3)
        
        self.pose_sub = self.create_subscription(Odometry, 
                                                odom_topic, 
                                                self.pose_callback, 
                                                3)

        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               drive_topic,
                                               2)

        self.transform = lambda theta: np.array([ [np.cos(theta), -np.sin(theta), 0],
                                                [np.sin(theta), np.cos(theta), 0],
                                                [0, 0, 1]
                                                ])

        # Motion model for pose interpolation     
        self.motion_model = MotionModel(self)

        # Fixed rate publishing
        controller_rate = 50.0 # Hz
        self.controller_timer = self.create_timer(1.0/controller_rate, self.issue_control)

        # Whether to use a local motion model to interpolate between localization updates
        self.interpolation = False

        # This link better explains some of the constants
        # https://www.mathworks.com/help/driving/ref/lateralcontrollerstanley.html
        self.k = 0.7            # Convergence time constant
        self.k_soft = 1.0       # Low speed compensation
        self.k_yaw = 0.0        # Turn dampening
        self.k_steer = -0.5      # Time delay compensation
        self.k_ag = 1.0         # Curvature offset
        self.k_curv = 0.15       # Future curvature
        
        self.v = 1.0
        self.MAX_TURN = 0.3    # Turning in the saturated region of the phase diagram
        self.OFFSET = -0.03    # Constant offset because our robot sucks ass

        self.prev_theta_track = 0.0
        self.prev_theta = 0.0
        self.prev_delta = 0.0
        self.prev_t = self.get_clock().now().nanoseconds/1e9
        self.prev_odom_t = self.get_clock().now().nanoseconds/1e9
        self.current_pose = None
        self.prev_estimate = None

        # ### FOR TESTING ONLY
        # self.trajectory = LineTrajectory(self, "/loaded_trajectory")
        # self.trajectory.load("/root/racecar_ws/src/path_planning/example_trajectories/right-lane.traj")
        # self.trajectory.updatePoints(self.trajectory.points)

    @property
    def success(self): return self._succeed

    def reset_success(self): 
        self._succeed = None
        self.index = 0

    def issue_control(self):
        """
        Tells the robot what to do given all current information
        """
        if len(self.trajectory.points) == 0 or self.current_pose is None:
            drive_cmd = AckermannDriveStamped()
            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_cmd)
            # self.get_logger().info("Not initialized")
            return

        x, y, theta = self.current_pose

        # Previous and next points
        prev_i = self.find_next_index(self.current_pose)
        R = self.transform(theta)
        distance_to_goal = self.distance(np.array([0,0,0]), np.matmul(self.goal-self.current_pose, R)) 

        # Stop if at goal
        # if prev_i + 1 == len(self.trajectory.points):
        #     drive_cmd = AckermannDriveStamped()
        #     drive_cmd.drive.speed = 0.0
        #     drive_cmd.drive.steering_angle = 0.0
        #     self.drive_pub.publish(drive_cmd)
        #     self.get_logger().info("At goal")
        #     return
        if (self.follow_lane and distance_to_goal < 3.0) or self._succeed:
            self.get_logger().info("Within radius...")
            drive_cmd = AckermannDriveStamped()
            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_cmd)
            # self.last_points = self.points
            # self.last_point = None
            self._succeed = True
        elif distance_to_goal < 0.5 or self._succeed:
            self.get_logger().info("Reached goal...")
            drive_cmd = AckermannDriveStamped()
            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_cmd)
            # self.last_points = self.points
            # self.last_point = None
            self._succeed = True
        else:
            prev_p = self.trajectory.points[prev_i]
            next_p = self.trajectory.points[prev_i + 1]

            # Cross track error (distance from current pose to the path)
            e_num = (next_p[0] - prev_p[0]) * (prev_p[1] - y) - (prev_p[0] - x) * (next_p[1] - prev_p[1])
            e_denom = np.linalg.norm(np.array([prev_p[0] - next_p[0], prev_p[1] - next_p[1]]))
            cross_track = e_num / e_denom

            # Cross track steering correction
            theta_xc = np.arctan2(self.k * cross_track, self.k_soft + self.v)

            # Direction of the trajectory (where we should be pointing)
            theta_track = next_p[2]

            # Heading error
            psi = theta_track - theta
            psi = (psi + np.pi) % (2 * np.pi) - np.pi

            # Yaw rates
            now = self.get_clock().now().nanoseconds/1e9
            r_traj = (theta_track - self.prev_theta_track) / (now - self.prev_t)
            r_meas = (theta - self.prev_theta) / (now - self.prev_t)

            # Future curvature
            eps_f = (self.trajectory.points[prev_i + 3][2] - self.prev_theta_track) if prev_i + 3 <= len(self.trajectory.points) else 0

            # Yaw setpoint
            psi_ss = self.k_ag * self.v * r_traj

            # Updated steering
            delta = psi - psi_ss
            delta += theta_xc
            delta += self.k_yaw * (r_meas - r_traj)
            delta += self.k_curv * eps_f
            delta += self.k_steer * (delta - self.prev_delta)   # Assumes instantaneous steering

            # Store rate values
            self.prev_theta = theta
            self.prev_theta_track = theta_track
            self.prev_t = now

            # Publish the drive command
            drive_cmd = AckermannDriveStamped()
            if eps_f >= self.MAX_TURN * (2/3):
                drive_cmd.drive.speed = self.v * (2/3)
            else:
                drive_cmd.drive.speed = self.v
            drive_cmd.drive.steering_angle = np.clip(delta + self.OFFSET, -self.MAX_TURN, self.MAX_TURN)
            self.drive_pub.publish(drive_cmd)

    def odom_callback(self, msg):
        """
        Predicts where the robot actually is between pose updates
        """
        if not self.interpolation:
            return

        now = self.get_clock().now().nanoseconds/1e9

        dt = (now - self.prev_odom_t)

        # Calculate our change in pose
        v = [odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z]
        dx, dy, dtheta = v[0] * dt, v[1] * dt, v[2] * dt
        delta_x = [dx, dy, dtheta]

        # Move the pose based on odom
        self.current_pose = self.motion_model.evaluate_noiseless(self.current_pose, delta_x)

        self.prev_odom_t = now

    def trajectory_callback(self, msg):
        """
        Takes in a pose array trajectory and converts it to our shitty representation of a trajectory 
        """
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

    def pose_callback(self, odometry_msg):
        """
        Takes in estimated pose from our localization and stores it in the node
        """
        # Unpack the odom message
        x = odometry_msg.pose.pose.position.x
        y = odometry_msg.pose.pose.position.y
        orientation = euler_from_quaternion((
            odometry_msg.pose.pose.orientation.x,
            odometry_msg.pose.pose.orientation.y,
            odometry_msg.pose.pose.orientation.z,
            odometry_msg.pose.pose.orientation.w))
        theta = orientation[2]      

        # Update the current pose whenever the localization actually updates it
        self.current_pose = np.array([x, y, theta])

    def find_next_index(self, position):
        """
        Finds the index of the next point on the line and returns it
        """        
        # Initialize the minimum distance and nearest point
        min_distance = float('inf')
        nearest_index = 0
        
        # Iterate through each point in the trajectory
        for i, point in enumerate(self.trajectory.points):
            point_x, point_y, point_theta = point
            
            # Calculate the vector from the car's position to the point
            dx = point_x - position[0]
            dy = point_y - position[1]
            
            # Calculate the angle from the car's orientation to the point
            angle_to_point = np.arctan2(dy, dx) - position[2]
            
            # Normalize the angle to be between 0 and 2pi
            angle_to_point = (angle_to_point + np.pi) % (2 * np.pi)
            
            # Calculate the euclidean distance from the car's position to the point
            dist = np.linalg.norm(np.array([dx, dy]))
            
            # Update the minimum distance and nearest point
            if dist < min_distance:
                min_distance = dist
                nearest_index = i
        
        return nearest_index

def main(args=None):
    rclpy.init(args=args)
    follower = StanleyController()
    rclpy.spin(follower)
    rclpy.shutdown()

# colcon build --symlink-install --packages-select path_planning && source install/setup.bash

# ros2 launch path_planning sim_stanley.launch.xml
# ros2 launch racecar_simulator simulate.launch.xml