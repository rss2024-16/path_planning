import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

from tf_transformations import euler_from_quaternion
import numpy as np

from .utils import LineTrajectory

class StanleyController(Node):

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "/pf/pose/odom")
        self.declare_parameter('real_odom_topic', "/odom")
        self.declare_parameter('drive_topic', "/vesc/input/navigation")

        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        real_odom_topic = self.get_parameter("real_odom_topic").get_parameter_value().string_value   
        drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value  
        self.get_logger().info(f"Controller drive topic: {drive_topic}")
        self.get_logger().info(f"Controller odom topic: {real_odom_topic}")      
        
        self.trajectory = LineTrajectory("/followed_trajectory")

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

        # Motion model for pose interpolation     
        self.motion_model = MotionModel(self)

        # Fixed rate publishing
        controller_rate = 50.0 # Hz
        self.controller_timer = self.create_timer(1.0/controller_rate, self.issue_control)

        # Whether to use a local motion model to interpolate between localization updates
        self.interpolation = False

        self.k = 0.2            # Convergence time constant
        self.k_soft = 1.0       # Low speed compensation
        self.k_yaw = 0.0        # Turn dampening
        self.k_steer = 0.0      # Time delay compensation
        self.k_ag = 0.0         # Curvature offset
        
        self.v = 2.0
        self.MAX_TURN = 0.2     # Real max turn is 0.34, but that is too much often
        self.OFFSET = -0.04     # Constant offset because our robot sucks ass

        self.prev_theta_track = 0.0
        self.prev_theta = 0.0
        self.prev_delta = 0.0
        self.prev_t = self.get_clock().now()
        self.prev_odom_t = self.get_clock().now()
        self.current_pose = None
        self.prev_estimate = None

    def issue_control(self):
        """
        Tells the robot what to do given all current information
        """
        if len(self.trajectory.points) == 0 and self.current_pose is not None:
            drive_cmd = AckermannDriveStamped()
            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_cmd)
            return

        # Previous and next points
        prev_i = self.find_next_index(self.current_pose)

        # Stop if at goal
        if prev_i + 1 == len(self.trajectory.points):
            drive_cmd = AckermannDriveStamped()
            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_cmd)
            return

        prev_p = self.trajectory.points[prev_i]
        next_p = self.trajectory.points[prev_i + 1]

        # Cross track error (distance from current pose to the path)
        e_num = (next_p[0] - prev_p[0]) * (prev_p[1] - y) - (prev_p[0] - x) * (next_p[1] - prev_p[1])
        e_denom = np.linalg.norm(np.array([prev_p[0] - next_p[0], prev_p[1] - next_p[1]]))
        cross_track = e_num / e_denom

        # Cross track steering correction
        theta_xc = np.arctan2(self.k * cross_track, self.k_soft + self.v)

        # Direction of the trajectory (where we should be pointing)
        theta_track = prev_p[2]

        # Heading error
        psi = theta_track - theta
        psi = (psi + np.pi) % (2 * np.pi) - np.pi

        # Yaw rates
        now = self.get_clock().now()
        r_traj = (theta_track - self.prev_theta_track) / (now - self.prev_t)
        r_meas = (theta - self.prev_theta) / (now - self.prev_t)

        # Yaw setpoint
        psi_ss = self.k_ag * self.v * r_traj

        # Updated steering
        delta = psi
        delta += theta_xc
        delta += self.k_yaw * (r_meas - r_traj)
        delta += self.k_steer * (delta - self.prev_delta)

        # Store rate values
        self.prev_theta = theta
        self.prev_theta_track = theta_track
        self.prev_t = now

        # Publish the drive command
        drive_cmd = AckermannDriveStamped()
        drive_cmd.drive.speed = self.v
        drive_cmd.drive.steering_angle = np.clip(delta + self.OFFSET, -self.MAX_TURN, self.MAX_TURN)
        self.drive_pub.publish(drive_cmd)

    def odom_callback(self, msg):
        """
        Predicts where the robot actually is between pose updates
        """
        if not self.interpolation:
            return

        now = self.get_clock().now()

        dt = (now - self.prev_odom_t).nanoseconds / 1e9

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