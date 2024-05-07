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
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('drive_topic', "/drive")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value        
        
        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        
        self.pose_sub = self.create_subscription(Odometry, 
                                                self.odom_topic, 
                                                self.pose_callback, 
                                                1)

        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)

        self.transform = lambda theta: np.array([[np.cos(theta),    -np.sin(theta), 0],
                                                [np.sin(theta),     np.cos(theta),  0],
                                                [0,                 0,              1]])

        self.k = 0.25          # Steering constant
        self.v = 2.5        # Constant velocity for now
        self.MAX_TURN = 0.33

    def trajectory_callback(self, msg):
        """
        Takes in a pose array trajectory and converts it to our shitty representation of a trajectory 
        """
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

    def pose_callback(self, odometry_msg):
        """
        Takes in estimated pose from our localization, and tells the robot how to steer
        """
        if len(self.trajectory.points) == 0:
            drive_cmd = AckermannDriveStamped()
            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_cmd)
            return

        # Unpack the odom message
        x = odometry_msg.pose.pose.position.x
        y = odometry_msg.pose.pose.position.y
        orientation = euler_from_quaternion((
            odometry_msg.pose.pose.orientation.x,
            odometry_msg.pose.pose.orientation.y,
            odometry_msg.pose.pose.orientation.z,
            odometry_msg.pose.pose.orientation.w))
        theta = orientation[2]

        # Current position in the world frame (why is odom in the world frame?)
        current_pose = np.array([x, y, theta])

        # Previous and next points
        prev_i = self.find_next_index(current_pose)

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
        theta_xc = np.arctan(self.k * cross_track / self.v)

        # Direction of the trajectory (where we should be pointing)
        theta_track = prev_p[2]

        # Heading error
        psi = theta_track - theta
        psi = (psi + np.pi) % (2 * np.pi) - np.pi

        # Updated steering
        delta = psi + theta_xc

        # Publish the drive command
        drive_cmd = AckermannDriveStamped()
        drive_cmd.drive.speed = self.v
        drive_cmd.drive.steering_angle = np.clip(delta, -self.MAX_TURN, self.MAX_TURN)
        self.drive_pub.publish(drive_cmd)

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