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

from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        # self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        # self.drive_topic = '/vesc/low_level/ackermann_cmd'
        self.drive_topic = '/vesc/input/navigation'

        self.lookahead = .5  # FILL IN #
        self.speed = 1.0  # FILL IN #
        self.wheelbase_length = 0.3  # FILL IN #

        self.MIN_SPEED = 1.6
        self.MAX_SPEED = 2.5

        self.MAX_LOOKAHEAD = 6.0
        self.MIN_LOOKAHEAD = 3.0

        self.trajectory = LineTrajectory("/followed_trajectory")
        # self.get_logger().info('/followed_trajectory')

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        
        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_callback,1)

        self.pointpub = self.create_publisher(MarkerArray,'/points',1)

        self.closestpub = self.create_publisher(Marker,'/closest_point',1)

        

        self.MAX_TURN = 0.15
        
        self.points = None
        self.current_pose = None
        self.relative_points = None

        self.transform = lambda theta: np.array([ [np.cos(theta), -np.sin(theta), 0],
                                             [np.sin(theta), np.cos(theta), 0],
                                             [0, 0, 1]
                                            ])

        # self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # self.converge_pub = self.create_publisher(Float32, 'converge', 10)

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

    def circle_intersection(self, line_slope, line_intercept, circle_radius):
        # Coefficients of the quadratic equation
        a = line_slope**2 + 1
        b = 2 * line_slope * line_intercept
        c = line_intercept**2 - circle_radius**2

        # Solve the quadratic equation for x
        x_solutions = np.roots([a, b, c])

        # Find corresponding y values using the line equation
        intersection_points = [(float(x_val), float(line_slope * x_val + line_intercept)) for x_val in x_solutions]
        
        # +X forward, +Y left, -Y right

        if intersection_points[0][0] > 0 and intersection_points[1][0] < 0:
            return intersection_points[0]
        elif intersection_points[0][0] < 0 and intersection_points[1][0] > 0:
            return intersection_points[1]
        elif intersection_points[0][1] > 0 and self.SIDE == -1 or intersection_points[0][1] < 0 and self.SIDE == 1:
            return intersection_points[0]
        else:
            return intersection_points[1]
        
    

    def find_closest_point(self):
        '''
        Finds the closest point that is in front of the car
        '''
        if self.current_pose is not None and self.points is not None:
            self.get_logger().info(f'curr pose: {self.current_pose}')

            R = self.transform(self.current_pose[2])
            pose_init = self.current_pose
            #get transform matrix between global and robot frame

            differences = self.points - self.current_pose

            relative_points = np.array([np.matmul(i,R) for i in differences])
            self.get_logger().info(f'{relative_points}')
            #multiply each difference by transformation matrix to get
            #poses in the robot frame

            #check that the point is in front of current pose
            # xdot = np.dot(relative_points[:,0], 1) #dot will return 0 if difference is negative (pt is behind)
            DIST = 0.25
            # filtered_points = relative_points[(xdot[0] >= .25)] #filter to only look at points ahead (same direction)
            filtered_points = [i for i in relative_points if i[0]>=DIST]
            if len(filtered_points) == 0:
                self.get_logger().info("No points ahead of car")
                return True

            distances = np.linalg.norm(filtered_points,axis=1)

            closest_xy = filtered_points[np.argmin(distances)]


            closest_xy_global = np.matmul(np.linalg.inv(R),closest_xy)+pose_init

            marker = self.to_marker(closest_xy_global,rgb=[0.0,0.5,0.5],scale=0.5)
            self.closestpub.publish(marker)

            return np.array(closest_xy)

    def pose_callback(self, odometry_msg):
        '''
        updates pose from localization
        '''
        x = odometry_msg.pose.pose.position.x
        y = odometry_msg.pose.pose.position.y

        orientation = euler_from_quaternion((
        odometry_msg.pose.pose.orientation.x,
        odometry_msg.pose.pose.orientation.y,
        odometry_msg.pose.pose.orientation.z,
        odometry_msg.pose.pose.orientation.w))

        theta = orientation[2]

        # R = self.transform(theta)

        #car's coordinates in global frame
        self.current_pose = np.array([x,y,theta]) 

        closest_point = self.find_closest_point()
        
        if isinstance(closest_point,bool):  #then no more points in front of car, stop
            drive_cmd = AckermannDriveStamped()
            
            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0

            self.drive_pub.publish(drive_cmd)
            

        elif closest_point is not None:
            relative_x, relative_y = closest_point[:2]

            slope = relative_y/relative_x
            # self.get_logger().info(f'{slope}')

            self.speed = 6/(10*abs(slope))
            if self.speed > self.MAX_SPEED:
                self.speed = self.MAX_SPEED
            elif self.speed < self.MIN_SPEED:
                self.speed = self.MIN_SPEED

            # self.lookahead = np.linalg.norm(np.array([relative_x,relative_y])) / 2
            self.lookahead = 3/(10*abs(slope))
            if self.lookahead > self.MAX_LOOKAHEAD:
                self.lookahead = self.MAX_LOOKAHEAD
            elif self.lookahead < self.MIN_LOOKAHEAD:
                self.lookahead = self.MIN_LOOKAHEAD

            intersect = self.circle_intersection(slope,0,self.lookahead)

            turning_angle = np.arctan2(2 * self.wheelbase_length * intersect[1], self.lookahead**2)
            
            if abs(turning_angle) > self.MAX_TURN:
                turning_angle = self.MAX_TURN if turning_angle > 0 else -self.MAX_TURN

            drive_cmd = AckermannDriveStamped()
            drive_cmd.drive.speed = self.speed
            drive_cmd.drive.steering_angle = turning_angle

            self.drive_pub.publish(drive_cmd)


    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.points = np.array([(i.position.x,i.position.y,0) for i in msg.poses]) #no theta needed

        markers = []
        count = 0
        for p in self.points:
            
            marker = self.to_marker(p,count)

            markers.append(marker)
            count+=1

        markerarray = MarkerArray()
        markerarray.markers = markers

        self.pointpub.publish(markerarray)


        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
