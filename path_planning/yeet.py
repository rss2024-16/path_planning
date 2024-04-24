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
import math

"""
TODO:
np argmin
optimize for time
test irl
fix lookahead
bug if goal pose is faceing forward

ros2 launch path_planning sim_yeet.launch.xml
"""

class PurePursuit(Node):
    """ 
    Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = .5  # FILL IN #
        self.speed = 1.0  # FILL IN #
        self.wheelbase_length = 0.3  # FILL IN #

        self.MIN_SPEED = 1.6
        self.MAX_SPEED = 2.5

        self.MAX_LOOKAHEAD = 3.0
        self.MIN_LOOKAHEAD = 0.5

        self.MAX_TURN = 0.15

        self.trajectory = LineTrajectory("/followed_trajectory")

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
        
        self.pointpub = self.create_publisher(MarkerArray,'/points',1)
        self.relative_point_pub = self.create_publisher(MarkerArray,'/relative',1)
        self.circle = self.create_publisher(MarkerArray, '/circle_marker', 1)
        self.intersection = self.create_publisher(Marker,'/intersection',1)
        # self.intersection_1 = self.create_publisher(Marker,'/intersection_1',1)
        # self.intersection_2 = self.create_publisher(Marker, '/intersection_2' ,1)
        
        self.points = None
        self.current_pose = None

    def find_closest_point_on_trajectory(self, relative_points):
        """
        Find the point on the trajectory nearest to the car's position.
        relative points: the points on trajectory, converted into the car's frame of reference
        """
        closest_point , index, closest_segment = None, 0, None
        min_distance = float('inf')
        xdot = np.dot(relative_points[:,0], 1) #dot will return 0 if difference is negative (pt is behind)

        #TODO optimize using argmin??
        for i in range(len(relative_points) - 1):
            segment_start = relative_points[i]
            segment_end = relative_points[i + 1]
            
            if xdot[i]> 0: #in front of car
                distance = self.find_minimum_distance(segment_start, segment_end, np.array([0,0,0]))
                if distance < min_distance:
                    min_distance = distance
                    closest_point = segment_start
                    closest_segment = (segment_start, segment_end)
                    index = i

        # distances = self.find_minimum_distance_array(self.segments, np.array([0,0,0]))
        # self.get_logger().info(f"array: {distances}")
        # self.get_logger().info(f"min distance: {min_distance}")
        
        distance_to_goal = self.distance(np.array([0,0,0]), relative_points[-1]) #distance to goal pose

        if closest_point is None:
            # self.get_logger().info("no points in front of car")
            return True, None, None   
        if distance_to_goal < 0.001: 
            # self.get_logger().info("close enough to goal")
            return True, None, None
        return closest_point, index, distance_to_goal 
    

    def find_circle_intersection(self, center, radius, p1, p2, R):
        """
        https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm/86428#86428
        done from car's frame of reference, car center is origin

        some notes:
        got rid of theta for function to work 
        """
        Q = np.array([center[0], center[1]])     # Centre of circle
        r = radius                               # Radius of circle
        P1 = np.array([p1[0], p1[1]])    # Start of line segment
        P2 = np.array([p2[0], p2[1]])    # End of line segment
        V = P2-P1  # Vector along line segment
        a = np.dot(V, V)
        b = 2*np.dot(V, P1 - Q)
        c = np.dot(P1, P1) + np.dot(Q, Q) - 2*np.dot(P1, Q) - r**2
        epsilon = 0.001

        disc = b**2 - 4 * a * c
        if disc < 0: #line misses the circle entirely, should never happen
            # self.get_logger().info("circle missed")
            return False, None
        
        sqrt_disc = disc**0.5
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        # self.get_logger().info(f"t1: {t1}")
        # self.get_logger().info(f"t2: {t2}")
  
        if not (0 <= t1 <= 1+epsilon or 0 <= t2 <= 1+epsilon): #line misses the circle (but would hit it if extended)
            # self.get_logger().info("extend")
            return False, None
        
        # intersection_1 = P1 + t1*V
        # intersection_1 = np.array([intersection_1[0], intersection_1[1], 0])
        # global_intersect_1 = np.matmul(intersection_1, np.linalg.inv(R)) + self.current_pose
        # self.intersection_1.publish(self.to_marker(global_intersect_1, 0, [0.0, 1.0, 0.0], 0.5))

        # intersection_2 = P1 + t2*V
        # intersection_2 = np.array([intersection_2[0], intersection_2[1], 0])
        # global_intersect_2 = np.matmul(intersection_2, np.linalg.inv(R)) + self.current_pose
        # self.intersection_2.publish(self.to_marker(global_intersect_2, 0, [0.0, 0.0, 1.0], 0.5))
        
        if 0 <= t1 <= 1+epsilon:
            return True, P1 + t1*V
        
        else:
            return True, P1 + t2*V
        
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

        if self.points is not None:
            differences = self.points - self.current_pose
            relative_points = np.array([np.matmul(i,R) for i in differences])
            closest_point, index, distance_to_goal = self.find_closest_point_on_trajectory(relative_points)
            # self.get_logger().info("index: " + str(index))
            # self.get_logger().info("distance to goal: " + str(distance_to_goal))

            drive_cmd = AckermannDriveStamped()

            if isinstance(closest_point, bool):
                drive_cmd.drive.speed = 0.0
                drive_cmd.drive.steering_angle = 0.0
            else:
                self.speed = 4.0
                self.lookahead = 3.0
                if self.lookahead > distance_to_goal:
                    self.lookahead = distance_to_goal
                # self.get_logger().info("distance: " + str(self.lookahead))
                self.publish_circle_marker(self.current_pose, self.lookahead)

                #finding the circle intersection 
                success = False
                i = index
                while not success and i < len(relative_points) - 1:
                    segment_start = relative_points[i]
                    segment_end = relative_points[i + 1]
                    segment = (segment_start, segment_end)
                    success, intersect = self.find_circle_intersection(np.array([0,0,0]), self.lookahead, segment_start, segment_end, R)
                    i += 1
                    # self.publish_marker_array(self.relative_point_pub, segment, R, self.current_pose)

                if not success:
                    pass
                    # self.get_logger().info("No intersection found")
                else:
                    #pure pursuit formula
                    intersect = np.array([intersect[0], intersect[1], 0])
                    turning_angle = np.arctan2(2 * self.wheelbase_length * intersect[1], self.lookahead**2)
                
                    if abs(turning_angle) > self.MAX_TURN:
                        turning_angle = self.MAX_TURN if turning_angle > 0 else -self.MAX_TURN
                    
                    drive_cmd.drive.speed = self.speed
                    drive_cmd.drive.steering_angle = turning_angle
                    global_intersect = np.matmul(intersect, np.linalg.inv(R)) + self.current_pose
                    self.intersection.publish(self.to_marker(global_intersect, 0, [0.0, 1.0, 0.0], 0.5))

            self.drive_pub.publish(drive_cmd)

        
    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def find_minimum_distance(self, v, w, p):
        """
        returns the minimum distance between point p and the line segment vw
        """
        distance_squared = (v[0] - w[0])**2 + (v[1] - w[1])**2
        if distance_squared == 0.0:
            return self.distance(p, v)
        # self.get_logger().info(f"distance squared: {distance_squared}")
        t = max(0, min(1, np.dot(p - v, w - v) / distance_squared))
        projection = v + t * (w - v)
        return self.distance(p, projection)

    def find_minimum_distance_array(self, segments, p):
        """
        Returns the minimum distance between point p and an array of line segments.
        Each row of `segments` represents a line segment with columns [v, w].
        """
        v = segments[:, 0]  # Extract the v points
        w = segments[:, 1]  # Extract the w points
        
        distance_squared = np.sum((v - w)**2, axis=1)
        zero_indices = np.where(distance_squared == 0.0)
        non_zero_indices = np.where(distance_squared != 0.0)
        
        distances = np.empty(len(segments))
        
        # Handle the case where distance_squared == 0.0
        # self.get_logger().info(f"distance squared two: {distance_squared}")
        if len(zero_indices[0]) > 0:
            distances[zero_indices] = np.linalg.norm(p - v[zero_indices], axis=1)
    
        # Handle the case where distance_squared != 0.0
        t = np.clip(np.sum((p - v[non_zero_indices]) * (w[non_zero_indices] - v[non_zero_indices]), axis=1) / distance_squared[non_zero_indices], 0, 1)
        # print(f"t: {t.shape}")
        projection = v[non_zero_indices] + t[:, np.newaxis] * (w[non_zero_indices] - v[non_zero_indices])
        distances[non_zero_indices] = np.linalg.norm(p - projection, axis=1)
        
        return distances
            

    def trajectory_callback(self, msg):
        """
        msg: PoseArray
        geometry_msgs/msg/Pose[] poses
            geometry_msgs/msg/Point position
            geometry_msgs/msg/Quaternion orientation
        """
        # self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.points = np.array([(i.position.x,i.position.y,0) for i in msg.poses]) #no theta needed
        self.segments = np.array([[self.points[i], self.points[i+1]] for i in range(len(self.points) - 1)])
        #shape (85, 2, 3) 2 is num points, 3 is x,y,theta
        # self.get_logger().info(f"segments: {self.segments}, shape: {self.segments.shape}")

        # markers = []
        # count = 0
        # for p in self.points:
            
        #     marker = self.to_marker(p,count)

        #     markers.append(marker)
        #     count+=1

        # markerarray = MarkerArray()
        # markerarray.markers = markers

        # self.pointpub.publish(markerarray)

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
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
    

    def publish_circle_marker(self, center, radius):
        markers = []
        num_points = 50  # Number of points to approximate the circle
        angle_increment = 2 * math.pi / num_points
        for i in range(num_points + 1):
            point = (center[0] + radius * math.cos(i * angle_increment), center[1] + radius * math.sin(i * angle_increment), 0)
            markers.append(self.to_marker(point, i))

        markerarray = MarkerArray()
        markerarray.markers = markers
        self.circle.publish(markerarray)

    def publish_marker_array(self, publisher, point_array, R, car_pos, rgb=[1.0,0.0,0.0],):
        """
        converts the point array from car to global frame and publishes the markerarray
        """
        markers = []
        count = 0
        for p in point_array:
            global_p = np.matmul(p, np.linalg.inv(R)) + car_pos
            marker = self.to_marker(global_p, count, rgb=rgb)
            markers.append(marker)
            count+=1

        markerarray = MarkerArray()
        markerarray.markers = markers
        publisher.publish(markerarray)


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()




    