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
from path_planning.visualization_tools import VisualizationTools


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
        self.MAX_SPEED = 3.0

        self.MAX_LOOKAHEAD = 3.0
        self.MIN_LOOKAHEAD = 0.7

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

        self.intersectpub = self.create_publisher(Marker, '/intersect', 1 )

        self.pointpub = self.create_publisher(MarkerArray,'/points',1)
        self.intersectinopub = self.create_publisher(MarkerArray,'/intersections',1)

        self.closestpub = self.create_publisher(Marker,'/closest_point',1)

        self.MAX_TURN = 0.25

        self.points = None
        self.current_pose = None
        self.relative_points = None
        self.intersections = None
        self.intersect_to_line = None
        self.lines = None
        self.point_history = set()
        self.last_point = None

        self.transform = lambda theta: np.array([ [np.cos(theta), -np.sin(theta), 0],
                                             [np.sin(theta), np.cos(theta), 0],
                                             [0, 0, 1]
                                            ])
        
        self.slopes = []
        self.controls = []

        self.path_pub = self.create_publisher(Marker, '/intersect', 10)

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
        # publish this for debugging
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
        if self.current_pose is not None and self.intersections is not None:
            # self.get_logger().info(f'curr pose: {self.current_pose}')

            R = self.transform(self.current_pose[2])
            pose_init = self.current_pose
            #get transform matrix between global and robot frame
            # self.get_logger().info(f'curr_pose: {self.current_pose}')

            differences = self.intersections - self.current_pose

            relative_points = np.array([np.matmul(i,R) for i in differences])
            # self.get_logger().info(f'{relative_points}')
            #multiply each difference by transformation matrix to get
            #poses in the robot frame

            #check that the point is in front of current pose
            xdot = np.dot(relative_points[:,0], 1) #dot will return 0 if difference is negative (pt is behind)

            filtered_points = relative_points[(xdot > 0.4)] #filter to only look at points ahead (same direction)

            if len(filtered_points) == 0:
                if xdot[-1] >= 0:
                    filtered_points = np.array([relative_points[-1]])
                else:
                    self.get_logger().info("No points ahead of car")
                    return True

            distances = np.linalg.norm(filtered_points,axis=1)
            # behind_dist = np.linalg.norm(behind_points,axis=1)
            closest_point = filtered_points[np.argmin(distances)]
            # if tuple(closest_point) not in self.point_history:
            #     self.get_logger().info(f'ahsjdhaksd')
            #     #if closest point has been traveled, choose second closest
            #     closest_point = filtered_points[np.argsort(distances)[1]]
            # smallest_back = np.argmin(behind_dist)
            # closest_front = filtered_points[smallest_front]
            # if len(behind_points) == 0:
            #     closest_behind = closest_front
            # else:
            #     closest_behind = behind_points[smallest_back]


            # closest_xy_global = np.matmul(np.linalg.inv(R),closest_xy)+pose_init
            # closest_xy2_global =

            # marker = self.to_marker(closest_xy_global,rgb=[0.0,0.5,0.5],scale=0.5)
            # self.closestpub.publish(marker)
            # self.get_logger().info(f'{closest_front},{closest_behind}')
            # slope,y_int = np.polyfit([closest_behind[0],closest_front[0]],[closest_behind[1],closest_front[1]],1)

            return closest_point

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
        # if closest_point is not None and self.last_point is not None and tuple(closest_point) != self.last_point:
        #     self.point_history.add(tuple(closest_point))
        #     self.last_point = tuple(closest_point)

        if isinstance(closest_point,bool):
            drive_cmd = AckermannDriveStamped()

            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0

            self.drive_pub.publish(drive_cmd)


        elif closest_point is not None:
            relative_x, relative_y = closest_point[:2]
            slope = relative_y/relative_x
            # EPS = 0.05
            # if abs(slope) < EPS:
            pure_pursuit = True
            # else:
                # pure_pursuit = False

            if not pure_pursuit:
                
                kp = 1.0
                kd = 1.0/6.0
                ki = 1.0/8.0
                P = -kp*np.arctan2(relative_y,relative_x)
                if len(self.controls) > 5:
                    if len(self.controls) < 10:
                        sample = self.controls
                    else:
                        sample = self.controls[-10:]

                    I = -ki*np.trapz(sample)
                    slope, _ = np.polyfit(np.arange(len(sample)),sample,1)
                    D = -kd*slope
                    control = P+I+D
                else:
                    control = P
                
                self.get_logger().info(f'{control}')
                self.controls.append(control)
                if abs(control) > self.MAX_TURN:
                    control = self.MAX_TURN if control > 0 else -self.MAX_TURN

                drive_cmd = AckermannDriveStamped()
                drive_cmd.drive.speed = 1.6

                drive_cmd.drive.steering_angle = control

                self.drive_pub.publish(drive_cmd)
                

            elif pure_pursuit:
                # slope,y = self.intersect_to_line[tuple(closest_point)][0]

                self.slopes.append(slope)
                self.get_logger().info(f'{slope}')

                # self.speed = 3/(10*abs(slope))
                self.speed = 4*np.exp(-2*abs(slope))
                # self.speed = 2.0
                if self.speed > self.MAX_SPEED:
                    self.speed = self.MAX_SPEED
                elif self.speed < self.MIN_SPEED:
                    self.speed = self.MIN_SPEED

                # self.lookahead = np.linalg.norm(np.array([relative_x,relative_y]))
                # self.lookahead = 1.0
                # self.lookahead = 3/(10*abs(slope))
                # DIST = 0.75
                closest_dist = np.linalg.norm(np.array([relative_x,relative_y]))
                if abs(slope) > 0.7 and closest_dist < 0.75:
                    # self.lookahead = closest_dist
                    self.lookahead = (self.speed*1.25) * np.exp(-1.3*abs(slope))
                    OFFSET = 0
                else:
                    self.lookahead = self.speed
                    OFFSET = -0.05

                if self.lookahead < self.MIN_LOOKAHEAD:
                    self.lookahead = self.MIN_LOOKAHEAD

                intersect = self.circle_intersection(slope,0,self.lookahead)

                ang_dest = np.linspace(0, 2*np.pi, 20)
                x_dest = intersect[0] + 0.1 * np.cos(ang_dest)
                y_dest = intersect[1] + 0.1 * np.sin(ang_dest)
                turning_angle = np.arctan2(2 * self.wheelbase_length * intersect[1], self.lookahead**2)
                turning_angle += OFFSET
                # self.get_logger().info('hi1')
                VisualizationTools.plot_line(x_dest, y_dest, self.path_pub, frame='/base_link', color=(0., 1., 0.))
                # self.get_logger().info('hi2')
                if abs(turning_angle) > self.MAX_TURN:
                    turning_angle = self.MAX_TURN if turning_angle > 0 else -self.MAX_TURN

                drive_cmd = AckermannDriveStamped()
                
                drive_cmd.drive.speed = self.speed
                drive_cmd.drive.steering_angle = turning_angle

                self.drive_pub.publish(drive_cmd)

    def get_intersections(self):
        '''
        Returns:
        intersect_to_line - dict mapping intersect to the lines it intersects with
        intersections - list of (x,y) intersections
        lines - list of (slope,y_int) that replicate line
        '''
        # segments = []
        path = self.points

        orientation = lambda p1,p2: np.arctan2( (p2[1]-p1[1]),(p2[0]-p1[0]) )

        idx = 1
        # segment = [path[0]]
        intersections = [path[0]]
        # intersect_to_line = {tuple(path[0]): []}
        # lines = []
        p = path[0]

        eps = 1e-3

        last_angle = None
        # last_p = tuple(path[0])
        while idx < len(path):
            p2 = path[idx]
            angle = orientation(p2,p)
            if last_angle is None or abs(angle-last_angle) < eps:
                pass
            else:
                intersections.append(p)
                # slope,y_int = np.polyfit([last_p[0],p[0]],[last_p[1],p[1]],1)
                # lines.append((slope,y_int))
                # intersect_to_line[last_p].append((slope,y_int))
                # intersect_to_line[tuple(p)] = [(slope,y_int)]
                # last_p = tuple(p)
            last_angle = angle
            p = path[idx]
            idx+=1

        intersections.append(path[-1])
        self.intersections = intersections
        # self.intersect_to_line = intersect_to_line
        # self.get_logger().info(f'{self.intersect_to_line}')
        # self.lines = lines
    
    def plot_intersections(self):
        markers = []
        id = 0
        for i in self.intersections:
            m = self.to_marker(i,rgb=[0.5,0.0,0.5],id=id)
            id+=1
            markers.append(m)

        pub = MarkerArray()
        pub.markers = markers
        self.intersectinopub.publish(pub)
    
    # def plot_segments(self):
    #     markers = []
    #     id = 0
    #     for i in self.intersections:
    #         s = self.to_marker(i[0],rgb=[0.2,0.6,0.2],id=id)
    #         id+=1
    #         # e = self.to_marker(i[-1],rgb=[0.6,0.2,0.2],id=id)
    #         # id+=1
    #         markers.append(s)
    #         # markers.append(e)
    #     pub = MarkerArray()
    #     pub.markers = markers
    #     self.segmentpub.publish(pub)


    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.points = np.array([(i.position.x,i.position.y,0) for i in msg.poses])

        self.get_intersections()
        self.plot_intersections()

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
    try:
        rclpy.spin(follower)
    except KeyboardInterrupt:
        np.save('slopes',follower.slopes)
        pass
    rclpy.shutdown()
