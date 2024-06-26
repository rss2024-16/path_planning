#!/usr/bin/env python3
import rclpy
import time
from geometry_msgs.msg import PoseArray
from rclpy.node import Node

from path_planning.utils import LineTrajectory


class LoadTrajectory(Node):
    """ Loads a trajectory from the file system and publishes it to a ROS topic.
    """

    def __init__(self):
        super().__init__("trajectory_loader")

        self.declare_parameter("trajectory", "default")
        self.path = self.get_parameter("trajectory").get_parameter_value().string_value

        # initialize and load the trajectory
        self.trajectory = LineTrajectory(self, "/loaded_trajectory")
        self.get_logger().info(f"Loading from {self.path}")
        self.trajectory.load(self.path)
        self.trajectory.updatePoints(self.trajectory.points[:])

        self.pub_topic = "/trajectory/current"
        self.traj_pub = self.create_publisher(PoseArray, self.pub_topic, 1)

        # need to wait a short period of time before publishing the first message
        time.sleep(0.5)

        # visualize the loaded trajectory
        self.trajectory.publish_viz()

        # send the trajectory
        self.publish_trajectory()

        
        ### MANUAL
        # initialize and load the trajectory
        self.path = '/home/racecar/racecar_ws/src/path_planning/example_trajectories/left-lane.traj'
        self.trajectory_right = LineTrajectory(self, "/loaded_trajectory")
        self.get_logger().info(f"Loading from {self.path}")
        self.trajectory_right.load(self.path)
        self.trajectory_right.updatePoints(self.trajectory_right.points[:])

        self.pub_topic = "/trajectory/right"
        self.traj_pub_right = self.create_publisher(PoseArray, self.pub_topic, 1)

        # need to wait a short period of time before publishing the first message
        time.sleep(0.5)

        # send the trajectory
        print("Publishing trajectory to:", self.pub_topic)
        self.traj_pub_right.publish(self.trajectory_right.toPoseArray())

    def publish_trajectory(self):
        print("Publishing trajectory to:", self.pub_topic)
        self.traj_pub.publish(self.trajectory.toPoseArray())


def main(args=None):
    rclpy.init(args=args)
    load_trajectory = LoadTrajectory()
    rclpy.spin(load_trajectory)
    load_trajectory.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
