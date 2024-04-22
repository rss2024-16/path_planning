import rclpy

import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseArray, Point
from std_msgs.msg import Header
import os
from typing import List, Tuple
import json
import random
import math

from tf_transformations import euler_from_quaternion

# from skimage.morphology import dilation, erosion
# from skimage.morphology import square, disk

import heapq
from collections import deque
import cv2


EPSILON = 0.00000000001

''' These data structures can be used in the search function
First run
ros2 launch path_planning sim_plan.launch.xml
then 
ros2 launch racecar_simulator simulate.launch.xml
'''
class LineTrajectory:
    """ A class to wrap and work with piecewise linear trajectories. """

    def __init__(self, node, viz_namespace=None):
        self.points: List[Tuple[float, float]] = []
        self.distances = []
        self.has_acceleration = False
        self.visualize = False
        self.viz_namespace = viz_namespace
        self.node = node

        if viz_namespace:
            self.visualize = True
            self.start_pub = self.node.create_publisher(Marker, viz_namespace + "/start_point", 1)
            self.traj_pub = self.node.create_publisher(Marker, viz_namespace + "/path", 1)
            self.end_pub = self.node.create_publisher(Marker, viz_namespace + "/end_pose", 1)

    # compute the distances along the path for all path segments beyond those already computed
    def update_distances(self):
        num_distances = len(self.distances)
        num_points = len(self.points)

        for i in range(num_distances, num_points):
            if i == 0:
                self.distances.append(0)
            else:
                p0 = self.points[i - 1]
                p1 = self.points[i]
                delta = np.array([p0[0] - p1[0], p0[1] - p1[1]])
                self.distances.append(self.distances[i - 1] + np.linalg.norm(delta))

    def distance_to_end(self, t):
        if not len(self.points) == len(self.distances):
            print(
                "WARNING: Different number of distances and points, this should never happen! Expect incorrect results. See LineTrajectory class.")
        dat = self.distance_along_trajectory(t)
        if dat == None:
            return None
        else:
            return self.distances[-1] - dat

    def distance_along_trajectory(self, t):
        # compute distance along path
        # ensure path boundaries are respected
        if t < 0 or t > len(self.points) - 1.0:
            return None
        i = int(t)  # which segment
        t = t % 1.0  # how far along segment
        if t < EPSILON:
            return self.distances[i]
        else:
            return (1.0 - t) * self.distances[i] + t * self.distances[i + 1]

    def addPoint(self, point: Tuple[float, float]) -> None:
        print("adding point to trajectory:", point)
        self.points.append(point)
        self.update_distances()
        self.mark_dirty()

    def clear(self):
        self.points = []
        self.distances = []
        self.mark_dirty()

    def empty(self):
        return len(self.points) == 0

    def save(self, path):
        print("Saving trajectory to:", path)
        data = {}
        data["points"] = []
        for p in self.points:
            data["points"].append({"x": p[0], "y": p[1]})
        with open(path, 'w') as outfile:
            json.dump(data, outfile)

    def mark_dirty(self):
        self.has_acceleration = False

    def dirty(self):
        return not self.has_acceleration

    def load(self, path):
        print("Loading trajectory:", path)

        # resolve all env variables in path
        path = os.path.expandvars(path)

        with open(path) as json_file:
            json_data = json.load(json_file)
            for p in json_data["points"]:
                self.points.append((p["x"], p["y"]))
        self.update_distances()
        print("Loaded:", len(self.points), "points")
        self.mark_dirty()

    # build a trajectory class instance from a trajectory message
    def fromPoseArray(self, trajMsg):
        for p in trajMsg.poses:
            self.points.append((p.position.x, p.position.y))
        self.update_distances()
        self.mark_dirty()
        print("Loaded new trajectory with:", len(self.points), "points")

    def toPoseArray(self):
        traj = PoseArray()
        traj.header = self.make_header("/map")
        for i in range(len(self.points)):
            p = self.points[i]
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            traj.poses.append(pose)
        return traj

    def publish_start_point(self, duration=0.0, scale=0.1):
        should_publish = len(self.points) > 0
        self.node.get_logger().info("Before Publishing start point")
        if self.visualize and self.start_pub.get_subscription_count() > 0:
            self.node.get_logger().info("Publishing start point")
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace + "/trajectory"
            marker.id = 0
            marker.type = 2  # sphere
            marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
            if should_publish:
                marker.action = 0
                marker.pose.position.x = self.points[0][0]
                marker.pose.position.y = self.points[0][1]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                # delete marker
                marker.action = 2

            self.start_pub.publish(marker)
        elif self.start_pub.get_subscription_count() == 0:
            self.node.get_logger().info("Not publishing start point, no subscribers")

    def publish_end_point(self, duration=0.0):
        should_publish = len(self.points) > 1
        if self.visualize and self.end_pub.get_subscription_count() > 0:
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace + "/trajectory"
            marker.id = 1
            marker.type = 2  # sphere
            marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
            if should_publish:
                marker.action = 0
                marker.pose.position.x = self.points[-1][0]
                marker.pose.position.y = self.points[-1][1]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                # delete marker
                marker.action = 2

            self.end_pub.publish(marker)
        elif self.end_pub.get_subscription_count() == 0:
            print("Not publishing end point, no subscribers")

    def publish_trajectory(self, duration=0.0):
        should_publish = len(self.points) > 1
        if self.visualize and self.traj_pub.get_subscription_count() > 0:
            self.node.get_logger().info("Publishing trajectory")
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace + "/trajectory"
            marker.id = 2
            marker.type = marker.LINE_STRIP  # line strip
            marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
            if should_publish:
                marker.action = marker.ADD
                marker.scale.x = 0.3
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                for p in self.points:
                    pt = Point()
                    pt.x = p[0]
                    pt.y = p[1]
                    pt.z = 0.0
                    marker.points.append(pt)
            else:
                # delete
                marker.action = marker.DELETE
            self.traj_pub.publish(marker)
            print('publishing traj')
        elif self.traj_pub.get_subscription_count() == 0:
            print("Not publishing trajectory, no subscribers")

    def publish_viz(self, duration=0):
        if not self.visualize:
            print("Cannot visualize path, not initialized with visualization enabled")
            return
        self.publish_start_point(duration=duration)
        self.publish_trajectory(duration=duration)
        self.publish_end_point(duration=duration)
        print("Path published!")

    def make_header(self, frame_id, stamp=None):
        if stamp == None:
            stamp = self.node.get_clock().now().to_msg()
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return header

class Node():
    """
    Node class for search: each node represents a state of the path with an associated
    pose, parent(previous node), gscore = cost, and fscore = heuristic
    """
    def __init__(self,position,fscore=float('inf'),gscore=float('inf'),parent=None):
        self._pose = position
        self._fscore = fscore
        self._gscore = gscore
        self._parent = parent

    @property
    def pose(self): return self._pose
    
    @property
    def fscore(self): return self._fscore
    
    @property
    def gscore(self): return self._gscore
    
    @property
    def parent(self): return self._parent
    
    def set_gscore(self,score): self._gscore = score
    
    def set_fscore(self,score): self._fscore = score

    def __lt__(self,other): return self.fscore < other.fscore
    
    def extract_path(self):
        """
        Extracts path from start node to current node
        """
        curr = self
        path = [curr.pose]
        while curr.parent is not None:
            path.append(curr.parent.pose)
            curr = curr.parent
        return path[::-1]

class PriorityQueue:
    """
    Priority Queue implementation using heapq
    """
    def __init__(self):
        self.elements = []
        self.element_set = set()

    def empty(self):
        return len(self.elements) == 0

    def put(self, item):
        heapq.heappush(self.elements, item)
        self.element_set.add(item.pose)

    def get(self):
        return heapq.heappop(self.elements)
    
    def __contains__(self,item):
        return item.pose in self.element_set

class Map():
    """
    Occupancy Grid gives information about the location of obstacles
    grid.data lists the occupancy values of map cells: 100 = occupied, 0 = free, -1 = unknown
    """
    def __init__(self, occupany_grid) -> None:
        self._height = occupany_grid.info.height
        self._width = occupany_grid.info.width

        self._resolution = occupany_grid.info.resolution

        p = occupany_grid.info.origin.position #Ros2 Point
        self.origin_p = np.array([[p.x], [p.y], [p.z]]) #3x1

        o = occupany_grid.info.origin.orientation #Ros2 Quaternion
        self.origin_o = [o.x, o.y, o.z, o.w]

        self.R_z = lambda theta: np.array([ [np.cos(theta), -np.sin(theta), 0],
                                             [np.sin(theta), np.cos(theta), 0],
                                             [0, 0, 1]
                                            ])
        
        self.MAX_TURN_RADIUS = 0.34
        
        #2d (int) array of pixel coords indexed by grid[v][u] 

        # self.grid = np.array(occupany_grid.data).reshape((occupany_grid.info.height, occupany_grid.info.width))
        # self.grid = dilation(self.grid,square(10))
        # cv2.imwrite('test.png',self.grid)
        self.grid = np.load('/root/racecar_ws/grid.npy')

        #here we are dilating the map in order to avoid cutting corners
        # self.grid = np.array(occupany_grid.data).reshape((occupany_grid.info.height, occupany_grid.info.width))
        # self.grid = erosion(self.grid, disk(8))
        # np.save('grid.npy',self.grid)
        # cv2.imwrite('test.png',self.grid)

        self.x_step = abs(self.pixel_to_xy(0, 0)[0] - self.pixel_to_xy(1, 0)[0])
        self.MAX_DIST = 5

    @property
    def height(self) -> int: return self._height

    @property
    def width(self) -> int: return self._width  

    @property
    def resolution(self) -> float: return self._resolution   

    def __len__(self) -> int: return self._height * self._width   

    def xy_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """
        Converts x,y point in map frame to u,v pixel in occupancy grid
        """
        q = np.array([[x], [y], [0]]) #3x1 x,y,z

        q = q - self.origin_p
        
        _, _, yaw = euler_from_quaternion(self.origin_o)

        q = np.matmul(self.R_z(-yaw),q)

        q = q / self._resolution

        u, v = q[:2, :]

        return (int(round(u[0])), int(round(v[0])))
    
    def pixel_to_xy(self, u: int, v: int) -> Tuple[float, float]:
        """
        Converts u,v pixel to x,y point in map frame
        """
        pixel = np.array([[u], [v], [0]])

        pixel = pixel * self._resolution 

        _, _, yaw = euler_from_quaternion(self.origin_o)

        q = np.matmul(self.R_z(yaw),pixel)

        q = q + self.origin_p

        x, y = q[:2, :]

        return (x[0], y[0])
    
    def is_free(self, u, v) -> bool:
        return self.grid[v][u] == 0
    
    def astar(self, start: Tuple[float,float], goal: Tuple[float,float]):
        '''
        simple alg taken from 
        https://en.wikipedia.org/wiki/A*_search_algorithm
        '''
        start_pose = self.discretize_point(start)
        goal = self.discretize_point(goal)

        h = lambda x,y: ( (y[0]-x[0])**2 + (y[1]-x[1])**2 )**(1/2)
        #heuristic is just Euclidean distance

        nodelookup = {}

        start = Node(start_pose,parent=None,gscore=0,fscore=h(start_pose,goal))
        nodelookup[start_pose] = start

        q = PriorityQueue()
        q.put(start)

        while not q.empty():
            node = q.get()

            if node.pose == goal:
                return node.extract_path()
            
            for n in self.get_neighbors(node.pose):
                try:
                    n_obj = nodelookup[n]
                except KeyError:
                    n_obj = Node(n,parent=node)
                    nodelookup[n] = n_obj

                tentative_gscore = node.gscore + h(node.pose,n_obj.pose)

                if tentative_gscore < n_obj.gscore:
                    n_obj.set_gscore(tentative_gscore)
                    n_obj.set_fscore(tentative_gscore + h(n_obj.pose,goal))
                    if n_obj not in q:
                        q.put(n_obj)
    
    def bfs(self, start: Tuple[float, float], goal: Tuple[float, float]):
        """
        start: tuple of (x, y) coord in map frame
        goal: tuple of (x, y) coord in map frame

        Returns path from start to goal
        """
        start = self.discretize_point(start)
        goal = self.discretize_point(goal)
        
        visited = {start}
        queue = deque([start])
        parent = {start: None}

        end = None

        while queue:
            current = queue.popleft() 
            if current == goal:
                end = current
                break
            for n in self.get_neighbors(current):
                if n not in visited:
                    visited.add(n)
                    queue.append(n)
                    parent[n] = current 

        # if no path was found
        if end not in parent:
            return []
        
        i = end
        path = [end]
        while i != start:
            i = parent[i]
            path.append(i)
        
        return path[::-1] #path start -> goal in tuples of x,y point nodes
    
    def generate_circle(self,point: Tuple[float,float]):
        u,v = self.xy_to_pixel(point)

    def prune_path(self,path):
        '''
        gets rid of unnecessary (low slope) points

        need to fix
        '''
        EPS = 0.01

        p = path[0]

        idx = 1

        while idx != len(path)-1:
            diff = np.array(p) - np.array(path[idx])
            orientation = np.arctan2(diff[1],diff[0])
            diff = np.array([diff[0],diff[1],orientation])
            R = self.R_z(orientation)

            relative = np.matmul(diff,R)
            slope = relative[0]/relative[1]
            
            try:
                if slope < EPS:
                    path[idx] = 0
                else:
                    p = path[idx]
                
                prev1 = s1
                prev2 = s2
            except ZeroDivisionError: #one of the slopes are 0 so line is straight
                path[idx] = 0
            idx+=1
        
        return [i for i in path if i!=0]

    def rrt(self, start: Tuple[float, float], goal: Tuple[float, float]):
        """
        RRT done in continuous space, without discretization
        """
        ## NOTES:
        # Need to implement max length of a line segment
        # Dynamics on trajectories
        # Even RRT* might be suboptimal for compute time? Probably compare
        # Divide by zero protection in straight path

        class Node():
            def __init__(self, path, loc):
                self.path = path
                self.loc = loc
                self.parent = None
                self.children = []

        def dist(loc1, loc2):
            return (loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2

        def find_nearest(loc, parent):
            min_dist = dist(parent.loc, loc)
            near_node = parent

            for child in parent.children:
                near_child, child_dist = find_nearest(loc, child)

                if child_dist <= min_dist:
                    min_dist = child_dist
                    near_node = near_child

            return near_node, min_dist

        def straight_path(begin, end):
            dy = end[1] - begin[1]
            dx = end[0] - begin[0]
            
            length = math.sqrt(dist(begin, end))
            if length > self.MAX_DIST:
                end = (begin[0] + dx / length * self.MAX_DIST, begin[1] + dy / length * self.MAX_DIST)

            slope = dy / dx 

            step = self.x_step if begin[0] < end[0] else -self.x_step

            path = [end]
            stepped = (end[0] - step, end[1] - slope * step)

            while (stepped[0] > begin[0] and begin[0] < end[0]) or (stepped[0] < begin[0] and begin[0] > end[0]):
                path.append(stepped)
                stepped = (stepped[0] - step, stepped[1] - slope * step)

            return path, end

        def collision_free(path):
            for point in path:
                u, v = self.xy_to_pixel(point[0], point[1])
                if not self.is_free(u, v):
                    return False
            
            return True

        def path_to(node):
            full_path = node.path

            if node.parent != None:
                full_path = full_path + path_to(node.parent)

            return full_path

        # Add start to the tree
        head = Node([start], start)
        previous = head

        # While previous point was not goal
        samples = 0
        max_samples = 50000
        while previous.loc != goal and samples < max_samples: ## THIS TUPLE COMPARISON MIGHT NOT WORK, IDK

            # Pick a random grid cell or sample goal with set probability
            if random.random() > 0.15:
                target = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
                target = self.pixel_to_xy(target[0], target[1])
            else:
                target = goal

            # Find the nearest node in the tree
            nearest, _ = find_nearest(target, head)

            # Extend the newest point to the nearest node
            nearest_path, act_target = straight_path(nearest.loc, target)

            # If the extended path has no collisions
            if collision_free(nearest_path):
            
                # Add the newest point to the tree
                newest_node = Node(nearest_path, act_target)
                newest_node.parent = nearest
                nearest.children.append(newest_node)

                # Set this new point as our previous node
                previous = newest_node

            samples += 1

        if samples < max_samples:
            return path_to(previous)
        else:
            return None

    def get_neighbors(self, point: Tuple[float, float]) -> List[Tuple[float, float]]:
        x, y = point
        neighbors = []

        # radius = 8
        step = 1.0
        possibilities = [(-step, 0), (0, step), (step, 0), (0, -step), (step, step), (step, -step), (-step, step), (-step, -step)]
        for (dx, dy) in possibilities:
            u, v = self.xy_to_pixel(x + dx, y + dy)
            if not self.is_free(u, v):
                neighbors = []
                break
            if (0 <= u and u < self._width) and (0 <= v and v < self._height) and self.is_free(u,v):
                neighbors.append((x + dx, y + dy)) 
        
        for (dx, dy) in possibilities:
            u, v = self.xy_to_pixel(x + dx/2, y + dy/2)
            if (0 <= u and u < self._width) and (0 <= v and v < self._height) and self.is_free(u,v):
                neighbors.append((x + dx/2, y + dy/2)) 



        # radius = 4
        # step = 0.5
        # possibilities = [(-step, 0), (0, step), (step, 0), (0, -step), (step, step), (step, -step), (-step, step), (-step, -step)]
        # for i, (dx, dy) in enumerate(possibilities):
        #     u, v = self.xy_to_pixel(x + dx, y + dy)
        #     if not self.is_free(u, v):
        #         break
        #     if (0 <= u and u < self._width) and (0 <= v and v < self._height) and self.is_free(u,v):
        #         neighbors.append((x + dx, y + dy)) 




        # cuts corner on path pruning
        # radius = 7
        # step = 1.0
        # possibilities = [(-step, 0), (0, step), (step, 0), (0, -step), (step, step), (step, -step), (-step, step), (-step, -step)]
        # for (dx, dy) in possibilities:
        #     u, v = self.xy_to_pixel(x + dx, y + dy)
        #     if not self.is_free(u, v):
        #         u, v = self.xy_to_pixel(x + dx/2, y + dy/2)
        #         if (0 <= u and u < self._width) and (0 <= v and v < self._height) and self.is_free(u,v):
        #             neighbors.append((x + dx/2, y + dy/2))

        #     elif (0 <= u and u < self._width) and (0 <= v and v < self._height):
        #         neighbors.append((x + dx, y + dy))  
        
        return neighbors

    def discretize_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Discretizes a point (x, y) to a node in the space such that nodes are the
        center of each 1x1 grid square. In the case where a point is on the edge
        of a grid square, the point will be assigned to the center with decreasing 
        x and increasing y.
        """
        x, y = point

        mid = 0.5

        new_x = int(x)
        new_y = int(y)

        if x >= 0 and y >= 0:
            if x - new_x == 0:
                new_x -= mid
            else:
                new_x += mid
            
            new_y += mid
        
        elif x < 0 and y >= 0:
            new_x -= mid
            new_y += mid

    
        elif x <= 0 and y < 0:
            if y - new_y == 0:
                new_y += mid
            else:
                new_y -= mid
            
            new_x -= mid 

        elif x > 0 and y < 0:
            if y - new_y == 0:
                new_y += mid
            else:
                new_y -= mid

            if x - new_x == 0:
                new_x -= mid
            else:
                new_x += mid

        return (new_x, new_y)

