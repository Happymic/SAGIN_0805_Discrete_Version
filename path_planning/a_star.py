import heapq
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, position, g_cost, h_cost, parent=None):
        self.position = tuple(position)
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)


class AStar:
    def __init__(self, env, resolution=1.0, agent_type="ground", agent_size=1.0):
        self.env = env
        self.resolution = resolution
        self.agent_type = agent_type
        self.agent_size = agent_size
        self.max_iterations = 50000
        self.directions = [
            (x, y, 0) for x in [-1, 0, 1] for y in [-1, 0, 1]
            if (x, y) != (0, 0)
        ]
        self.goal = None  # Initialize goal attribute

    def plan(self, start, goal):
        self.goal = goal  # Set the goal for this planning session
        start_node = Node(start, 0, self.heuristic(start, goal))
        goal_node = Node(goal, 0, 0)

        open_set = {}
        closed_set = set()

        open_set[start_node.position] = start_node

        iterations = 0
        while open_set and iterations < self.max_iterations:
            current = min(open_set.values(), key=lambda n: n.f_cost)

            if self.is_goal_reached(current, goal_node):
                return self.reconstruct_path(current)

            del open_set[current.position]
            closed_set.add(current.position)

            for neighbor in self.get_neighbors(current):
                if neighbor.position in closed_set:
                    continue

                if neighbor.position not in open_set:
                    open_set[neighbor.position] = neighbor
                elif neighbor.g_cost < open_set[neighbor.position].g_cost:
                    open_set[neighbor.position] = neighbor

            iterations += 1

        logger.warning(f"A* path planning reached maximum iterations ({self.max_iterations})")
        return None

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]

    def get_neighbors(self, node):
        neighbors = []
        for direction in self.directions:
            new_pos = tuple(np.array(node.position) + np.array(direction) * self.resolution)
            if self.is_valid(new_pos):
                g_cost = node.g_cost + self.resolution * np.linalg.norm(direction)
                h_cost = self.heuristic(new_pos, self.goal)
                neighbors.append(Node(new_pos, g_cost, h_cost, node))
        return neighbors

    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def is_valid(self, position):
        return (0 <= position[0] < self.env.world.width and
                0 <= position[1] < self.env.world.height and
                self.env.world.is_valid_position(position, self.agent_type, self.agent_size, 0))

    def is_goal_reached(self, current, goal):
        return np.linalg.norm(np.array(current.position) - np.array(goal.position)) < self.resolution

    def set_goal(self, goal):
        self.goal = goal