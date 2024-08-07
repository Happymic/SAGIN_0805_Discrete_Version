import heapq

import numpy as np
from queue import PriorityQueue
import logging

logger = logging.getLogger(__name__)

class Node:
    def __init__(self, position, g_cost, h_cost, parent=None):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost

class AStar:
    def __init__(self, env, resolution=1.0, agent_type="ground", agent_size=1.0):
        self.env = env
        self.resolution = resolution
        self.agent_type = agent_type
        self.agent_size = agent_size
        self.max_iterations = 10000  # 添加最大迭代次数限制

    def plan(self, start, goal):
        self.goal = goal
        start_node = Node(start, 0, self.heuristic(start, goal))
        open_list = []
        heapq.heappush(open_list, start_node)
        closed_set = set()
        iterations = 0

        while open_list and iterations < self.max_iterations:
            current_node = heapq.heappop(open_list)

            if np.allclose(current_node.position, goal, atol=self.resolution):
                return self.reconstruct_path(current_node)

            closed_set.add(tuple(current_node.position))

            for neighbor in self.get_neighbors(current_node):
                if tuple(neighbor.position) in closed_set:
                    continue

                if not self.is_valid(neighbor.position):
                    continue

                existing = next((node for node in open_list if np.allclose(node.position, neighbor.position)), None)
                if existing is None or neighbor.g_cost < existing.g_cost:
                    if existing:
                        open_list.remove(existing)
                    heapq.heappush(open_list, neighbor)

            iterations += 1

        logger.warning(f"A* path planning reached maximum iterations ({self.max_iterations})")
        return None  # No path found

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]

    def get_neighbors(self, node):
        neighbors = []
        for dx in [-self.resolution, 0, self.resolution]:
            for dy in [-self.resolution, 0, self.resolution]:
                if dx == 0 and dy == 0:
                    continue
                new_position = node.position + np.array([dx, dy, 0])
                g_cost = node.g_cost + np.linalg.norm([dx, dy])
                h_cost = self.heuristic(new_position, self.goal)
                neighbors.append(Node(new_position, g_cost, h_cost, node))
        return neighbors

    def heuristic(self, a, b):
        return np.linalg.norm(a - b)

    def is_valid(self, position):
        return (0 <= position[0] < self.env.world.width and
                0 <= position[1] < self.env.world.height and
                self.env.world.is_valid_position(position, self.agent_type, self.agent_size,
                                                 0))  # Assuming ground agents have altitude 0
    def set_goal(self, goal):
        self.goal = goal