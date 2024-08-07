import numpy as np
from queue import PriorityQueue

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

    def plan(self, start, goal):
        self.goal = goal  # 添加这一行
        start_node = Node(start, 0, self.heuristic(start, goal))
        open_list = PriorityQueue()
        open_list.put(start_node)
        closed_set = set()

        while not open_list.empty():
            current_node = open_list.get()

            if np.allclose(current_node.position, goal, atol=self.resolution):
                return self.reconstruct_path(current_node)

            closed_set.add(tuple(current_node.position))

            for neighbor in self.get_neighbors(current_node):
                if tuple(neighbor.position) in closed_set:
                    continue

                if not self.is_valid(neighbor.position):
                    continue

                if not self.env.terrain.is_traversable(neighbor.position, "ground"):
                    continue

                open_list.put(neighbor)

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