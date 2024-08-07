import numpy as np

class RRT:
    def __init__(self, env, max_iterations=1000, step_size=1.0):
        self.env = env
        self.max_iterations = max_iterations
        self.step_size = step_size

    def plan(self, start, goal):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.nodes = [self.start]
        self.parents = {tuple(self.start): None}

        for _ in range(self.max_iterations):
            random_point = self.random_state()
            nearest_node = self.nearest_neighbor(random_point)
            new_node = self.steer(nearest_node, random_point)

            if self.obstacle_free(nearest_node, new_node):
                self.nodes.append(new_node)
                self.parents[tuple(new_node)] = tuple(nearest_node)

                if np.linalg.norm(new_node - self.goal) < self.step_size:
                    return self.reconstruct_path(new_node)

        return None  # No path found

    def random_state(self):
        if np.random.random() < 0.1:  # 10% chance to sample the goal
            return self.goal
        return np.array([
            np.random.uniform(0, self.env.world.width),
            np.random.uniform(0, self.env.world.height),
            np.random.uniform(0, self.env.world.height)  # Assuming height is used for z-coordinate
        ])

    def nearest_neighbor(self, point):
        distances = [np.linalg.norm(node - point) for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def steer(self, from_point, to_point):
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        if distance > self.step_size:
            direction = direction / distance * self.step_size
        return from_point + direction

    def obstacle_free(self, from_point, to_point):
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        steps = int(distance / (self.step_size / 2))
        for i in range(steps + 1):
            point = from_point + direction * i / steps
            if not self.env.world.is_valid_position(point) or not self.env.terrain.is_traversable(point, "air"):
                return False
        return True

    def reconstruct_path(self, node):
        path = [node]
        while tuple(node) in self.parents:
            node = np.array(self.parents[tuple(node)])
            path.append(node)
        return path[::-1]

    def set_goal(self, goal):
        self.goal = np.array(goal)