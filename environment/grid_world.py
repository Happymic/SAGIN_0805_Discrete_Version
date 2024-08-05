import numpy as np

class GridWorld:
    def __init__(self, config):
        self.width = config['grid_width']
        self.height = config['grid_height']
        self.obstacles = self.generate_obstacles(config['num_obstacles'])
        self.pois = self.generate_pois(config['num_pois'])
        self.disaster_areas = self.generate_disaster_areas(config['num_disaster_areas'])

    def generate_obstacles(self, num_obstacles):
        obstacles = set()
        while len(obstacles) < num_obstacles:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            obstacles.add((x, y))
        return obstacles

    def generate_pois(self, num_pois):
        pois = set()
        while len(pois) < num_pois:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if (x, y) not in self.obstacles:
                pois.add((x, y))
        return pois

    def generate_disaster_areas(self, num_disaster_areas):
        disaster_areas = set()
        while len(disaster_areas) < num_disaster_areas:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if (x, y) not in self.obstacles and (x, y) not in self.pois:
                disaster_areas.add((x, y))
        return disaster_areas

    def is_valid_position(self, position):
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles

    def get_random_position(self):
        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.is_valid_position((x, y)):
                return np.array([x, y])

    def get_neighbors(self, position):
        x, y = position
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position((new_x, new_y)):
                neighbors.append((new_x, new_y))
        return neighbors

    def get_manhattan_distance(self, pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')  # 返回无穷大表示无效距离
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])