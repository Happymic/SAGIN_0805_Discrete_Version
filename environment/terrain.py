import numpy as np
from scipy.ndimage import gaussian_filter


class Terrain:
    def __init__(self, width, height, resolution=1.0):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.elevation = np.zeros((self.grid_width, self.grid_height))
        self.terrain_types = ["flat", "hilly", "mountainous", "water"]
        self.terrain_map = np.zeros((self.grid_width, self.grid_height), dtype=int)
        self.generate_terrain()

    def generate_terrain(self):
        # Generate a random terrain using Perlin noise or similar method
        self.elevation = np.random.rand(self.grid_width, self.grid_height)
        self.elevation = gaussian_filter(self.elevation, sigma=5)
        self.elevation = (self.elevation - self.elevation.min()) / (self.elevation.max() - self.elevation.min())

        # Generate terrain types
        self.terrain_map = np.zeros((self.grid_width, self.grid_height), dtype=int)
        self.terrain_map[self.elevation < 0.2] = self.terrain_types.index("water")
        self.terrain_map[(self.elevation >= 0.2) & (self.elevation < 0.5)] = self.terrain_types.index("flat")
        self.terrain_map[(self.elevation >= 0.5) & (self.elevation < 0.8)] = self.terrain_types.index("hilly")
        self.terrain_map[self.elevation >= 0.8] = self.terrain_types.index("mountainous")

    def get_elevation(self, position):
        x, y = int(position[0] / self.resolution), int(position[1] / self.resolution)
        x = np.clip(x, 0, self.grid_width - 1)
        y = np.clip(y, 0, self.grid_height - 1)
        return self.elevation[x, y]

    def get_terrain_type(self, position):
        x, y = int(position[0] / self.resolution), int(position[1] / self.resolution)
        x = np.clip(x, 0, self.grid_width - 1)
        y = np.clip(y, 0, self.grid_height - 1)
        return self.terrain_types[self.terrain_map[x, y]]

    def get_slope(self, position):
        x, y = int(position[0] / self.resolution), int(position[1] / self.resolution)
        x = np.clip(x, 0, self.grid_width - 2)
        y = np.clip(y, 0, self.grid_height - 2)
        dx = self.elevation[x + 1, y] - self.elevation[x, y]
        dy = self.elevation[x, y + 1] - self.elevation[x, y]
        return np.array([dx, dy]) / self.resolution
    def is_traversable(self, position, agent_type):
        terrain_type = self.get_terrain_type(position)
        slope = np.linalg.norm(self.get_slope(position))

        if agent_type == "ground":
            if terrain_type == "water":
                return False
            max_slope = {"flat": 0.3, "hilly": 0.5, "mountainous": 0.7}
            return slope <= max_slope.get(terrain_type, 0.3)
        elif agent_type in ["uav", "satellite"]:
            return True
        else:
            return False

    def get_movement_cost(self, position, agent_type):
        terrain_type = self.get_terrain_type(position)
        slope = np.linalg.norm(self.get_slope(position))

        if agent_type == "ground":
            base_cost = {"flat": 1.0, "hilly": 1.5, "mountainous": 2.0, "water": 5.0}
            return base_cost[terrain_type] * (1 + slope)
        elif agent_type in ["uav", "satellite"]:
            return 1.0
        else:
            return float('inf')

    def reset(self):
        self.generate_terrain()

    def update(self):
        # Terrain doesn't change in this implementation, but you could add dynamic terrain features here
        pass

    def get_state(self):
        # Return a downsampled version of the elevation map and terrain type map as the state
        downsampled_elevation = self.elevation[::10, ::10].flatten()
        downsampled_terrain = self.terrain_map[::10, ::10].flatten()
        return np.concatenate([downsampled_elevation, downsampled_terrain])

    def get_state_dim(self):
        return len(self.get_state())