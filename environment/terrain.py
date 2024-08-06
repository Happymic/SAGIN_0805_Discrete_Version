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
        self.generate_terrain()
    def generate_terrain(self):
        # Generate a random terrain using Perlin noise or similar method
        self.elevation = np.random.rand(self.grid_width, self.grid_height)
        self.elevation = gaussian_filter(self.elevation, sigma=5)
        self.elevation = (self.elevation - self.elevation.min()) / (self.elevation.max() - self.elevation.min())

    def get_elevation(self, position):
        x, y = int(position[0] / self.resolution), int(position[1] / self.resolution)
        return self.elevation[x, y]

    def get_slope(self, position):
        x, y = int(position[0] / self.resolution), int(position[1] / self.resolution)
        x = np.clip(x, 0, self.grid_width - 1)
        y = np.clip(y, 0, self.grid_height - 1)
        dx = self.elevation[min(x+1, self.grid_width-1), y] - self.elevation[max(x-1, 0), y]
        dy = self.elevation[x, min(y+1, self.grid_height-1)] - self.elevation[x, max(y-1, 0)]
        return np.array([dx, dy]) / (2 * self.resolution)
    def is_traversable(self, position, agent_type):
        slope = np.linalg.norm(self.get_slope(position))
        max_slope = {
            "ground": 0.5,
            "uav": float('inf'),
            "satellite": float('inf')
        }
        return slope <= max_slope.get(agent_type, 0.3)

    def reset(self):
        self.generate_terrain()

    def update(self):
        # Terrain doesn't change in this implementation, but you could add dynamic terrain features here
        pass

    def get_state(self):
        # Return a downsampled version of the elevation map as the state
        downsampled = self.elevation[::10, ::10].flatten()
        return downsampled

    def get_state_dim(self):
        return len(self.get_state())