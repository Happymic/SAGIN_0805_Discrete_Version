import numpy as np
from shapely.geometry import Point

class DynamicObstacle:
    def __init__(self, position, velocity, size, env_width, env_height):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.size = size
        self.env_width = env_width
        self.env_height = env_height

    def update(self, time_step):
        self.position += self.velocity * time_step

        # Bounce off the edges of the environment
        if self.position[0] - self.size < 0 or self.position[0] + self.size > self.env_width:
            self.velocity[0] *= -1
        if self.position[1] - self.size < 0 or self.position[1] + self.size > self.env_height:
            self.velocity[1] *= -1

        self.position = np.clip(self.position, [self.size, self.size],
                                [self.env_width - self.size, self.env_height - self.size])

    def is_colliding(self, agent_position):
        return np.linalg.norm(self.position - agent_position) <= self.size

    def to_shapely(self):
        return Point(self.position).buffer(self.size)

    def get_state(self):
        return np.concatenate([self.position, self.velocity, [self.size]])

    def get_state_dim(self):
        return self.get_state().shape[0]