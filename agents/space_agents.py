import numpy as np
from .base_agent import BaseAgent

class Satellite(BaseAgent):
    def __init__(self, env, config, agent_id):
        super().__init__(env, config, agent_id)
        self.type = "satellite"
        self.orbit_position = 0
        self.orbit_speed = config['satellite_orbit_speed']

    def move(self, action):
        # Satellites move in a predefined orbit
        self.orbit_position = (self.orbit_position + self.orbit_speed) % 360
        self.position = self.calculate_position()
        self.battery -= 0.05  # Satellites consume less battery

    def calculate_position(self):
        # Convert orbit position to grid position
        x = int(self.env.grid_world.width / 2 + (self.env.grid_world.width / 2 - 1) * np.cos(np.radians(self.orbit_position)))
        y = int(self.env.grid_world.height / 2 + (self.env.grid_world.height / 2 - 1) * np.sin(np.radians(self.orbit_position)))
        return np.array([x, y])

    def act(self, state):
        # Satellites always perform their action (global monitoring and task assignment)
        action = np.zeros(5)
        action[4] = 1
        return action

    def update(self, state, action, reward, next_state, done):
        # For simplicity, we're not implementing learning here
        pass

    def global_monitoring(self):
        return self.env.get_global_info()

    def schedule_task(self, task):
        return self.env.assign_task(task)