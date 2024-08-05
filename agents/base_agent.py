import numpy as np


class BaseAgent:
    def __init__(self, env, config, agent_id):
        self.env = env
        self.config = config
        self.id = agent_id
        self.position = None
        self.type = None
        self.battery = 100.0

    def reset(self):
        self.position = self.env.grid_world.get_random_position()
        self.battery = 100.0

    def move(self, action):
        if action == 0:  # Stay
            return

        directions = [
            (0, 1),  # Up
            (1, 0),  # Right
            (0, -1),  # Down
            (-1, 0)  # Left
        ]

        dx, dy = directions[action - 1]
        new_position = self.position + np.array([dx, dy])

        if self.env.grid_world.is_valid_position(new_position):
            self.position = new_position
            self.battery -= 0.1  # Decrease battery for movement

    def communicate(self, message, target=None):
        if target:
            return self.env.send_message(self, message, target)
        else:
            return self.env.broadcast_message(self, message)

    def observe(self):
        return self.env.get_observation()[self.env.agents.index(self)]

    def act(self, state):
        raise NotImplementedError

    def update(self, state, action, reward, next_state, done):
        raise NotImplementedError