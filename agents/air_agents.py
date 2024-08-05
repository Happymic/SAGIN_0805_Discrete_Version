import numpy as np
from .base_agent import BaseAgent


class UAV(BaseAgent):
    def __init__(self, env, config, agent_id):
        super().__init__(env, config, agent_id)
        self.type = "uav"
        self.altitude = 5
        self.max_altitude = config['uav_max_altitude']

    def move(self, action):
        super().move(action)

        # Adjust altitude
        if action == 4:  # Perform action
            self.altitude = np.clip(self.altitude + np.random.randint(-1, 2), 1, self.max_altitude)

        self.battery -= 0.2  # UAVs consume more battery

    def act(self, state):
        action = np.zeros(5)
        # Simple behavior: alternate between moving and performing action
        if self.env.time % 2 == 0:
            action[np.random.randint(0, 4)] = 1  # Random movement
        else:
            action[4] = 1  # Perform action (area monitoring)
        return action

    def update(self, state, action, reward, next_state, done):
        # For simplicity, we're not implementing learning here
        pass

    def relay_signal(self, message, source, target):
        return self.env.relay_message(self, message, source, target)

    def monitor_area(self):
        return self.env.get_area_info(self.position, self.altitude)