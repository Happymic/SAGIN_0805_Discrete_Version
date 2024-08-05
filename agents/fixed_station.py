from .base_agent import BaseAgent
import numpy as np
class FixedStation(BaseAgent):
    def __init__(self, env, config, agent_id):
        super().__init__(env, config, agent_id)
        self.type = "fixed_station"
        self.communication_range = config['fixed_station_range']
        self.computing_power = config['fixed_station_computing_power']
        self.position = self.env.grid_world.get_random_position()  # 在初始化时设置位置

    def reset(self):
        # Fixed stations don't change position upon reset
        self.battery = 100.0  # 只重置电池，不改变位置

    def move(self, action):
        # Fixed stations don't move
        pass

    def act(self, state):
        # Fixed stations always perform their action (computation and communication)
        action = np.zeros(5)
        action[4] = 1
        return action

    def update(self, state, action, reward, next_state, done):
        # Fixed stations don't learn, so no update needed
        pass

    def provide_computation(self, task):
        return self.env.process_task(self, task)