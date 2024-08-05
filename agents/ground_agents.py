import numpy as np
from .base_agent import BaseAgent

class SignalDetector(BaseAgent):
    def __init__(self, env, config, agent_id):
        super().__init__(env, config, agent_id)
        self.type = "signal_detector"

    def act(self, state):
        # Simple rule-based action for signal detector
        if self.env.detect_signal(self.position):
            action = np.zeros(5)
            action[4] = 1  # Perform action
        else:
            action = np.zeros(5)
            action[np.random.randint(0, 4)] = 1  # Random movement
        return action

    def update(self, state, action, reward, next_state, done):
        # Signal detector doesn't learn, so no update needed
        pass

class TransportVehicle(BaseAgent):
    def __init__(self, env, config, agent_id):
        super().__init__(env, config, agent_id)
        self.type = "transport_vehicle"
        self.assigned_task = None

    def act(self, state):
        action = np.zeros(5)
        if self.assigned_task:
            # Move towards the task end position
            dx = self.assigned_task["end"][0] - self.position[0]
            dy = self.assigned_task["end"][1] - self.position[1]
            if dx != 0:
                action[1 if dx > 0 else 3] = 1  # Right or Left
            elif dy != 0:
                action[0 if dy > 0 else 2] = 1  # Up or Down
            else:
                action[4] = 1  # Perform action (complete task)
        else:
            action[np.random.randint(0, 4)] = 1  # Random movement
        return action

    def update(self, state, action, reward, next_state, done):
        # For simplicity, we're not implementing learning here
        pass

class RescueVehicle(BaseAgent):
    def __init__(self, env, config, agent_id):
        super().__init__(env, config, agent_id)
        self.type = "rescue_vehicle"
        self.assigned_task = None

    def act(self, state):
        action = np.zeros(5)
        if self.assigned_task:
            # Move towards the task end position
            dx = self.assigned_task["end"][0] - self.position[0]
            dy = self.assigned_task["end"][1] - self.position[1]
            if dx != 0:
                action[1 if dx > 0 else 3] = 1  # Right or Left
            elif dy != 0:
                action[0 if dy > 0 else 2] = 1  # Up or Down
            else:
                action[4] = 1  # Perform action (complete rescue)
        else:
            action[np.random.randint(0, 4)] = 1  # Random movement
        return action

    def update(self, state, action, reward, next_state, done):
        # For simplicity, we're not implementing learning here
        pass