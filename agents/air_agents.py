import numpy as np
from .base_agent import BaseAgent

class UAV(BaseAgent):
    def __init__(self, agent_id, position, env):
        super().__init__(agent_id, position, env)
        self.type = "uav"
        self.altitude = 50.0
        self.max_altitude = 100.0
        self.min_altitude = 10.0
        self.vertical_speed = 5.0
        self.camera_range = 30.0
        self.task_types = ["transport", "rescue", "monitor"]

    def act(self, state):
        if self.current_task:
            return self.perform_task()
        return self.explore()

    def update(self, action):
        super().update(action)
        vertical_action = action[2]
        self.altitude += vertical_action * self.vertical_speed * self.env.time_step
        self.altitude = np.clip(self.altitude, self.min_altitude, self.max_altitude)
        self.consume_energy(0.3 + 0.1 * abs(vertical_action))

    def perform_task(self):
        if self.current_task.type == "transport":
            return self.transport_action()
        elif self.current_task.type == "rescue":
            return self.rescue_action()
        elif self.current_task.type == "monitor":
            return self.monitor_action()
        else:
            return self.explore()

    def transport_action(self):
        target = self.current_task.get_current_target()
        direction = target - self.position
        horizontal_action = direction[:2] / np.linalg.norm(direction[:2])
        vertical_action = np.sign(target[2] - self.altitude)
        return np.append(horizontal_action, vertical_action)

    def rescue_action(self):
        return self.transport_action()

    def monitor_action(self):
        center = self.current_task.get_center()
        radius = self.current_task.get_radius()
        angle = (self.env.current_time * 0.1) % (2 * np.pi)
        target = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
        return self.transport_action()

    def explore(self):
        horizontal_action = np.random.uniform(-1, 1, 2)
        vertical_action = np.random.uniform(-1, 1)
        return np.append(horizontal_action, vertical_action)

    def get_camera_view(self):
        return self.env.get_objects_in_range(self.position, self.camera_range)

    def get_state_dim(self):
        return super().get_state_dim() + 1  # Add altitude to state

    def get_action_dim(self):
        return 3  # horizontal acceleration (2) and vertical speed (1)