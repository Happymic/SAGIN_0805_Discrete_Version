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
        self.max_speed = 10.0  # Faster than ground vehicles
        self.energy_consumption_rate = 0.2  # Higher energy consumption due to flight

    def act(self, state):
        if self.current_task:
            return self.perform_task()
        return self.explore()

    def update(self, action):
        super().update(action)
        vertical_action = action[2]
        self.altitude += vertical_action * self.vertical_speed * self.env.time_step
        self.altitude = np.clip(self.altitude, self.min_altitude, self.max_altitude)
        self.position[2] = self.altitude
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
        return np.concatenate([horizontal_action * self.max_acceleration, [vertical_action], [0, 0, 0]])

    def rescue_action(self):
        return self.transport_action()

    def monitor_action(self):
        center = self.current_task.get_center()
        radius = self.current_task.get_radius()
        angle = (self.env.time * 0.1) % (2 * np.pi)
        target = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
        return self.transport_action()

    def explore(self):
        horizontal_action = np.random.uniform(-1, 1, 2)
        vertical_action = np.random.uniform(-1, 1)
        return np.concatenate([horizontal_action, [vertical_action], [0, 0, 0]])

    def get_camera_view(self):
        return self.env.get_objects_in_range(self.position, self.camera_range, "air")

    def get_state(self):
        base_state = super().get_state()
        uav_state = np.array([self.altitude / self.max_altitude])
        return np.concatenate([base_state, uav_state])

    def get_state_dim(self):
        return super().get_state_dim() + 1  # Add altitude to state

    def get_action_dim(self):
        return 6  # 3D acceleration (x, y, z) and 3D angular velocity

    def get_agent_type(self):
        return "air"

    def is_valid_position(self, position):
        # Check if the position is within the environment boundaries
        if not (0 <= position[0] < self.env.world.width and
                0 <= position[1] < self.env.world.height and
                self.min_altitude <= position[2] <= self.max_altitude):
            return False

        # Check for collisions with obstacles that are taller than the UAV's altitude
        for obstacle in self.env.world.obstacles:
            if obstacle['height'] > position[2]:
                if obstacle['type'] == 'circle':
                    distance = np.linalg.norm(position[:2] - np.array(obstacle['center']))
                    if distance <= obstacle['radius'] + self.size:
                        return False
                elif obstacle['type'] == 'polygon':
                    # This is a simplified check. For more accuracy, you might want to implement
                    # a proper point-in-polygon check.
                    polygon = np.array(obstacle['points'])
                    if np.all(np.min(polygon, axis=0) <= position[:2]) and np.all(
                            position[:2] <= np.max(polygon, axis=0)):
                        return False

        return True