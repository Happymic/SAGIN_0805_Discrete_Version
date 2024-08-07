import numpy as np
from .base_agent import BaseAgent

class SignalDetector(BaseAgent):
    def __init__(self, agent_id, position, env):
        super().__init__(agent_id, position, env)
        self.type = "signal_detector"
        self.detection_range = 20.0
        self.max_speed = 3.0  # Slower than other ground agents
        self.energy_consumption_rate = 0.05  # Lower energy consumption
        self.task_types = ["monitor"]

    def act(self, state):
        signals = self.detect_signals()
        if signals:
            return self.move_towards_strongest_signal(signals)
        return self.random_movement()

    def detect_signals(self):
        return self.env.get_objects_in_range(self.position, self.detection_range, "ground")

    def move_towards_strongest_signal(self, signals):
        strongest_signal = max(signals, key=lambda s: s.strength if hasattr(s, 'strength') else 0)
        direction = strongest_signal['position'] - self.position if isinstance(strongest_signal, dict) else strongest_signal.position - self.position
        direction = self.apply_obstacle_avoidance(direction)
        return np.concatenate([direction[:2] * self.max_acceleration, [0, 0, 0, 0]])

    def random_movement(self):
        direction = np.random.uniform(-1, 1, 2)
        direction = self.apply_obstacle_avoidance(direction)
        return np.concatenate([direction * self.max_acceleration, [0, 0, 0, 0]])

    def apply_obstacle_avoidance(self, direction):
        for obstacle in self.env.world.obstacles:
            if obstacle['type'] == 'rectangle':
                obstacle_center = np.array(obstacle['center'])
                obstacle_vector = obstacle_center - self.position[:2]
                distance = np.linalg.norm(obstacle_vector)
                if distance < obstacle['width'] + 5:  # 如果靠近障碍物
                    avoidance_vector = self.position[:2] - obstacle_center
                    avoidance_vector = avoidance_vector / np.linalg.norm(avoidance_vector)
                    direction += avoidance_vector * 10  # 添加避障向量
        return direction / np.linalg.norm(direction)

    def get_agent_type(self):
        return "ground"

class TransportVehicle(BaseAgent):
    def __init__(self, agent_id, position, env):
        super().__init__(agent_id, position, env)
        self.type = "transport_vehicle"
        self.cargo_capacity = 100.0
        self.current_cargo = 0.0
        self.max_speed = 5.0  # Moderate speed
        self.task_types = ["transport"]

    def act(self, state):
        if self.current_task:
            return self.move_towards_task()
        return self.random_movement()

    def move_towards_task(self):
        if self.current_task.is_completed():
            self.complete_task()
            return self.random_movement()

        target = self.current_task.get_current_target()
        direction = target - self.position
        direction = self.apply_obstacle_avoidance(direction)
        return np.concatenate([direction[:2] * self.max_acceleration, [0, 0, 0, 0]])

    def random_movement(self):
        direction = np.random.uniform(-1, 1, 2)
        direction = self.apply_obstacle_avoidance(direction)
        return np.concatenate([direction * self.max_acceleration, [0, 0, 0, 0]])

    def apply_obstacle_avoidance(self, direction):
        for obstacle in self.env.world.obstacles:
            if obstacle['type'] == 'rectangle':
                obstacle_center = np.array(obstacle['center'])
                obstacle_vector = obstacle_center - self.position[:2]
                distance = np.linalg.norm(obstacle_vector)
                if distance < obstacle['width'] + 5:  # 如果靠近障碍物
                    avoidance_vector = self.position[:2] - obstacle_center
                    avoidance_vector = avoidance_vector / np.linalg.norm(avoidance_vector)
                    direction[:2] += avoidance_vector * 10  # 添加避障向量
        return direction / np.linalg.norm(direction[:2])

    def load_cargo(self, amount):
        available_space = self.cargo_capacity - self.current_cargo
        loaded_amount = min(amount, available_space)
        self.current_cargo += loaded_amount
        return loaded_amount

    def unload_cargo(self, amount):
        unloaded_amount = min(amount, self.current_cargo)
        self.current_cargo -= unloaded_amount
        return unloaded_amount

    def get_agent_type(self):
        return "ground"

class RescueVehicle(BaseAgent):
    def __init__(self, agent_id, position, env):
        super().__init__(agent_id, position, env)
        self.type = "rescue_vehicle"
        self.rescue_capacity = 5
        self.current_rescued = 0
        self.max_speed = 6.0  # Faster than other ground agents
        self.energy_consumption_rate = 0.15  # Higher energy consumption due to equipment
        self.task_types = ["rescue"]

    def act(self, state):
        if self.current_task:
            return self.move_towards_task()
        return self.move_towards_nearest_disaster()

    def move_towards_task(self):
        if self.current_task.is_completed():
            self.complete_task()
            return self.move_towards_nearest_disaster()

        target = self.current_task.get_current_target()
        direction = target - self.position
        direction = self.apply_obstacle_avoidance(direction)
        return np.concatenate([direction[:2] * self.max_acceleration, [0, 0, 0, 0]])

    def move_towards_nearest_disaster(self):
        disaster_areas = self.env.get_disaster_areas()
        if not disaster_areas:
            return self.random_movement()

        nearest_disaster = min(disaster_areas, key=lambda d: np.linalg.norm(d['position'] - self.position))
        direction = nearest_disaster['position'] - self.position
        direction = self.apply_obstacle_avoidance(direction)
        return np.concatenate([direction[:2] * self.max_acceleration, [0, 0, 0, 0]])

    def random_movement(self):
        direction = np.random.uniform(-1, 1, 2)
        direction = self.apply_obstacle_avoidance(direction)
        return np.concatenate([direction * self.max_acceleration, [0, 0, 0, 0]])

    def apply_obstacle_avoidance(self, direction):
        for obstacle in self.env.world.obstacles:
            if obstacle['type'] == 'rectangle':
                obstacle_center = np.array(obstacle['center'])
                obstacle_vector = obstacle_center - self.position[:2]
                distance = np.linalg.norm(obstacle_vector)
                if distance < obstacle['width'] + 5:  # 如果靠近障碍物
                    avoidance_vector = self.position[:2] - obstacle_center
                    avoidance_vector = avoidance_vector / np.linalg.norm(avoidance_vector)
                    direction[:2] += avoidance_vector * 10  # 添加避障向量
        return direction / np.linalg.norm(direction[:2])

    def rescue(self):
        if self.current_rescued < self.rescue_capacity:
            self.current_rescued += 1
            return True
        return False

    def drop_off_rescued(self):
        dropped_off = self.current_rescued
        self.current_rescued = 0
        return dropped_off

    def get_agent_type(self):
        return "ground"

    def get_state(self):
        base_state = super().get_state()
        rescue_state = np.array([self.current_rescued / self.rescue_capacity])
        return np.concatenate([base_state, rescue_state])

    def get_state_dim(self):
        return super().get_state_dim() + 1  # Additional state for current_rescued