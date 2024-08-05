import numpy as np
from .base_agent import BaseAgent


class SignalDetector(BaseAgent):
    def __init__(self, agent_id, position, env):
        super().__init__(agent_id, position, env)
        self.type = "signal_detector"
        self.detection_range = 20.0

    def act(self, state):
        # Implement signal detection logic
        signals = self.detect_signals()
        if signals:
            return self.move_towards_strongest_signal(signals)
        return self.random_movement()

    def update(self, action):
        self.acceleration = action
        self.move(self.env.time_step)
        self.consume_energy(0.1)

    def detect_signals(self):
        return self.env.get_signals_in_range(self.position, self.detection_range)

    def move_towards_strongest_signal(self, signals):
        strongest_signal = max(signals, key=lambda s: s.strength)
        direction = strongest_signal.position - self.position
        return direction / np.linalg.norm(direction)

    def random_movement(self):
        return np.random.uniform(-1, 1, 2)


class TransportVehicle(BaseAgent):
    def __init__(self, agent_id, position, env):
        super().__init__(agent_id, position, env)
        self.type = "transport_vehicle"
        self.cargo_capacity = 100.0
        self.current_cargo = 0.0

    def act(self, state):
        if self.current_task:
            return self.move_towards_task()
        return self.random_movement()

    def update(self, action):
        self.acceleration = action
        self.move(self.env.time_step)
        self.consume_energy(0.2)

    def move_towards_task(self):
        if self.current_task.is_completed():
            self.complete_task()
            return self.random_movement()

        target = self.current_task.get_current_target()
        direction = target - self.position
        return direction / np.linalg.norm(direction)

    def random_movement(self):
        return np.random.uniform(-1, 1, 2)

    def load_cargo(self, amount):
        available_space = self.cargo_capacity - self.current_cargo
        loaded_amount = min(amount, available_space)
        self.current_cargo += loaded_amount
        return loaded_amount

    def unload_cargo(self, amount):
        unloaded_amount = min(amount, self.current_cargo)
        self.current_cargo -= unloaded_amount
        return unloaded_amount


class RescueVehicle(BaseAgent):
    def __init__(self, agent_id, position, env):
        super().__init__(agent_id, position, env)
        self.type = "rescue_vehicle"
        self.rescue_capacity = 5
        self.current_rescued = 0

    def act(self, state):
        if self.current_task:
            return self.move_towards_task()
        return self.move_towards_nearest_disaster()

    def update(self, action):
        self.acceleration = action
        self.move(self.env.time_step)
        self.consume_energy(0.3)

    def move_towards_task(self):
        if self.current_task.is_completed():
            self.complete_task()
            return self.move_towards_nearest_disaster()

        target = self.current_task.get_current_target()
        direction = target - self.position
        return direction / np.linalg.norm(direction)

    def move_towards_nearest_disaster(self):
        disaster_areas = self.env.get_disaster_areas()
        if not disaster_areas:
            return self.random_movement()

        nearest_disaster = min(disaster_areas, key=lambda d: np.linalg.norm(d.position - self.position))
        direction = nearest_disaster.position - self.position
        return direction / np.linalg.norm(direction)

    def random_movement(self):
        return np.random.uniform(-1, 1, 2)

    def rescue(self):
        if self.current_rescued < self.rescue_capacity:
            self.current_rescued += 1
            return True
        return False

    def drop_off_rescued(self):
        dropped_off = self.current_rescued
        self.current_rescued = 0
        return dropped_off