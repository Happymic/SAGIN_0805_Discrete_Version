import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, agent_id, position, env):
        self.id = agent_id
        self.position = np.array(position)
        self.env = env
        self.battery = 100.0
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.max_speed = 5.0
        self.max_acceleration = 2.0
        self.communication_range = 10.0
        self.sensor_range = 15.0
        self.current_task = None

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def update(self, action):
        pass

    def move(self, delta_time):
        self.velocity += self.acceleration * delta_time
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)
        new_position = self.position + self.velocity * delta_time
        if self.env.is_valid_position(new_position):
            self.position = new_position
        else:
            self.velocity = np.zeros(2)

    def consume_energy(self, amount):
        self.battery -= amount
        if self.battery < 0:
            self.battery = 0

    def charge(self, amount):
        self.battery = min(100.0, self.battery + amount)

    def communicate(self, message, target=None):
        return self.env.communicate(self, message, target)

    def sense_environment(self):
        return self.env.get_objects_in_range(self.position, self.sensor_range)

    def assign_task(self, task):
        self.current_task = task

    def complete_task(self):
        self.current_task = None

    def is_in_communication_range(self, other_agent):
        distance = np.linalg.norm(self.position - other_agent.position)
        return distance <= self.communication_range