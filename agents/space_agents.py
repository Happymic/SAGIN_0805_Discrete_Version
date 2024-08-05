import numpy as np
from .base_agent import BaseAgent


class Satellite(BaseAgent):
    def __init__(self, agent_id, position, env):
        super().__init__(agent_id, position, env)
        self.type = "satellite"
        self.orbit_radius = 1000.0  # Large value to simulate high altitude
        self.orbit_speed = 0.001  # Slow orbit speed
        self.orbit_angle = 0
        self.global_communication_range = 500.0
        self.gps_accuracy = 1.0  # GPS accuracy in meters

    def act(self, state):
        # Satellites don't need to make decisions, they just orbit
        return np.zeros(2)

    def update(self, action):
        self.orbit()
        self.relay_communications()
        self.provide_gps_data()
        self.consume_energy(0.1)

    def orbit(self):
        self.orbit_angle += self.orbit_speed
        if self.orbit_angle >= 2 * np.pi:
            self.orbit_angle -= 2 * np.pi

        self.position[0] = self.env.width / 2 + self.orbit_radius * np.cos(self.orbit_angle)
        self.position[1] = self.env.height / 2 + self.orbit_radius * np.sin(self.orbit_angle)

    def relay_communications(self):
        messages = self.env.get_undelivered_messages()
        for message in messages:
            if self.can_relay(message):
                self.env.deliver_message(message)

    def can_relay(self, message):
        sender_distance = np.linalg.norm(message.sender.position - self.position)
        receiver_distance = np.linalg.norm(message.receiver.position - self.position)
        return (sender_distance <= self.global_communication_range and
                receiver_distance <= self.global_communication_range)

    def provide_gps_data(self):
        for agent in self.env.agents:
            if agent.type != "satellite":
                true_position = agent.position
                error = np.random.normal(0, self.gps_accuracy, 2)
                gps_position = true_position + error
                agent.update_gps_position(gps_position)

    def get_earth_observation_data(self):
        # Simulate earth observation capabilities
        observation_radius = self.orbit_radius / 2
        return self.env.get_objects_in_range(self.position, observation_radius)