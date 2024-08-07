import numpy as np
from .base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)

class Satellite(BaseAgent):
    def __init__(self, agent_id, position, env):
        super().__init__(agent_id, position, env)
        self.type = "satellite"
        self.orbit_radius = 1000.0  # Large value to simulate high altitude
        self.orbit_speed = 0.001  # Slow orbit speed
        self.orbit_angle = 0
        self.global_communication_range = 500.0
        self.gps_accuracy = 1.0  # GPS accuracy in meters
        self.earth_observation_radius = self.orbit_radius / 2
        self.max_speed = 7.5  # Orbital velocity
        self.energy_consumption_rate = 0.05  # Low energy consumption in space
        self.task_types = ["communication", "observation", "gps"]

    def act(self, state):
        # Satellites don't need to make decisions, they just orbit
        return np.zeros(6)

    def update(self, action):
        self.orbit()
        self.relay_communications()
        self.provide_gps_data()
        self.consume_energy(0.1)

    def orbit(self):
        self.orbit_angle += self.orbit_speed
        if self.orbit_angle >= 2 * np.pi:
            self.orbit_angle -= 2 * np.pi

        self.position[0] = self.env.world.width / 2 + self.orbit_radius * np.cos(self.orbit_angle)
        self.position[1] = self.env.world.height / 2 + self.orbit_radius * np.sin(self.orbit_angle)
        self.position[2] = self.orbit_radius

    def relay_communications(self):
        messages = self.env.communication_model.get_undelivered_messages()
        for message in messages:
            if self.can_relay(message):
                self.env.communication_model.deliver_message(message)

    def can_relay(self, message):
        sender_distance = np.linalg.norm(message.sender.position - self.position)
        receiver_distance = np.linalg.norm(message.receiver.position - self.position)
        return (sender_distance <= self.global_communication_range and
                receiver_distance <= self.global_communication_range)

    def provide_gps_data(self):
        for agent in self.env.agents:
            if agent.type != "satellite":
                true_position = agent.position
                error = np.random.normal(0, self.gps_accuracy, 3)
                gps_position = true_position + error
                agent.update_gps_position(gps_position)

    def get_earth_observation_data(self):
        return self.env.get_objects_in_range(self.position, self.earth_observation_radius, "space")

    def get_state(self):
        base_state = super().get_state()
        satellite_state = np.array([self.orbit_angle / (2 * np.pi)])
        return np.concatenate([base_state, satellite_state])

    def get_state_dim(self):
        return super().get_state_dim() + 1  # Add orbit angle to state

    def get_action_dim(self):
        return 6  # Consistent with BaseAgent, though satellites don't use actions

    def get_agent_type(self):
        return "space"

    def is_valid_position(self, position):
        # Satellites are always in a valid position as long as they're in their orbit
        distance_from_center = np.linalg.norm(
            position[:2] - np.array([self.env.world.width / 2, self.env.world.height / 2]))
        return np.isclose(distance_from_center, self.orbit_radius, atol=1.0)

    def get_reward(self):
        reward = super().get_reward()

        # Additional reward for successful communication relays
        num_relayed_messages = len(
            [msg for msg in self.env.communication_model.get_undelivered_messages() if self.can_relay(msg)])
        reward += num_relayed_messages * 0.1

        # Reward for providing accurate GPS data
        gps_accuracy_reward = 1.0 / (1.0 + self.gps_accuracy)  # Higher accuracy (lower value) gives higher reward
        reward += gps_accuracy_reward

        # Reward for earth observation coverage
        observed_objects = self.get_earth_observation_data()
        reward += len(observed_objects) * 0.05

        return reward

    def handle_collision(self):
        # Satellites shouldn't collide with anything, but if they do, it's catastrophic
        logger.error(f"Satellite {self.id} has collided! This should not happen.")
        self.is_functioning = False
        self.energy = 0