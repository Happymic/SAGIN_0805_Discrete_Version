import numpy as np
from .base_agent import BaseAgent
from collections import deque

class FixedStation(BaseAgent):
    def __init__(self, agent_id, position, env):
        super().__init__(agent_id, position, env)
        self.type = "fixed_station"
        self.computation_power = 100  # Computational units per time step
        self.storage_capacity = 1000  # Data storage units
        self.current_storage = 0
        self.task_queue = deque()
        self.extended_communication_range = 50.0  # Larger than regular agents
        self.energy_consumption_rate = 0.05  # Lower energy consumption for fixed stations
        self.max_speed = 0  # Fixed stations don't move
        self.task_types = ["compute", "store", "relay"]

    def act(self, state):
        # Fixed stations don't move
        return np.zeros(6)

    def update(self, action):
        self.process_tasks()
        self.manage_communications()
        self.consume_energy(self.energy_consumption_rate)

    def process_tasks(self):
        completed_tasks = []
        remaining_power = self.computation_power

        for task in self.task_queue:
            if remaining_power >= task.required_computation:
                task.progress += task.required_computation
                remaining_power -= task.required_computation
                if task.is_completed():
                    completed_tasks.append(task)
            else:
                task.progress += remaining_power
                break

        for task in completed_tasks:
            self.task_queue.remove(task)
            self.env.task_completed(task)
            self.current_storage -= task.data_size

    def offload_task(self, task):
        if self.current_storage + task.data_size <= self.storage_capacity:
            self.task_queue.append(task)
            self.current_storage += task.data_size
            return True
        return False

    def manage_communications(self):
        for agent in self.env.agents:
            if agent != self and self.is_in_communication_range(agent):
                self.relay_messages(agent)

    def relay_messages(self, agent):
        messages = self.env.communication_model.get_messages_for(agent)
        for message in messages:
            if self.env.communication_model.send_message(message.sender, agent, message.content):
                self.env.communication_model.remove_message(message)

    def is_in_communication_range(self, other_agent):
        distance = np.linalg.norm(self.position - other_agent.position)
        return distance <= self.extended_communication_range

    def provide_computation_service(self, task):
        if self.offload_task(task):
            return f"Task {task.id} accepted for computation"
        else:
            return f"Unable to accept task {task.id} due to capacity constraints"

    def get_state(self):
        base_state = super().get_state()
        fixed_station_state = np.array([
            self.current_storage / self.storage_capacity,
            len(self.task_queue) / 10,  # Normalize by assuming max 10 tasks in queue
            self.computation_power / 100  # Normalize computation power
        ])
        return np.concatenate([base_state, fixed_station_state])

    def get_state_dim(self):
        return super().get_state_dim() + 3  # Add current_storage, task_queue_size, and computation_power to state

    def get_action_dim(self):
        return 6  # Consistent with BaseAgent, though fixed stations don't use actions

    def get_agent_type(self):
        return "fixed"

    def is_valid_position(self, position):
        # Fixed stations don't move, so their initial position is always valid
        return np.all(np.isclose(position, self.position))

    def get_reward(self):
        reward = super().get_reward()

        # Reward for completing computational tasks
        reward += len([task for task in self.task_queue if task.is_completed()]) * 0.5

        # Reward for efficient use of storage
        storage_efficiency = self.current_storage / self.storage_capacity
        reward += storage_efficiency * 0.3

        # Reward for communication relays
        reward += self.env.communication_model.get_relay_count(self) * 0.1

        return reward

    def handle_collision(self):
        # Fixed stations shouldn't experience collisions, but if they do, it's serious
        self.is_functioning = False
        self.energy = 0

    def reset(self):
        super().reset()
        self.current_storage = 0
        self.task_queue.clear()
        self.is_functioning = True

    def move(self, delta_time):
        # Override move method to prevent any movement
        pass