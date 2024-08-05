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

    def act(self, state):
        # Fixed stations don't move
        return np.zeros(2)

    def update(self, action):
        self.process_tasks()
        self.manage_communications()
        self.consume_energy(0.05)  # Lower energy consumption for fixed stations

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
        messages = self.env.get_messages_for(agent)
        for message in messages:
            if self.env.send_message(message, agent):
                self.env.remove_message(message)

    def is_in_communication_range(self, other_agent):
        distance = np.linalg.norm(self.position - other_agent.position)
        return distance <= self.extended_communication_range

    def provide_computation_service(self, task):
        if self.offload_task(task):
            return f"Task {task.id} accepted for computation"
        else:
            return f"Unable to accept task {task.id} due to capacity constraints"