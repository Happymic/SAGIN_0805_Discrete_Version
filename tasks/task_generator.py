import numpy as np
from .task import Task

class TaskGenerator:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.task_types = ["transport", "rescue", "monitor"]
        self.task_counter = 0

    def generate_task(self):
        task_type = np.random.choice(self.task_types)
        start_position = self.env.world.get_random_valid_position()
        end_position = self.env.world.get_random_valid_position()
        priority = np.random.randint(1, 6)  # Priority from 1 to 5
        creation_time = self.env.time
        deadline = creation_time + np.random.uniform(self.config['min_task_duration'], self.config['max_task_duration'])

        task = Task(
            task_id=self.task_counter,
            task_type=task_type,
            start_position=start_position,
            end_position=end_position,
            priority=priority,
            creation_time=creation_time,
            deadline=deadline
        )

        self.task_counter += 1
        return task

    def update(self):
        if np.random.random() < self.config['task_generation_probability']:
            return self.generate_task()
        return None