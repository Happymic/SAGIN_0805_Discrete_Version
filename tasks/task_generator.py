import numpy as np
from .task import Task

class TaskGenerator:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.task_types = ["transport", "rescue", "monitor", "compute"]
        self.task_counter = 0

    def generate_task(self):
        task_type = np.random.choice(self.task_types)
        stages = self.generate_stages(task_type)
        priority = np.random.randint(1, 6)  # Priority from 1 to 5
        creation_time = self.env.time
        deadline = creation_time + np.random.uniform(self.config['min_task_duration'], self.config['max_task_duration'])
        data_size = np.random.randint(1, 100) if task_type == "compute" else 0
        required_computation = np.random.randint(1, 50) if task_type == "compute" else 0

        task = Task(
            task_id=self.task_counter,
            task_type=task_type,
            stages=stages,
            priority=priority,
            creation_time=creation_time,
            deadline=deadline,
            env=self.env,  # 这里传入了 env 参数
            data_size=data_size,
            required_computation=required_computation
        )

        self.task_counter += 1
        return task

    def generate_stages(self, task_type):
        if task_type in ["transport", "rescue"]:
            start = self.env.world.get_random_valid_position()
            end = self.env.world.get_random_valid_position()
            return [start, end]
        elif task_type == "monitor":
            center = self.env.world.get_random_valid_position()
            radius = np.random.uniform(10, 50)
            return [center, radius]
        else:  # compute
            return [self.env.world.get_random_valid_position()]

    def update(self):
        if np.random.random() < self.config['task_generation_probability']:
            new_task = self.generate_task()
            self.env.tasks.append(new_task)
            return new_task
        return None