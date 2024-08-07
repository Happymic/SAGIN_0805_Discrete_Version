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
            env=self.env,
            data_size=data_size,
            required_computation=required_computation
        )

        self.task_counter += 1
        return task

    def generate_stages(self, task_type):
        if task_type in ["transport", "rescue"]:
            start = self.env.world.get_random_valid_position("ground", 1.0)
            end = self.env.world.get_random_valid_position("ground", 1.0)
            return [start, end]
        elif task_type == "monitor":
            center = self.env.world.get_random_valid_position("ground", 1.0)
            radius = np.random.uniform(10, 50)
            return [center, radius]
        else:  # compute
            return [self.env.world.get_random_valid_position("ground", 1.0)]
    def update(self):
        if np.random.random() < self.config['task_generation_probability']:
            new_task = self.generate_task()
            self.env.task_allocator.add_task(new_task)
            return new_task
        return None

    def generate_disaster(self):
        position = self.env.world.get_random_valid_position()
        radius = np.random.uniform(20, 100)
        duration = np.random.uniform(100, 500)
        intensity = np.random.uniform(0.5, 1.0)

        disaster = {
            'position': position,
            'radius': radius,
            'duration': duration,
            'intensity': intensity,
            'start_time': self.env.time
        }

        self.env.add_disaster_area(disaster)

        # Generate associated tasks
        num_tasks = np.random.randint(3, 10)
        for _ in range(num_tasks):
            task_type = np.random.choice(["rescue", "transport"])
            priority = np.random.randint(3, 6)  # Higher priority for disaster-related tasks
            task = self.generate_disaster_task(disaster, task_type, priority)
            self.env.task_allocator.add_task(task)

    def generate_disaster_task(self, disaster, task_type, priority):
        creation_time = self.env.time
        deadline = creation_time + disaster['duration']

        if task_type == "rescue":
            stages = [disaster['position'], self.env.world.get_random_valid_position()]
        else:  # transport
            stages = [self.env.world.get_random_valid_position(), disaster['position']]

        task = Task(
            task_id=self.task_counter,
            task_type=task_type,
            stages=stages,
            priority=priority,
            creation_time=creation_time,
            deadline=deadline,
            env=self.env
        )

        self.task_counter += 1
        return task