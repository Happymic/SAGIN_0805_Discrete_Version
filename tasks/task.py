import numpy as np

class Task:
    def __init__(self, task_id, task_type, stages, priority, creation_time, deadline, env=None, data_size=0,
                 required_computation=0):
        self.id = task_id
        self.type = task_type
        self.stages = stages
        self.current_stage = 0
        self.priority = priority
        self.creation_time = creation_time
        self.deadline = deadline
        self.env = env  # 添加这行
        self.progress = 0
        self.status = "pending"
        self.assigned_to = None
        self.data_size = data_size
        self.required_computation = required_computation
        self.dynamic_priority = priority
    def update_progress(self, amount):
        self.progress += amount
        if self.progress >= 100:
            self.complete()
        self.update_dynamic_priority()

    def complete(self):
        self.progress = 100
        self.status = "completed"

    def fail(self):
        self.status = "failed"

    def assign(self, agent):
        self.assigned_to = agent
        self.status = "in_progress"

    def is_completed(self):
        return self.status == "completed"

    def is_failed(self):
        return self.status == "failed"

    def is_active(self):
        return self.status in ["pending", "in_progress"]

    def get_current_target(self):
        if self.type in ["transport", "rescue"]:
            return self.stages[self.current_stage]
        elif self.type == "monitor":
            center, radius = self.stages
            angle = (self.env.time * 0.1) % (2 * np.pi)
            return center + radius * np.array([np.cos(angle), np.sin(angle)])
        else:  # compute
            return self.stages[0]
    def get_completion_percentage(self):
        return self.progress

    def get_remaining_time(self, current_time):
        return max(0, self.deadline - current_time)

    def is_expired(self, current_time):
        return current_time > self.deadline

    def update_dynamic_priority(self):
        time_factor = (self.deadline - self.env.time) / (self.deadline - self.creation_time)
        self.dynamic_priority = self.priority * (2 - time_factor)

    def __str__(self):
        return f"Task {self.id}: {self.type} (Priority: {self.priority}, Status: {self.status}, Progress: {self.progress}%)"