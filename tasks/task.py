import numpy as np

class Task:
    def __init__(self, task_id, task_type, start_position, end_position, priority, creation_time, deadline):
        self.id = task_id
        self.type = task_type
        self.start_position = np.array(start_position)
        self.end_position = np.array(end_position)
        self.priority = priority
        self.creation_time = creation_time
        self.deadline = deadline
        self.progress = 0
        self.status = "pending"  # pending, in_progress, completed, failed
        self.assigned_to = None

    def update_progress(self, amount):
        self.progress += amount
        if self.progress >= 100:
            self.complete()

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
        if self.progress < 50:
            return self.start_position
        else:
            return self.end_position

    def get_completion_percentage(self):
        return self.progress

    def get_remaining_time(self, current_time):
        return max(0, self.deadline - current_time)

    def is_expired(self, current_time):
        return current_time > self.deadline

    def get_priority_score(self, current_time):
        time_factor = max(0, 1 - (current_time - self.creation_time) / (self.deadline - self.creation_time))
        return self.priority * time_factor

    def __str__(self):
        return f"Task {self.id}: {self.type} (Priority: {self.priority}, Status: {self.status}, Progress: {self.progress}%)"