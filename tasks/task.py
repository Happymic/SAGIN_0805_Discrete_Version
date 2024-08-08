import numpy as np
import uuid

class Task:
    def __init__(self, task_id, task_type, stages, priority, creation_time, deadline, env, data_size=0, required_computation=0):
        self.id = task_id
        self.type = task_type
        self.stages = stages
        self.current_stage = 0
        self.priority = priority
        self.creation_time = creation_time
        self.deadline = deadline
        self.env = env
        self.progress = 0
        self.status = "pending"
        self.assigned_to = None
        self.data_size = data_size
        self.required_computation = required_computation
        self.dynamic_priority = priority
        self.completion_rate = 1  # Units of progress per time step

    def update(self, time_step):
        self.progress += time_step * self.completion_rate
        if self.progress >= 100:
            self.complete()
        elif self.env.time > self.deadline:
            self.fail()
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
            return np.array(self.stages[self.current_stage], dtype=float)
        elif self.type == "monitor":
            center, radius = self.stages
            angle = (self.env.time * 0.1) % (2 * np.pi)
            return np.array([
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                center[2] if len(center) > 2 else 0
            ], dtype=float)
        else:  # compute
            return np.array(self.stages[0], dtype=float)

    def update_dynamic_priority(self):
        if self.env:
            time_factor = (self.deadline - self.env.time) / (self.deadline - self.creation_time)
            self.dynamic_priority = self.priority * (2 - time_factor)

    def __str__(self):
        return f"Task {self.id}: {self.type} (Priority: {self.priority}, Status: {self.status}, Progress: {self.progress}%)"

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'stages': self.stages,
            'current_stage': self.current_stage,
            'priority': self.priority,
            'creation_time': self.creation_time,
            'deadline': self.deadline,
            'progress': self.progress,
            'status': self.status,
            'assigned_to': self.assigned_to.id if self.assigned_to else None,
            'data_size': self.data_size,
            'required_computation': self.required_computation,
            'dynamic_priority': self.dynamic_priority
        }

    @classmethod
    def from_dict(cls, data, env):
        task = cls(
            task_id=data['id'],
            task_type=data['type'],
            stages=data['stages'],
            priority=data['priority'],
            creation_time=data['creation_time'],
            deadline=data['deadline'],
            env=env,
            data_size=data['data_size'],
            required_computation=data['required_computation']
        )
        task.current_stage = data['current_stage']
        task.progress = data['progress']
        task.status = data['status']
        task.assigned_to = env.get_agent_by_id(data['assigned_to']) if data['assigned_to'] else None
        task.dynamic_priority = data['dynamic_priority']
        return task