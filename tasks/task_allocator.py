import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class TaskAllocator:
    def __init__(self, env):
        self.env = env
        self.unassigned_tasks = []
        self.assigned_tasks = {}  # agent_id: task
        self.task_queue = []
        self.task_priority_threshold = 0.7  # Threshold for high priority tasks

    def reset(self):
        self.unassigned_tasks.clear()
        self.assigned_tasks.clear()
        self.task_queue.clear()

    def add_task(self, task):
        self.unassigned_tasks.append(task)
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda x: x.priority, reverse=True)

    def allocate_tasks(self):
        for task in self.unassigned_tasks[:]:
            suitable_agents = self.get_suitable_agents(task)
            if suitable_agents:
                best_agent = self.select_best_agent(suitable_agents, task)
                self.assign_task(task, best_agent)
                self.unassigned_tasks.remove(task)

    def get_suitable_agents(self, task):
        return [agent for agent in self.env.agents if self.is_agent_suitable(agent, task)]

    def is_agent_suitable(self, agent, task):
        return (task.type in agent.task_types and
                agent.energy > agent.energy_consumption_rate * self.estimate_task_duration(task))

    def estimate_task_duration(self, task):
        # Simple estimation, can be improved based on task type and complexity
        return (task.deadline - task.creation_time) / 2

    def select_best_agent(self, agents, task):
        scores = []
        for agent in agents:
            distance = np.linalg.norm(agent.position - task.get_current_target())
            score = (1 / distance) * agent.energy * self.get_agent_efficiency(agent, task)
            scores.append(score)
        return agents[np.argmax(scores)]

    def get_agent_efficiency(self, agent, task):
        if agent.type == "transport_vehicle" and task.type == "transport":
            return 1.2
        elif agent.type == "rescue_vehicle" and task.type == "rescue":
            return 1.5
        elif agent.type == "uav" and task.type == "monitor":
            return 1.3
        elif agent.type == "fixed_station" and task.type == "compute":
            return 2.0
        else:
            return 1.0

    def assign_task(self, task, agent):
        if agent.id in self.assigned_tasks:
            old_task = self.assigned_tasks[agent.id]
            self.unassigned_tasks.append(old_task)
            self.task_queue.append(old_task)

        self.assigned_tasks[agent.id] = task
        agent.assign_task(task)
        logger.info(f"Assigned {task} to {agent.type} {agent.id}")

    def update(self):
        self.update_task_priorities()
        self.check_task_completion()
        self.allocate_tasks()

    def update_task_priorities(self):
        current_time = self.env.time
        for task in self.task_queue:
            remaining_time = task.deadline - current_time
            if remaining_time > 0:
                task.dynamic_priority = task.priority * (1 + (task.deadline - current_time) / (task.deadline - task.creation_time))
            else:
                task.dynamic_priority = task.priority * 2  # Double priority for overdue tasks

        self.task_queue.sort(key=lambda x: x.dynamic_priority, reverse=True)

    def check_task_completion(self):
        for agent_id, task in list(self.assigned_tasks.items()):
            if task.is_completed():
                del self.assigned_tasks[agent_id]
                agent = next((a for a in self.env.agents if a.id == agent_id), None)
                if agent:
                    agent.complete_task()
                logger.info(f"Task {task.id} completed by agent {agent_id}")
            elif task.is_failed():
                del self.assigned_tasks[agent_id]
                agent = next((a for a in self.env.agents if a.id == agent_id), None)
                if agent:
                    agent.fail_task()
                logger.info(f"Task {task.id} failed and unassigned from agent {agent_id}")

    def get_high_priority_tasks(self) -> List[Dict]:
        return [{"id": task.id, "priority": task.dynamic_priority}
                for task in self.task_queue if task.dynamic_priority > self.task_priority_threshold]

    def handle_task_deadline(self, task):
        if task in self.unassigned_tasks:
            self.unassigned_tasks.remove(task)
        if task in self.task_queue:
            self.task_queue.remove(task)
        for agent_id, assigned_task in list(self.assigned_tasks.items()):
            if assigned_task == task:
                del self.assigned_tasks[agent_id]
                agent = next((a for a in self.env.agents if a.id == agent_id), None)
                if agent:
                    agent.fail_task()
        logger.info(f"Task {task.id} deadline reached and removed from the system")

    def get_state(self):
        return np.array([
            len(self.unassigned_tasks),
            len(self.assigned_tasks),
            np.mean([task.dynamic_priority for task in self.task_queue]) if self.task_queue else 0,
            len(self.get_high_priority_tasks())
        ])

    def get_state_dim(self):
        return 4  # Corresponds to the four values in get_state()