import numpy as np

class TaskAllocator:
    def __init__(self, env):
        self.env = env

    def allocate_tasks(self):
        unassigned_tasks = [task for task in self.env.tasks if task.status == "pending"]
        available_agents = [agent for agent in self.env.agents if agent.current_task is None]

        for task in unassigned_tasks:
            suitable_agents = [agent for agent in available_agents if self.is_agent_suitable(agent, task)]
            if suitable_agents:
                best_agent = self.select_best_agent(suitable_agents, task)
                self.assign_task(task, best_agent)
                available_agents.remove(best_agent)

    def is_agent_suitable(self, agent, task):
        return task.type in agent.task_types

    def select_best_agent(self, agents, task):
        scores = []
        for agent in agents:
            distance = np.linalg.norm(agent.position - task.start_position)
            score = (1 / distance) * agent.battery * self.get_agent_efficiency(agent, task)
            scores.append(score)
        return agents[np.argmax(scores)]

    def get_agent_efficiency(self, agent, task):
        if isinstance(agent, (TransportVehicle, RescueVehicle)) and task.type in ["transport", "rescue"]:
            return 1.2
        elif isinstance(agent, UAV) and task.type == "monitor":
            return 1.5
        else:
            return 1.0

    def assign_task(self, task, agent):
        task.assign(agent)
        agent.assign_task(task)
        print(f"Assigned {task} to {agent.type} {agent.id}")

    def update(self):
        self.allocate_tasks()
        self.check_task_timeouts()

    def check_task_timeouts(self):
        current_time = self.env.time
        for task in self.env.tasks:
            if task.is_active() and task.is_expired(current_time):
                if task.assigned_to:
                    task.assigned_to.current_task = None
                task.fail()
                print(f"Task {task.id} has expired and failed.")