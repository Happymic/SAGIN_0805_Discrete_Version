import numpy as np
from typing import List, Dict
import logging
from shapely import Point
logger = logging.getLogger(__name__)

class TaskAllocator:
    def __init__(self, env):
        self.env = env

    def allocate_tasks(self):
        for agent in self.env.agents:
            if agent.energy < agent.max_energy * 0.2:  # If energy is below 20%
                nearest_station = min(self.env.world.charging_stations,
                                      key=lambda s: Point(agent.position[0], agent.position[1]).distance(s))
                agent.assign_task({"type": "charge", "position": np.array([nearest_station.x, nearest_station.y, 0])})
            elif agent.current_task is None:
                suitable_pois = [poi for poi in self.env.task_generator.get_active_pois()
                                 if poi["type"] in agent.task_types]
                if suitable_pois:
                    nearest_poi = min(suitable_pois,
                                      key=lambda p: np.linalg.norm(agent.position - p["position"]))
                    agent.assign_task(nearest_poi)
                    logger.info(f"Assigned POI {nearest_poi['id']} to agent {agent.id}")

    def update(self):
        self.allocate_tasks()

    def get_state(self):
        active_pois = self.env.task_generator.get_active_pois()
        return np.array([
            len(active_pois),
            np.mean([poi['priority'] for poi in active_pois]) if active_pois else 0
        ])

    def get_state_dim(self):
        return 2  # 对应 get_state 中的两个值

    def reset(self):
        # 重置所有代理的任务
        for agent in self.env.agents:
            agent.current_task = None
            agent.last_completed_task = None

    def get_high_priority_tasks(self) -> List[Dict]:
        return [{"id": poi["id"], "priority": poi["priority"]}
                for poi in self.env.task_generator.get_active_pois()
                if poi["priority"] > 3]  # 假设优先级大于3为高优先级