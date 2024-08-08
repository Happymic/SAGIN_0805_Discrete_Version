import numpy as np
import logging

logger = logging.getLogger(__name__)

class TaskGenerator:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.pois = self.generate_pois()

    def generate_pois(self):
        pois = []
        for _ in range(self.config['num_pois']):
            position = self.env.world.get_random_valid_position("ground", 1.0)
            poi_type = np.random.choice(["monitor", "rescue", "transport"])
            priority = np.random.randint(1, 6)
            pois.append({
                "position": position,
                "type": poi_type,
                "priority": priority,
                "id": f"POI_{len(pois)}",
                "completed": False
            })
        return pois

    def update(self):
        for poi in self.pois:
            if np.random.random() < 0.1:  # 10% 概率更新
                poi["priority"] = np.random.randint(1, 6)
        logger.info(f"Updated POIs: {len(self.pois)}")
        return None  # 保持与原接口兼容

    def get_active_pois(self):
        return [poi for poi in self.pois if not poi["completed"]]

    def reset(self):
        for poi in self.pois:
            poi["completed"] = False
        np.random.shuffle(self.pois)