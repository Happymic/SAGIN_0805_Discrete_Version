import numpy as np
from abc import ABC, abstractmethod
from sensors.sensor_models import Camera, Lidar, Radar, GPS
import logging

logger = logging.getLogger(__name__)
class BaseAgent(ABC):

    def __init__(self, agent_id, position, env):
        self.size = 1.0  # 默认大小，可以根据需要调整
        self.id = agent_id
        self.position = np.array(position)
        self.env = env
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.energy = 100.0
        self.max_energy = 100.0
        self.energy_consumption_rate = 0.1
        self.max_speed = 5.0
        self.max_acceleration = 2.0
        self.communication_range = 10.0
        self.sensor_range = 15.0
        self.current_task = None
        self.path = None
        self.failure_probability = 0.001
        self.is_functioning = True

        self.sensors = {
            'camera': Camera(),
            'lidar': Lidar(),
            'radar': Radar(),
            'gps': GPS()
        }

    @abstractmethod
    def act(self, state):
        pass
    def update_gps_position(self, gps_position):
        self.gps_position = gps_position
    def update(self, action):
        if not self.is_functioning:
            return

        self.acceleration = np.clip(action[:2], -self.max_acceleration, self.max_acceleration)
        self.move(self.env.time_step)
        self.consume_energy(np.linalg.norm(self.velocity) * self.env.time_step * 0.1)
        self.sense()
        if self.current_task:
            self.update_task()
        self.check_failure()

    def move(self, delta_time):
        self.velocity += self.acceleration * delta_time
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)
        new_position = self.position + self.velocity * delta_time
        if self.env.terrain.is_traversable(new_position, self.type) and self.env.world.is_valid_position(new_position):
            self.position = new_position
        else:
            self.velocity = np.zeros(2)

    def consume_energy(self, amount):
        self.energy = max(0, self.energy - amount)
        if self.energy <= 0:
            self.is_functioning = False

    def charge(self, amount):
        self.energy = min(self.max_energy, self.energy + amount)

    def sense(self):
        sensor_data = {}
        for name, sensor in self.sensors.items():
            try:
                sensor_data[name] = sensor.sense(self, self.env)
            except Exception as e:
                logger.error(f"Error in sensor {name} for agent {self.id}: {str(e)}")
                sensor_data[name] = None
        return sensor_data
    def communicate(self, receiver, content, priority=1):
        return self.env.send_message(self, receiver, content, priority)

    def broadcast(self, content, range):
        return self.env.broadcast_message(self, content, range)

    def assign_task(self, task):
        self.current_task = task
        self.plan_path(task.get_current_target())

    def complete_task(self):
        self.current_task = None
        self.path = None

    def update_task(self):
        if self.current_task.is_completed():
            self.complete_task()
        elif np.linalg.norm(self.position - self.current_task.get_current_target()) < 1.0:
            self.current_task.update_progress(10)

    def plan_path(self, goal, method='a_star'):
        self.path = self.env.plan_path(self.position, goal, method)

    def follow_path(self):
        if self.path and len(self.path) > 1:
            next_point = self.path[1]
            direction = next_point - self.position
            if np.linalg.norm(direction) < self.env.config['agent_step_size']:
                self.path.pop(0)
            return direction / np.linalg.norm(direction)
        return np.zeros(2)

    def get_state(self):
        return np.concatenate([
            self.position,
            self.velocity,
            [self.energy / self.max_energy],
            [1 if self.current_task else 0],
            [1 if self.is_functioning else 0]
        ])

    def get_state_dim(self):
        return self.get_state().shape[0]
    def get_action_dim(self):
        return 2  # acceleration in x and y directions

    def get_reward(self):
        reward = 0

        # 任务完成奖励
        if self.current_task and self.current_task.is_completed():
            reward += 10 * self.current_task.priority

        # 能源消耗惩罚
        energy_consumption = self.max_energy - self.energy
        reward -= energy_consumption * 0.1

        # 接近任务目标的奖励
        if self.current_task:
            distance_to_target = np.linalg.norm(self.position - self.current_task.get_current_target())
            reward += 1 / (1 + distance_to_target)  # 越接近目标,奖励越高

        # 成功通信奖励
        if hasattr(self, 'last_communication_success') and self.last_communication_success:
            reward += 1

        # 避免碰撞奖励
        if not self.env.check_collision(self):
            reward += 0.1

        # 全局性能指标
        global_task_completion_rate = self.env.get_global_task_completion_rate()
        reward += global_task_completion_rate * 5

        # 记录详细的奖励信息
        logger.debug(f"Agent {self.id} reward breakdown: task completion: {10 * self.current_task.priority if self.current_task and self.current_task.is_completed() else 0}, "
                     f"energy consumption: {-energy_consumption * 0.1}, distance to target: {1 / (1 + distance_to_target) if self.current_task else 0}, "
                     f"communication: {1 if hasattr(self, 'last_communication_success') and self.last_communication_success else 0}, "
                     f"collision avoidance: {0.1 if not self.env.check_collision(self) else 0}, "
                     f"global performance: {global_task_completion_rate * 5}")

        return reward

    def is_done(self):
        return not self.is_functioning or self.energy <= 0

    def reset(self):
        self.position = self.env.world.get_random_valid_position()
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.energy = self.max_energy
        self.current_task = None
        self.path = None
        self.is_functioning = True

    def handle_collision(self):
        self.velocity = np.zeros(2)
        self.energy -= 5
    def check_failure(self):
        if np.random.random() < self.failure_probability:
            self.is_functioning = False

    def repair(self):
        if not self.is_functioning:
            if np.random.random() < 0.1:  # 10% chance of self-repair
                self.is_functioning = True
                self.energy = 0.5 * self.max_energy  # Partial energy restore after repair