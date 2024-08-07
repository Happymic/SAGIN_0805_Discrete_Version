import numpy as np
from abc import ABC, abstractmethod
from sensors.sensor_models import Camera, Lidar, Radar, GPS
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, agent_id, position, env):
        self.id = agent_id
        self.position = np.array(position, dtype=float)
        if self.position.shape != (3,):
            self.position = np.pad(self.position, (0, 3 - len(self.position)), 'constant')
        self.env = env
        self.velocity = np.zeros(3)  # 3D velocity vector
        self.acceleration = np.zeros(3)  # 3D acceleration vector
        self.orientation = np.array([1, 0, 0])  # Initial orientation (facing positive x-direction)
        self.angular_velocity = np.zeros(3)  # Angular velocity around x, y, z axes
        self.size = 1.0  # Default size, can be adjusted for different agent types
        self.altitude = 0.0  # Default altitude for ground agents
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
        self.collision_cooldown = 0

    @abstractmethod
    def act(self, state):
        pass

    def update(self, action):
        if not self.is_functioning:
            return

        self.acceleration = np.clip(action[:3], -self.max_acceleration, self.max_acceleration)
        self.angular_velocity = np.clip(action[3:6], -np.pi, np.pi)

        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1
            self.velocity *= 0.5  # 减速

        self.move(self.env.time_step)
        self.rotate(self.env.time_step)
        self.consume_energy(np.linalg.norm(self.velocity) * self.env.time_step * 0.1)
        self.sense()
        if self.current_task:
            self.update_task()
        self.check_failure()

    def move(self, delta_time):
        new_position = self.position + self.velocity * delta_time
        if self.env.is_valid_position(new_position, self.get_agent_type(), self.size, self.altitude):
            self.position = new_position
        else:
            # 碰撞响应
            self.velocity *= -0.5  # 反弹，减少速度
            self.position += self.velocity * delta_time  # 微小移动
            self.handle_collision()
    def handle_collision(self):
        self.velocity = -0.5 * self.velocity  # 反弹
        self.collision_cooldown = 10  # 设置冷却时间
        self.energy -= 1  # 碰撞损失能量
    def rotate(self, delta_time):
        rotation = self.angular_velocity * delta_time
        rotation_matrix = self.euler_to_rotation_matrix(rotation)
        self.orientation = np.dot(rotation_matrix, self.orientation)
        self.orientation /= np.linalg.norm(self.orientation)  # Normalize to ensure unit vector

    @staticmethod
    def euler_to_rotation_matrix(euler_angles):
        # Convert euler angles to rotation matrix
        cx, cy, cz = np.cos(euler_angles)
        sx, sy, sz = np.sin(euler_angles)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        return np.dot(Rz, np.dot(Ry, Rx))

    def consume_energy(self, amount):
        self.energy = max(0, self.energy - amount)
        if self.energy <= 0:
            self.is_functioning = False

    def charge(self, amount):
        self.energy = min(self.max_energy, self.energy + amount)

    def update_gps_position(self, gps_position):
        self.gps_position = gps_position
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
        self.path = self.env.plan_path(self.position, goal, method, self.get_agent_type(), self.size)
    def follow_path(self):
        if self.path and len(self.path) > 1:
            next_point = self.path[1]
            direction = next_point - self.position
            if np.linalg.norm(direction) < self.env.config['agent_step_size']:
                self.path.pop(0)
            return direction / np.linalg.norm(direction)
        return np.zeros(3)

    def get_state(self):
        return np.concatenate([
            self.position,
            self.velocity,
            self.orientation,
            self.angular_velocity,
            [self.altitude],
            [self.energy / self.max_energy],
            [1 if self.current_task else 0],
            [1 if self.is_functioning else 0]
        ])

    def get_state_dim(self):
        return self.get_state().shape[0]

    def get_action_dim(self):
        return 6  # 3 for linear acceleration, 3 for angular velocity

    def get_reward(self):
        reward = 0

        # Task completion reward
        if self.current_task and self.current_task.is_completed():
            reward += 10 * self.current_task.priority

        # Energy consumption penalty
        energy_consumption = self.max_energy - self.energy
        reward -= energy_consumption * 0.1

        # Proximity to task target reward
        if self.current_task:
            distance_to_target = np.linalg.norm(self.position - self.current_task.get_current_target())
            reward += 1 / (1 + distance_to_target)

        # Successful communication reward
        if hasattr(self, 'last_communication_success') and self.last_communication_success:
            reward += 1

        # Collision avoidance reward
        if not self.env.check_collision(self):
            reward += 0.1

        # Global performance metric
        global_task_completion_rate = self.env.get_global_task_completion_rate()
        reward += global_task_completion_rate * 5

        # Log detailed reward breakdown
        logger.debug(
            f"Agent {self.id} reward breakdown: task completion: {10 * self.current_task.priority if self.current_task and self.current_task.is_completed() else 0}, "
            f"energy consumption: {-energy_consumption * 0.1}, distance to target: {1 / (1 + distance_to_target) if self.current_task else 0}, "
            f"communication: {1 if hasattr(self, 'last_communication_success') and self.last_communication_success else 0}, "
            f"collision avoidance: {0.1 if not self.env.check_collision(self) else 0}, "
            f"global performance: {global_task_completion_rate * 5}")

        return reward

    def is_done(self):
        return not self.is_functioning or self.energy <= 0

    def reset(self):
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.orientation = np.array([1, 0, 0])
        self.angular_velocity = np.zeros(3)
        self.energy = self.max_energy
        self.current_task = None
        self.path = None
        self.is_functioning = True
        self.position = self.env.world.get_random_valid_position(self.get_agent_type(), self.size)
    def handle_collision(self):
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.energy -= 5

    def check_failure(self):
        if np.random.random() < self.failure_probability:
            self.is_functioning = False

    def repair(self):
        if not self.is_functioning:
            if np.random.random() < 0.1:  # 10% chance of self-repair
                self.is_functioning = True
                self.energy = 0.5 * self.max_energy  # Partial energy restore after repair

    @abstractmethod
    def get_agent_type(self):
        pass