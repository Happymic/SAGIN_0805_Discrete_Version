import gym
import numpy as np
from gym import spaces
from typing import List, Dict, Any
import logging

from .continuous_world import ContinuousWorld
from .weather_system import WeatherSystem
from .dynamic_obstacles import DynamicObstacle
from .terrain import Terrain
from communication.communication_model import CommunicationModel
from tasks.task_generator import TaskGenerator
from tasks.task_allocator import TaskAllocator
from path_planning.a_star import AStar
from path_planning.rrt import RRT
from agents.base_agent import BaseAgent
from utils.event_scheduler import EventScheduler

logger = logging.getLogger(__name__)

class SAGINEnv(gym.Env):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # Initialize world and environmental components
        self.world = ContinuousWorld(config['world_width'], config['world_height'],
                                     config['num_obstacles'], config['num_pois'])
        self.terrain = Terrain(config['world_width'], config['world_height'])
        self.weather_system = WeatherSystem(config['world_width'], config['world_height'],
                                            config['weather_change_rate'])
        self.dynamic_obstacles = self._create_dynamic_obstacles()

        # Initialize communication and task-related components
        self.communication_model = CommunicationModel(self)
        self.task_generator = TaskGenerator(self, config)
        self.task_allocator = TaskAllocator(self)

        # Initialize path planning algorithms
        self.a_star = AStar(self)
        self.rrt = RRT(self)

        # Initialize lists for agents, tasks, and disaster areas
        self.agents: List[BaseAgent] = []
        self.tasks = []
        self.disaster_areas = []

        # Initialize time-related variables
        self.time = 0
        self.max_time = config['max_time']
        self.time_step = config['time_step']
        self.current_step = 0
        self.max_steps = config['max_steps']
        self.time_of_day = 0  # 0-23 hours

        # Initialize event scheduler
        self.event_scheduler = EventScheduler()

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.get_action_dim(),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.get_state_dim(),), dtype=np.float32)
    def get_global_task_completion_rate(self):
        completed_tasks = sum(1 for task in self.tasks if task.is_completed())
        total_tasks = len(self.tasks)
        return completed_tasks / total_tasks if total_tasks > 0 else 0

    def _create_dynamic_obstacles(self):
        obstacles = [DynamicObstacle(
            position=np.random.uniform([0, 0], [self.config['world_width'], self.config['world_height']]),
            velocity=np.random.uniform(-1, 1, 2),
            size=np.random.uniform(1, 3),
            env_width=self.config['world_width'],
            env_height=self.config['world_height']
        ) for _ in range(self.config['num_dynamic_obstacles'])]
        print(f"Created {len(obstacles)} dynamic obstacles")
        return obstacles
    def reset(self):
        logger.info("Resetting SAGIN environment")
        self.time = 0
        self.current_step = 0
        self.tasks.clear()
        self.weather_system.reset()
        self.terrain.reset()
        self.time_of_day = np.random.randint(0, 24)
        self.dynamic_obstacles = self._create_dynamic_obstacles()
        print(f"Number of dynamic obstacles: {len(self.dynamic_obstacles)}")
        self.event_scheduler.clear()

        for agent in self.agents:
            agent.reset()
            agent.position = self.world.get_random_valid_position()
            logger.debug(f"Reset agent {agent.id}")

        logger.debug(f"Number of agents after reset: {len(self.agents)}")
        return self.get_state()

    def step(self, actions):
        self.current_step += 1
        self.time += self.time_step

        # 更新环境
        self._update_environment()

        # 执行智能体动作
        for agent, action in zip(self.agents, actions):
            agent.update(action)

        # 更新任务
        self.task_generator.update()
        self.task_allocator.update()

        # 获取新的状态
        new_state = self.get_state()

        # 计算奖励
        rewards = [agent.get_reward() for agent in self.agents]

        # 检查是否结束
        dones = [agent.is_done() or self.current_step >= self.max_steps for agent in self.agents]
        done = all(dones)

        # 准备信息字典
        info = {
            'current_step': self.current_step,
            'time': self.time,
            'global_task_completion_rate': self.get_global_task_completion_rate(),
            'num_active_tasks': sum(1 for task in self.tasks if not task.is_completed()),
        }

        return new_state, rewards, done, info

    def _update_environment(self):
        self.weather_system.update()
        self.terrain.update()
        for obstacle in self.dynamic_obstacles:
            obstacle.update(self.time_step)
        self.communication_model.update()
        self.update_time_of_day()

    def _process_agent_actions(self, actions):
        for agent, action in zip(self.agents, actions):
            agent.update(action)
            if self.check_collision(agent):
                agent.handle_collision()

    def _handle_tasks(self):
        new_task = self.task_generator.update()
        if new_task:
            self.tasks.append(new_task)
        self.task_allocator.update()

    def _process_events(self):
        events = self.event_scheduler.get_current_events(self.time)
        for event in events:
            self._handle_event(event)

    def _handle_event(self, event):
        if event.event_type == 'message_received':
            self.communication_model.deliver_message(event.data)
        elif event.event_type == 'task_deadline':
            self.task_allocator.handle_task_deadline(event.data)
        # Add more event types as needed

    def get_state(self):
        state_components = {}

        if self.config['state_components']['include_agents']:
            state_components['agents'] = np.concatenate([agent.get_state() for agent in self.agents])

        if self.config['state_components']['include_weather']:
            state_components['weather'] = self.weather_system.get_state()

        if self.config['state_components']['include_terrain']:
            state_components['terrain'] = self.terrain.get_state()

        if self.config['state_components']['include_time']:
            state_components['time'] = np.array([self.time_of_day / 24.0])

        if self.config['state_components']['include_tasks']:
            state_components['tasks'] = self.task_allocator.get_state()

        if self.config['state_components']['include_obstacles']:
            state_components['obstacles'] = np.concatenate(
                [obstacle.get_state() for obstacle in self.dynamic_obstacles])

        state = np.concatenate(list(state_components.values()))

        print("State components:")
        for key, value in state_components.items():
            print(f"  {key}: {value.shape}")
        print(f"Total state shape: {state.shape}")

        return state

    def get_state_dim(self):
        return self.get_state().shape[0]
    def get_action_dim(self):
        return sum(agent.get_action_dim() for agent in self.agents)

    def get_rewards(self):
        return [agent.get_reward() for agent in self.agents]

    def get_dones(self):
        return [agent.is_done() or self.current_step >= self.max_steps for agent in self.agents]

    def get_info(self):
        return {
            "time": self.time,
            "time_of_day": self.time_of_day,
            "weather": self.weather_system.current_weather,
            "num_completed_tasks": sum(task.is_completed() for task in self.tasks),
            "num_active_tasks": sum(not task.is_completed() for task in self.tasks),
            "agent_energy": {agent.id: agent.energy for agent in self.agents},
            "communication_status": self.communication_model.get_status()
        }

    def update_time_of_day(self):
        self.time_of_day = (self.time_of_day + self.time_step / 3600) % 24

    def check_collision(self, agent):
        # 检查与静态障碍物的碰撞
        for obstacle in self.world.obstacles:
            if obstacle['type'] == 'circle':
                distance = np.linalg.norm(agent.position - obstacle['center'])
                if distance <= obstacle['radius'] + agent.size:
                    return True
            elif obstacle['type'] == 'polygon':
                # 这里需要实现多边形碰撞检测
                pass

        # 检查与动态障碍物的碰撞
        for dynamic_obstacle in self.dynamic_obstacles:
            if dynamic_obstacle.is_colliding(agent.position):
                return True

        # 检查与其他智能体的碰撞
        for other_agent in self.agents:
            if other_agent != agent:
                distance = np.linalg.norm(agent.position - other_agent.position)
                if distance <= agent.size + other_agent.size:
                    return True

        return False

    def send_message(self, sender, receiver, content, priority=1):
        return self.communication_model.send_message(sender, receiver, content, priority)

    def broadcast_message(self, sender, content, range):
        return self.communication_model.broadcast(sender, content, range)

    def plan_path(self, start, goal, method='a_star'):
        if method == 'a_star':
            return self.a_star.plan(start, goal)
        elif method == 'rrt':
            return self.rrt.plan(start, goal)
        else:
            raise ValueError(f"Unknown path planning method: {method}")

    def add_agent(self, agent):
        self.agents.append(agent)
        logger.info(f"Added agent {agent.id} to the environment")

    def remove_agent(self, agent):
        self.agents.remove(agent)
        logger.info(f"Removed agent {agent.id} from the environment")

    def get_objects_in_range(self, position, range):
        objects = []
        for obstacle in self.world.obstacles:
            if obstacle['type'] == 'circle':
                if np.linalg.norm(np.array(obstacle['center']) - position) <= range:
                    objects.append(obstacle)
            elif obstacle['type'] == 'polygon':
                if 'points' in obstacle:
                    center = np.mean(obstacle['points'], axis=0)
                    if np.linalg.norm(center - position) <= range:
                        obstacle_with_center = obstacle.copy()
                        obstacle_with_center['center'] = center
                        objects.append(obstacle_with_center)
                else:
                    print(f"Warning: Polygon obstacle without 'points': {obstacle}")

        # ... rest of the method ...
        for agent in self.agents:
            if np.linalg.norm(agent.position - position) <= range and not np.array_equal(agent.position, position):
                objects.append({'type': 'agent', 'position': agent.position})

        for task in self.tasks:
            if np.linalg.norm(task.get_current_target() - position) <= range:
                objects.append({'type': 'task', 'position': task.get_current_target()})

        return objects
    def get_disaster_areas(self):
        return self.disaster_areas

    def add_disaster_area(self, position, radius):
        self.disaster_areas.append({"position": position, "radius": radius})
        logger.info(f"Added disaster area at position {position} with radius {radius}")

    def remove_disaster_area(self, index):
        removed_area = self.disaster_areas.pop(index)
        logger.info(f"Removed disaster area at position {removed_area['position']}")

    def schedule_event(self, time, event_type, data):
        self.event_scheduler.schedule_event(time, event_type, data)

    def render(self, mode='human'):
        # Implement rendering logic here if needed
        pass

    def close(self):
        # Clean up resources if needed
        pass