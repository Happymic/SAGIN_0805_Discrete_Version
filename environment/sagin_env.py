import gym
import numpy as np
from gym import spaces
from typing import List, Dict, Any
import logging
from shapely.geometry import Point, Polygon
from .continuous_world import ContinuousWorld
from .weather_system import WeatherSystem
from .terrain import Terrain
from communication.communication_model import CommunicationModel
from tasks.task_generator import TaskGenerator
from tasks.task_allocator import TaskAllocator
from path_planning.a_star import AStar
from path_planning.rrt import RRT
from agents.base_agent import BaseAgent
from utils.event_scheduler import EventScheduler
from visualization.pygame_visualizer import PygameVisualizer

logger = logging.getLogger(__name__)


class SAGINEnv(gym.Env):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._resources_to_clean = []
        # Initialize world and environmental components
        self.world = ContinuousWorld(config['world_width'], config['world_height'],
                                     config['num_obstacles'], config['num_pois'])
        self.terrain = Terrain(config['world_width'], config['world_height'])
        self.weather_system = WeatherSystem(config['world_width'], config['world_height'],
                                            config['weather_change_rate'])

        # Initialize communication and task-related components
        self.communication_model = CommunicationModel(self)
        self.task_generator = TaskGenerator(self, config)
        self.task_allocator = TaskAllocator(self)

        # Initialize path planning algorithms
        self.a_star = AStar(self)
        self.rrt = RRT(self)

        # Initialize lists for agents, tasks, and disaster areas
        self.agents: List[BaseAgent] = []
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

        # Create default agents
        self._create_default_agents()

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.get_action_dim(),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.get_state_dim(),), dtype=np.float32)

        # Initialize visualizer
        self.pygame_visualizer = PygameVisualizer(self, config)

        # List to track resources that need cleaning
        self._resources_to_clean = []

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Clean up all resources"""
        for resource in self._resources_to_clean:
            if hasattr(resource, 'close'):
                resource.close()
            elif hasattr(resource, '__del__'):
                resource.__del__()

        self._resources_to_clean.clear()

        if self.pygame_visualizer:
            self.pygame_visualizer.close()

        # Clear logging handlers
        logging.getLogger().handlers.clear()

    def add_resource_to_clean(self, resource):
        """Add a resource that needs to be cleaned when the environment closes"""
        self._resources_to_clean.append(resource)

    def _create_default_agents(self):
        from agents import SignalDetector, TransportVehicle, RescueVehicle, UAV, Satellite, FixedStation

        agent_types = {
            'signal_detector': (SignalDetector, self.config['num_signal_detectors']),
            'transport_vehicle': (TransportVehicle, self.config['num_transport_vehicles']),
            'rescue_vehicle': (RescueVehicle, self.config['num_rescue_vehicles']),
            'uav': (UAV, self.config['num_uavs']),
            'satellite': (Satellite, self.config['num_satellites']),
            'fixed_station': (FixedStation, self.config['num_fixed_stations'])
        }

        for agent_type, (AgentClass, num_agents) in agent_types.items():
            for i in range(num_agents):
                temp_agent = AgentClass(f"{agent_type.upper()}_{i}", np.zeros(3), self)
                position = self.world.get_random_valid_position(temp_agent.get_agent_type(), temp_agent.size)
                agent = AgentClass(f"{agent_type.upper()}_{i}", position, self)
                self.agents.append(agent)
                self.add_resource_to_clean(agent)

    def reset(self):
        logger.info("Resetting SAGIN environment")
        self.time = 0
        self.current_step = 0
        self.weather_system.reset()
        self.terrain.reset()
        self.time_of_day = np.random.randint(0, 24)
        self.event_scheduler.clear()

        for agent in self.agents:
            agent.reset()
            agent.position = self.world.get_random_valid_position(agent.get_agent_type(), agent.size)
            logger.debug(f"Reset agent {agent.id}")

        self.task_generator.reset()
        self.task_allocator.reset()
        logger.debug(f"Number of agents after reset: {len(self.agents)}")
        return self.get_state()

    def step(self, actions):
        self.current_step += 1
        self.time += self.time_step

        # Execute agent actions
        rewards = []
        for agent, action in zip(self.agents, actions):
            old_position = agent.position.copy()
            old_energy = agent.energy

            # Check if agent is at a charging station
            if self.is_agent_at_charging_station(agent):
                agent.charge(1.0)  # Charge by 1 unit per time step
                action = np.zeros_like(action)  # Agent doesn't move while charging

            agent.update(action)

            # Calculate rewards
            movement_reward = np.linalg.norm(agent.position - old_position)
            energy_reward = (agent.energy - old_energy) * 0.1  # Reward for charging
            task_reward = self.calculate_reward(agent)

            total_reward = movement_reward + energy_reward + task_reward
            rewards.append(total_reward)

        # Update environment and tasks
        self._update_environment()
        self.task_generator.update()
        self.task_allocator.update()

        # Get new state
        new_state = self.get_state()

        # Check if episode is done
        dones = [agent.is_done() or self.current_step >= self.max_steps for agent in self.agents]
        done = all(dones)

        info = {
            'current_step': self.current_step,
            'time': self.time,
            'global_poi_completion_rate': self.get_global_poi_completion_rate(),
            'num_active_pois': len(self.task_generator.get_active_pois()),
            'num_charging_agents': sum(1 for agent in self.agents if self.is_agent_at_charging_station(agent)),
        }

        return new_state, rewards, done, info

    def is_agent_at_charging_station(self, agent):
        agent_point = Point(agent.position[0], agent.position[1])
        for station in self.world.charging_stations:
            if agent_point.distance(station) < 5:  # Within 5 units
                return True
        return False
    def calculate_reward(self, agent):
        reward = 0

        # POI completion reward
        if agent.last_completed_task:
            reward += 10 * agent.last_completed_task["priority"]
            agent.last_completed_task = None

        # Collision penalty
        if self.check_collision(agent):
            reward -= 5

        # Energy efficiency reward
        energy_efficiency = agent.energy / agent.max_energy
        reward += energy_efficiency

        # Distance to task reward
        if agent.current_task:
            distance_to_task = np.linalg.norm(agent.position - agent.current_task["position"])
            reward += 20 / (1 + distance_to_task)

        return reward

    def _update_environment(self):
        self.weather_system.update()
        self.terrain.update()
        self.communication_model.update()
        self.update_time_of_day()
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

        state = np.concatenate([component for component in state_components.values() if component.size > 0])

        logger.debug(f"State shape: {state.shape}")
        return state

    def get_state_dim(self):
        return self.get_state().shape[0]

    def get_action_dim(self):
        return sum(agent.get_action_dim() for agent in self.agents)

    def get_global_poi_completion_rate(self):
        completed_pois = sum(1 for poi in self.task_generator.pois if poi["completed"])
        total_pois = len(self.task_generator.pois)
        return completed_pois / total_pois if total_pois > 0 else 0

    def update_time_of_day(self):
        self.time_of_day = (self.time_of_day + self.time_step / 3600) % 24

    def check_collision(self, agent):
        agent_position = Point(agent.position[0], agent.position[1])

        for obstacle in self.world.obstacles:
            if obstacle['height'] > agent.altitude:
                if obstacle['type'] == 'circle':
                    center = Point(obstacle['center'])
                    if agent_position.distance(center) <= obstacle['radius'] + agent.size:
                        return True
                elif obstacle['type'] == 'polygon':
                    polygon = Polygon(obstacle['points'])
                    if polygon.distance(agent_position) <= agent.size:
                        return True

        for other_agent in self.agents:
            if other_agent != agent:
                distance = np.linalg.norm(agent.position[:2] - other_agent.position[:2])
                if distance <= agent.size + other_agent.size:
                    return True

        return False

    def is_valid_position(self, position, agent_type, agent_size, altitude):
        return self.world.is_valid_position(position, agent_type, agent_size, altitude)

    def get_objects_in_range(self, position, range, agent_type):
        objects = []
        for obstacle in self.world.obstacles:
            if agent_type == "ground" or obstacle['height'] > position[2]:
                if obstacle['type'] == 'circle':
                    center = np.array(obstacle['center'] + [obstacle['height']])
                    if np.linalg.norm(center - position) <= range:
                        objects.append(obstacle)
                elif obstacle['type'] == 'polygon':
                    center = np.mean(obstacle['points'], axis=0)
                    center = np.append(center, obstacle['height'])
                    if np.linalg.norm(center - position) <= range:
                        objects.append(obstacle)

        for agent in self.agents:
            if np.linalg.norm(agent.position - position) <= range:
                objects.append({'type': 'agent', 'position': agent.position, 'id': agent.id})

        for poi in self.task_generator.get_active_pois():
            if np.linalg.norm(poi["position"] - position) <= range:
                objects.append({'type': 'poi', 'position': poi["position"], 'id': poi["id"]})

        return objects

    def send_message(self, sender, receiver, content, priority=1):
        return self.communication_model.send_message(sender, receiver, content, priority)

    def broadcast_message(self, sender, content, range):
        return self.communication_model.broadcast(sender, content, range)

    def plan_path(self, start, goal, method='a_star', agent_type="ground", agent_size=1.0):
        if method == 'a_star':
            self.a_star.agent_type = agent_type
            self.a_star.agent_size = agent_size
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

    def get_disaster_areas(self):
        return self.disaster_areas

    def add_disaster_area(self, disaster_area):
        self.disaster_areas.append(disaster_area)
        logger.info(f"Added disaster area at position {disaster_area['position']} with radius {disaster_area['radius']}")

    def remove_disaster_area(self, index):
        removed_area = self.disaster_areas.pop(index)
        logger.info(f"Removed disaster area at position {removed_area['position']}")

    def schedule_event(self, time, event_type, data):
        self.event_scheduler.schedule_event(time, event_type, data)

    def render(self, mode='human'):
        if self.pygame_visualizer:
            self.pygame_visualizer.render()
    def close(self):
        # Clean up resources if needed
        pass