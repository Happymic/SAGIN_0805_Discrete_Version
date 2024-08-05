import gym
import numpy as np
from .continuous_world import ContinuousWorld
from .weather_system import WeatherSystem
from .dynamic_obstacles import DynamicObstacle


class SAGINEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.world = ContinuousWorld(config['world_width'], config['world_height'],
                                     config['num_obstacles'], config['num_pois'])
        self.weather_system = WeatherSystem(config['world_width'], config['world_height'])
        self.dynamic_obstacles = [DynamicObstacle(
            position=np.random.uniform([0, 0], [config['world_width'], config['world_height']]),
            velocity=np.random.uniform(-1, 1, 2),
            size=np.random.uniform(1, 3),
            env_width=config['world_width'],
            env_height=config['world_height']
        ) for _ in range(config['num_dynamic_obstacles'])]

        self.agents = []
        self.time = 0
        self.max_time = config['max_time']
        self.time_step = config['time_step']

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.get_state_dim(),),
                                                dtype=np.float32)


    def reset(self):
        self.time = 0
        for agent in self.agents:
            agent.reset()
        self.weather_system = WeatherSystem(self.config['world_width'], self.config['world_height'])
        return self.get_state()

    def step(self, actions):
        self.time += self.time_step

        # Update weather and dynamic obstacles
        self.weather_system.update()
        for obstacle in self.dynamic_obstacles:
            obstacle.update(self.time_step)

        # Update agents
        for agent, action in zip(self.agents, actions):
            agent.update(action)

        # Check for collisions and interactions
        self.handle_collisions()
        self.handle_interactions()

        # Get new state, rewards, and done flags
        new_state = self.get_state()
        rewards = self.get_rewards()
        dones = self.get_dones()
        info = self.get_info()

        return new_state, rewards, dones, info

    def get_state(self):
        state = []
        for agent in self.agents:
            agent_state = np.concatenate([
                agent.position,
                agent.velocity,
                [agent.battery],
                self.weather_system.get_weather_at_position(agent.position),
                [len(self.get_objects_in_range(agent.position, agent.sensor_range))],
                [1 if agent.current_task else 0]
            ])
            state.append(agent_state)
        return np.array(state)

    def get_state_dim(self):
        return 9  # position (2), velocity (2), battery (1), weather (1), objects in range (1), has task (1), task progress (1)

    def get_rewards(self):
        rewards = []
        for agent in self.agents:
            reward = 0
            # Base reward for staying operational
            reward += 0.1

            # Reward for task completion
            if agent.current_task and agent.current_task.is_completed():
                reward += 10 * agent.current_task.priority

            # Penalty for battery usage
            reward -= (100 - agent.battery) * 0.01

            # Penalty for collisions
            if self.check_collision(agent):
                reward -= 5

            rewards.append(reward)
        return rewards

    def get_dones(self):
        return [self.time >= self.max_time or agent.battery <= 0 for agent in self.agents]

    def get_info(self):
        return {
            "time": self.time,
            "weather": self.weather_system.current_weather,
            "num_completed_tasks": sum(
                1 for agent in self.agents if agent.current_task and agent.current_task.is_completed())
        }

    def handle_collisions(self):
        for agent in self.agents:
            if self.check_collision(agent):
                agent.velocity = np.zeros(2)
                agent.consume_energy(5)  # Extra energy consumption for collision

    def handle_interactions(self):
        for agent in self.agents:
            if isinstance(agent, (TransportVehicle, RescueVehicle)):
                if agent.current_task and self.is_at_task_location(agent):
                    agent.current_task.update_progress(50)  # Arbitrary progress value
            elif isinstance(agent, UAV):
                objects = self.get_objects_in_range(agent.position, agent.camera_range)
                for obj in objects:
                    if isinstance(obj, Point) and obj in self.world.pois:
                        # UAV has observed a POI
                        pass  # You might want to add some reward or update some state here

    def check_collision(self, agent):
        # Check collision with static obstacles
        if not self.world.is_valid_position(agent.position):
            return True

        # Check collision with dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            if obstacle.is_colliding(agent.position):
                return True

        return False

    def is_at_task_location(self, agent):
        if agent.current_task:
            return np.linalg.norm(agent.position - agent.current_task.target_position) < 1.0
        return False

    def get_objects_in_range(self, position, range):
        objects = self.world.get_objects_in_range(position, range)
        objects.extend([obs for obs in self.dynamic_obstacles if obs.is_colliding(position)])
        return objects

    def add_agent(self, agent):
        self.agents.append(agent)

    def remove_agent(self, agent):
        self.agents.remove(agent)