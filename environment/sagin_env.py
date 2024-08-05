import numpy as np
import gym
from gym import spaces
from .grid_world import GridWorld


class SAGINEnv(gym.Env):
    def __init__(self, config):
        super(SAGINEnv, self).__init__()
        self.config = config
        self.grid_world = GridWorld(config)
        self.agents = []
        self.tasks = []
        self.time = 0
        self.max_time = config['max_time']

        # 为不同类型的智能体定义不同的动作空间
        self.action_spaces = {
            "signal_detector": spaces.Discrete(5),
            "transport_vehicle": spaces.Discrete(5),
            "rescue_vehicle": spaces.Discrete(5),
            "uav": spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32),
            "satellite": spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32),
            "fixed_station": spaces.Discrete(5)
        }

        # 定义一个通用的观察空间
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=0, high=max(config['grid_width'], config['grid_height']), shape=(2,),
                                   dtype=np.float32),
            "agent_type": spaces.Discrete(6),
            "nearby_agents": spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
            "nearby_tasks": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
            "battery_level": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "communication_range": spaces.Box(low=0, high=max(config['grid_width'], config['grid_height']), shape=(1,),
                                              dtype=np.float32),
            "global_task_completion": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        print("Observation space structure:", self.observation_space)
        for key, space in self.observation_space.items():
            print(f"{key}: {space}")

    def reset(self):
        self.time = 0
        self.tasks = self.generate_tasks()
        for agent in self.agents:
            agent.reset()
            if agent.position is None:
                agent.position = self.grid_world.get_random_position()
        return self.get_observation()

    def step(self, actions):
        self.time += 1
        rewards = []
        dones = []
        infos = []

        print("Actions received:", actions)  # 调试信息

        for agent, action in zip(self.agents, actions):
            print(f"Processing action for agent {agent.type}: {action}")  # 调试信息

            if isinstance(self.action_spaces[agent.type], spaces.Discrete):
                discrete_action = action
            else:
                # 对于连续动作，我们需要将其转换为离散动作
                if isinstance(action, np.ndarray):
                    if action.size == 1:
                        discrete_action = int(action.item())
                    else:
                        discrete_action = int(np.argmax(action))
                elif isinstance(action, (list, tuple)):
                    discrete_action = int(np.argmax(action))
                else:
                    discrete_action = int(action)

            print(f"Discrete action: {discrete_action}")  # 调试信息

            if discrete_action < 4:  # Movement action
                agent.move(discrete_action)
            else:  # Perform action
                self.perform_agent_action(agent)

            reward = self.calculate_reward(agent)
            rewards.append(reward)
            dones.append(self.time >= self.max_time)

            try:
                infos.append(self.get_agent_info(agent))
            except Exception as e:
                print(f"Error getting agent info for {agent.type}: {e}")
                print(f"Agent attributes: {dir(agent)}")
                infos.append({})  # 添加一个空字典作为替代

        next_states = self.get_observation()
        print("Next states shape:", [state.shape for state in next_states])
        return next_states, rewards, dones, infos
    def generate_tasks(self):
        tasks = []
        for _ in range(self.config['num_tasks']):
            task_type = np.random.choice(["transport", "rescue", "monitor"])
            start = self.grid_world.get_random_position()
            end = self.grid_world.get_random_position()
            priority = np.random.randint(1, 4)
            tasks.append({
                "type": task_type,
                "start": start,
                "end": end,
                "priority": priority,
                "status": "pending",
                "progress": 0.0,
                "assigned_to": None
            })
        return tasks

    def calculate_reward(self, agent):
        reward = 0

        # Base reward for staying operational
        reward += 0.1

        # Reward for task completion
        if agent.type in ["transport_vehicle", "rescue_vehicle", "uav"]:
            for task in self.tasks:
                if task["assigned_to"] == agent.id:
                    if task["status"] == "completed":
                        reward += 10 * task["priority"]
                    elif task["status"] == "in_progress":
                        reward += 0.1 * task["progress"] * task["priority"]

        # Penalty for battery usage
        if hasattr(agent, "battery"):
            reward -= (100 - agent.battery) * 0.01

        # Reward for successful communication
        if agent.type in ["uav", "satellite"]:
            reward += len(self.broadcast_message(agent, "test")) * 0.5

        # Special rewards for specific agent types
        if agent.type == "signal_detector":
            if self.detect_signal(agent.position):
                reward += 5
        elif agent.type == "uav":
            visible_area = self.get_area_info(agent.position, agent.altitude)
            reward += len(visible_area["pois"]) * 0.2 + len(visible_area["disaster_areas"]) * 0.3

        return reward

    def get_observation(self):
        agent_types = ["signal_detector", "transport_vehicle", "rescue_vehicle", "uav", "satellite", "fixed_station"]
        observations = []
        for agent in self.agents:
            obs = np.concatenate([
                agent.position,
                [agent_types.index(agent.type)],
                self.get_nearby_agents(agent),
                self.get_nearby_tasks(agent),
                [agent.battery if hasattr(agent, "battery") else 100.0],
                [self.config['communication_range']],
                [sum(task["status"] == "completed" for task in self.tasks) / len(self.tasks)]
            ])
            observations.append(obs)
        return observations

    def get_nearby_agents(self, agent):
        agent_types = ["signal_detector", "transport_vehicle", "rescue_vehicle", "uav", "satellite", "fixed_station"]
        nearby = [0] * len(agent_types)
        for other in self.agents:
            if other != agent and other.position is not None and agent.position is not None:
                if self.grid_world.get_manhattan_distance(agent.position, other.position) <= self.config[
                    'communication_range']:
                    if other.type in agent_types:
                        nearby[agent_types.index(other.type)] += 1
        return np.array(nearby) / len(self.agents)

    def get_nearby_tasks(self, agent):
        nearby = [0] * 3
        for task in self.tasks:
            if self.grid_world.get_manhattan_distance(agent.position, task["start"]) <= self.config[
                'communication_range']:
                nearby[["transport", "rescue", "monitor"].index(task["type"])] += 1
        return np.array(nearby) / len(self.tasks)

    def get_agent_info(self, agent):
        info = {
            "id": agent.id,
            "type": agent.type,
            "position": agent.position.tolist(),
            "battery": agent.battery if hasattr(agent, "battery") else 100,
        }

        if hasattr(agent, "assigned_task") and agent.assigned_task:
            if isinstance(agent.assigned_task, dict) and "id" in agent.assigned_task:
                info["assigned_task"] = agent.assigned_task["id"]
            else:
                info["assigned_task"] = "Unknown"
        else:
            info["assigned_task"] = None

        return info

    def perform_agent_action(self, agent):
        if agent.type == "signal_detector":
            if self.detect_signal(agent.position):
                self.create_task_from_signal(agent.position)
        elif agent.type in ["transport_vehicle", "rescue_vehicle"]:
            self.process_ground_vehicle_action(agent)
        elif agent.type == "uav":
            self.process_uav_action(agent)
        elif agent.type == "satellite":
            self.process_satellite_action(agent)
        elif agent.type == "fixed_station":
            self.process_fixed_station_action(agent)

    def process_fixed_station_action(self, agent):
        print(f"Fixed station {agent.id} is processing data")
    def create_task_from_signal(self, position):
        task_type = np.random.choice(["transport", "rescue"])
        end = self.grid_world.get_random_position()
        priority = np.random.randint(1, 4)
        new_task = {
            "type": task_type,
            "start": position,
            "end": end,
            "priority": priority,
            "status": "pending",
            "progress": 0.0,
            "assigned_to": None
        }
        self.tasks.append(new_task)

    def process_ground_vehicle_action(self, agent):
        if agent.assigned_task:
            task = agent.assigned_task
            if np.array_equal(agent.position, task["end"]):
                task["status"] = "completed"
                task["progress"] = 1.0
                agent.assigned_task = None
            else:
                task["progress"] += 0.1
                task["status"] = "in_progress"

    def process_uav_action(self, agent):
        visible_area = self.get_area_info(agent.position, agent.altitude)
        for poi in visible_area["pois"]:
            if np.random.random() < 0.1:  # 10% chance to create a new task at a POI
                self.create_task_from_signal(poi)

    def process_satellite_action(self, agent):
        global_info = self.get_global_info()
        unassigned_tasks = [task for task in self.tasks if task["status"] == "pending"]
        available_agents = [a for a in self.agents if
                            a.type in ["transport_vehicle", "rescue_vehicle", "uav"] and not hasattr(a,
                                                                                                     "assigned_task")]

        for task in unassigned_tasks:
            if available_agents:
                assigned_agent = min(available_agents,
                                     key=lambda a: self.grid_world.get_manhattan_distance(a.position, task["start"]))
                task["assigned_to"] = assigned_agent.id
                task["status"] = "assigned"
                assigned_agent.assigned_task = task
                available_agents.remove(assigned_agent)

    def detect_signal(self, position):
        return any(self.grid_world.get_manhattan_distance(position, task["start"]) < 5 for task in self.tasks if
                   task["status"] == "pending")

    def get_area_info(self, position, altitude):
        visible_range = altitude // 2
        visible_area = set()
        for dx in range(-visible_range, visible_range + 1):
            for dy in range(-visible_range, visible_range + 1):
                x, y = position[0] + dx, position[1] + dy
                if 0 <= x < self.grid_world.width and 0 <= y < self.grid_world.height:
                    visible_area.add((x, y))
        return {
            "obstacles": visible_area.intersection(self.grid_world.obstacles),
            "pois": visible_area.intersection(self.grid_world.pois),
            "disaster_areas": visible_area.intersection(self.grid_world.disaster_areas),
        }

    def get_global_info(self):
        return {
            "agents": [self.get_agent_info(agent) for agent in self.agents],
            "tasks": self.tasks,
            "pois": list(self.grid_world.pois),
            "disaster_areas": list(self.grid_world.disaster_areas),
        }

    def send_message(self, sender, message, target):
        if self.grid_world.get_manhattan_distance(sender.position, target.position) <= self.config[
            'communication_range']:
            return True
        return False

    def broadcast_message(self, sender, message):
        received_by = []
        for agent in self.agents:
            if agent != sender and self.grid_world.get_manhattan_distance(sender.position, agent.position) <= \
                    self.config['communication_range']:
                received_by.append(agent)
        return received_by

    def relay_message(self, relay_agent, message, source, target):
        if (self.grid_world.get_manhattan_distance(source.position, relay_agent.position) <= self.config[
            'communication_range'] and
                self.grid_world.get_manhattan_distance(relay_agent.position, target.position) <= self.config[
                    'communication_range']):
            return True
        return False