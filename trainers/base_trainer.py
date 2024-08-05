import numpy as np
from tqdm import tqdm
from models import MADDPG

class BaseTrainer:
    def __init__(self, env, agents, models, config):
        self.env = env
        self.agents = agents
        self.models = models
        self.config = config

    def train_episode(self, visualizer, episode):
        states = self.env.reset()
        total_rewards = np.zeros(len(self.agents))
        done = False
        step = 0

        while not done and step < self.config['max_steps']:
            if self.models and isinstance(self.models[-1], MADDPG):
                actions = self.models[-1].act(states)
            elif self.models:
                actions = [model.act(state) for model, state in zip(self.models, states)]
            else:
                actions = [agent.act(state) for agent, state in zip(self.agents, states)]

            discrete_actions = [np.argmax(action) if isinstance(action, np.ndarray) else action for action in actions]

            next_states, rewards, dones, _ = self.env.step(discrete_actions)

            self.update_models(states, actions, rewards, next_states, dones)

            states = next_states
            total_rewards += rewards
            done = all(dones)
            step += 1

            if visualizer:
                if not visualizer.update(episode, step):
                    return total_rewards, step

        return total_rewards, step

    def update_models(self, states, actions, rewards, next_states, dones):
        raise NotImplementedError

    def train(self, num_episodes, visualizer=None):
        for episode in tqdm(range(num_episodes)):
            episode_rewards = self.train_episode()

            if visualizer and episode % self.config['visualize_every'] == 0:  # 修改这里
                visualizer.update()

            if episode % self.config['print_every'] == 0:  # 修改这里
                print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards)}")

    def evaluate(self, num_episodes, visualizer=None):
        total_rewards = np.zeros(len(self.agents))
        for episode in tqdm(range(num_episodes)):
            states = self.env.reset()
            episode_rewards = np.zeros(len(self.agents))
            done = False
            step = 0

            while not done and step < self.config['max_steps']:  # 修改这里
                actions = [model.act(state, noise_std=0) for model, state in zip(self.models, states)]
                next_states, rewards, dones, _ = self.env.step(actions)

                if visualizer:
                    visualizer.update()

                states = next_states
                episode_rewards += rewards
                done = all(dones)
                step += 1

            total_rewards += episode_rewards

        avg_rewards = total_rewards / num_episodes
        print(f"Evaluation over {num_episodes} episodes: Average Reward: {np.mean(avg_rewards)}")
        return avg_rewards

    def save_models(self, path):
        for i, model in enumerate(self.models):
            model.save(f"{path}/model_{i}.pth")

    def load_models(self, path):
        for i, model in enumerate(self.models):
            model.load(f"{path}/model_{i}.pth")