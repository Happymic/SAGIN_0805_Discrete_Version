import numpy as np
from .distributed_trainer import DistributedTrainer
from .integrated_trainer import IntegratedTrainer


class HybridTrainer:
    def __init__(self, env, agents, models, config):
        self.distributed_trainer = DistributedTrainer(env, agents, models[:-1], config)
        self.integrated_trainer = IntegratedTrainer(env, agents, [models[-1]], config)
        self.config = config
        self.switch_threshold = config['switch_threshold']
        self.current_performance = 0

    def train(self, num_episodes, visualizer):
        for episode in range(num_episodes):
            if self.should_switch_to_integrated():
                episode_rewards, steps = self.integrated_trainer.train_episode(visualizer, episode)
            else:
                episode_rewards, steps = self.distributed_trainer.train_episode(visualizer, episode)

            self.update_performance(np.mean(episode_rewards))

            if episode % self.config['print_every'] == 0:
                print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards)}")

    # ... (rest of the code remains the same)
    def should_switch_to_integrated(self):
        return self.current_performance < self.switch_threshold

    def update_performance(self, episode_reward):
        self.current_performance = 0.95 * self.current_performance + 0.05 * episode_reward

    def evaluate(self, num_episodes, visualizer=None):
        return self.integrated_trainer.evaluate(num_episodes, visualizer)

    def save_models(self, path):
        self.integrated_trainer.save_models(path)

    def load_models(self, path):
        self.integrated_trainer.load_models(path)