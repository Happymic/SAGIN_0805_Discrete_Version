import argparse
import os
import numpy as np
import yaml
import logging
from environment.sagin_env import SAGINEnv
from learning.hierarchical_maddpg import HierarchicalMADDPG
from visualization.pygame_visualizer import PygameVisualizer
from utils.distributed_simulation import DistributedSimulation

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    try:
        parser = argparse.ArgumentParser(description="SAGIN Simulation")
        parser.add_argument("--config", type=str, default="config/base_config.yaml", help="Path to config file")
        parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train", help="Run mode")
        args = parser.parse_args()

        config = load_config(args.config)
        env = SAGINEnv(config)

        initial_state = env.get_state()
        state_dim = initial_state.shape[0]
        action_dim = env.get_action_dim()
        logger.info(f"State dimension: {state_dim}")
        logger.info(f"Action dimension: {action_dim}")

        hierarchical_maddpg = HierarchicalMADDPG(
            num_agents=len(env.agents),
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config['hidden_dim'],
            num_options=config['num_options'],
            actor_lr=config['actor_lr'],
            critic_lr=config['critic_lr'],
            gamma=config['gamma'],
            tau=config['tau']
        )

        visualizer = PygameVisualizer(env, config)
        distributed_sim = DistributedSimulation(num_processes=config['num_processes'])

        if args.mode == "train":
            train(env, hierarchical_maddpg, config, visualizer, distributed_sim)
        else:
            evaluate(env, hierarchical_maddpg, config, visualizer)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

def train(env, hierarchical_maddpg, config, visualizer, distributed_sim):
    episode_rewards = []
    for episode in range(config['num_episodes']):
        state = env.reset()
        episode_reward = 0
        for step in range(config['max_steps_per_episode']):
            action = hierarchical_maddpg.select_action(state)
            next_state, reward, done, _ = env.step(action)
            hierarchical_maddpg.store_transition(state, action, reward, next_state, done)
            state = next_state
            episode_reward += sum(reward)

            if visualizer.update(env, episode, step, episode_reward) == False:
                return  # 用户关闭了可视化窗口

            if done:
                break

        episode_rewards.append(episode_reward)

        # 每个episode之后更新网络
        hierarchical_maddpg.update()

        # 记录日志
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(f"Episode {episode}, Average Reward: {avg_reward}")

        # 定期保存模型
        if episode % config['save_interval'] == 0:
            hierarchical_maddpg.save(f"models/hmaddpg_episode_{episode}.pth")

        # 定期评估
        if episode % config['eval_interval'] == 0:
            eval_reward = evaluate(env, hierarchical_maddpg, config, visualizer, num_episodes=5)
            logger.info(f"Evaluation at episode {episode}, Average Reward: {eval_reward}")

    # 保存最终模型
    hierarchical_maddpg.save("models/hmaddpg_final.pth")

def evaluate(env, hierarchical_maddpg, config, visualizer, num_episodes=5):
    try:
        hierarchical_maddpg.load("models/hmaddpg_final.pth")
    except FileNotFoundError:
        logger.warning("No saved model found. Using the current model state.")

    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = hierarchical_maddpg.select_action(state, explore=False)
            next_state, reward, done, _ = env.step(action)
            episode_reward += sum(reward)
            state = next_state

            if visualizer.update(env, episode, env.current_step, episode_reward) == False:
                return np.mean(total_rewards)  # 用户关闭了可视化窗口

        total_rewards.append(episode_reward)
        logger.info(f"Evaluation Episode {episode}, Total Reward: {episode_reward}")

    avg_reward = np.mean(total_rewards)
    logger.info(f"Average Evaluation Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    main()