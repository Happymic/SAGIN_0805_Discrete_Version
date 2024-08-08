import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
import logging
from tqdm import tqdm
from environment.sagin_env import SAGINEnv
from learning.hierarchical_maddpg import HierarchicalMADDPG
from visualization.pygame_visualizer import PygameVisualizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="SAGIN Simulation")
    parser.add_argument("--config", type=str, default="config/base_config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "visualize"], default="train", help="Run mode")
    args = parser.parse_args()

    config = load_config(args.config)
    env = SAGINEnv(config)

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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
        tau=config['tau'],
        device=device
    )

    visualizer = PygameVisualizer(env, config)

    if args.mode == "train":
        train(env, hierarchical_maddpg, config, device, visualizer)
    elif args.mode == "evaluate":
        evaluate(env, hierarchical_maddpg, config, visualizer)
    elif args.mode == "visualize":
        print("Visualize")
        # visualize(env, hierarchical_maddpg, config)

def train(env, hierarchical_maddpg, config, device, visualizer):
    os.makedirs("models", exist_ok=True)
    episode_rewards = []
    global_completion_rates = []

    for episode in range(config['num_episodes']):
        state = env.reset()
        episode_reward = 0
        for step in range(config['max_steps_per_episode']):
            action = hierarchical_maddpg.select_action(state)
            next_state, reward, done, info = env.step(action)

            hierarchical_maddpg.store_transition(state, action, reward, next_state, done)
            state = next_state
            episode_reward += sum(reward)

            # 更新可视化
            if step % config['visualization_interval'] == 0:
                if not visualizer.update(env, episode, step, episode_reward):
                    logger.info("Visualization window closed. Stopping training.")
                    return

            if done:
                break

        hierarchical_maddpg.update()
        episode_rewards.append(episode_reward)
        global_completion_rates.append(env.get_global_poi_completion_rate())

        logger.info(f"Episode {episode + 1}/{config['num_episodes']}, Reward: {episode_reward:.2f}, Completion Rate: {global_completion_rates[-1]:.2f}")

        if (episode + 1) % config['save_interval'] == 0:
            hierarchical_maddpg.save(f"models/hmaddpg_episode_{episode+1}.pth")

    hierarchical_maddpg.save("models/hmaddpg_final.pth")
    plot_training_curve(episode_rewards, global_completion_rates)

def evaluate(env, hierarchical_maddpg, config, visualizer, num_episodes=5):
    try:
        hierarchical_maddpg.load("models/hmaddpg_final.pth")
    except FileNotFoundError:
        logger.warning("No saved model found. Using the current model state.")

    total_rewards = []
    global_completion_rates = []

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
        global_completion_rates.append(env.get_global_poi_completion_rate())
        logger.info(
            f"Evaluation Episode {episode}, Total Reward: {episode_reward}, Global Completion Rate: {global_completion_rates[-1]}")

    avg_reward = np.mean(total_rewards)
    avg_completion_rate = np.mean(global_completion_rates)
    logger.info(f"Average Evaluation Reward over {num_episodes} episodes: {avg_reward}")
    logger.info(f"Average Global Completion Rate over {num_episodes} episodes: {avg_completion_rate}")
    return avg_reward, avg_completion_rate

def plot_training_curve(episode_rewards, global_completion_rates):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Training Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(global_completion_rates)
    plt.title('Global POI Completion Rate')
    plt.xlabel('Episode')
    plt.ylabel('Completion Rate')

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == "__main__":
    main()