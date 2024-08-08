import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
import logging
import torch
import cProfile
import pstats
from datetime import datetime
from environment.sagin_env import SAGINEnv
from learning.hierarchical_maddpg import HierarchicalMADDPG
from visualization.pygame_visualizer import PygameVisualizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(description="SAGIN Simulation")
    parser.add_argument("--config", type=str, default="config/base_config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "visualize"], default="train",
                        help="Run mode")
    args = parser.parse_args()

    config = load_config(args.config)
    env = SAGINEnv(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    timestamp = get_timestamp()
    model_dir = f"models_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    hierarchical_maddpg = HierarchicalMADDPG(
        num_agents=len(env.agents),
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
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
        profile_train(env, hierarchical_maddpg, config, device, visualizer, model_dir, timestamp)
    elif args.mode == "evaluate":
        profile_evaluate(env, hierarchical_maddpg, config, visualizer, model_dir, timestamp)
    elif args.mode == "visualize":
        profile_visualize(env, hierarchical_maddpg, config, model_dir, timestamp)


def profile_train(env, hierarchical_maddpg, config, device, visualizer, model_dir, timestamp):
    profiler = cProfile.Profile()
    profiler.enable()

    train(env, hierarchical_maddpg, config, device, visualizer, model_dir, timestamp)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats_path = os.path.join(model_dir, f"train_profile_{timestamp}.txt")
    stats.dump_stats(stats_path)
    logger.info(f"Training profile saved to {stats_path}")


def profile_evaluate(env, hierarchical_maddpg, config, visualizer, model_dir, timestamp):
    profiler = cProfile.Profile()
    profiler.enable()

    evaluate(env, hierarchical_maddpg, config, visualizer, model_dir, timestamp)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats_path = os.path.join(model_dir, f"evaluate_profile_{timestamp}.txt")
    stats.dump_stats(stats_path)
    logger.info(f"Evaluation profile saved to {stats_path}")


def profile_visualize(env, hierarchical_maddpg, config, model_dir, timestamp):
    profiler = cProfile.Profile()
    profiler.enable()

    visualize(env, hierarchical_maddpg, config, model_dir, timestamp)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats_path = os.path.join(model_dir, f"visualize_profile_{timestamp}.txt")
    stats.dump_stats(stats_path)
    logger.info(f"Visualization profile saved to {stats_path}")


def train(env, hierarchical_maddpg, config, device, visualizer, model_dir, timestamp):
    logger.info(f"Starting training at {timestamp}")

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

            if step % config['visualization_interval'] == 0:
                if not visualizer.update(env, episode, step, episode_reward):
                    logger.info("Visualization window closed. Stopping training.")
                    return

            if done:
                break

        hierarchical_maddpg.update()
        episode_rewards.append(episode_reward)
        global_completion_rates.append(env.get_global_poi_completion_rate())

        logger.info(
            f"Episode {episode + 1}/{config['num_episodes']}, Reward: {episode_reward:.2f}, Completion Rate: {global_completion_rates[-1]:.2f}")

        if (episode + 1) % config['save_interval'] == 0:
            model_path = os.path.join(model_dir, f"hmaddpg_episode_{episode + 1}_{timestamp}.pth")
            hierarchical_maddpg.save(model_path)
            logger.info(f"Model saved to {model_path}")

    final_model_path = os.path.join(model_dir, f"hmaddpg_final_{timestamp}.pth")
    hierarchical_maddpg.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    plot_training_curve(episode_rewards, global_completion_rates, model_dir, timestamp)


def evaluate(env, hierarchical_maddpg, config, visualizer, model_dir, timestamp):
    logger.info(f"Starting evaluation at {timestamp}")

    model_path = os.path.join(model_dir, f"hmaddpg_final_{timestamp}.pth")
    try:
        hierarchical_maddpg.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        return

    num_episodes = 10
    total_rewards = []
    global_completion_rates = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            action = hierarchical_maddpg.select_action(state, explore=False)
            next_state, reward, done, info = env.step(action)
            episode_reward += sum(reward)
            state = next_state

            if step % config['visualization_interval'] == 0:
                if not visualizer.update(env, episode, step, episode_reward):
                    logger.info("Visualization window closed. Stopping evaluation.")
                    return

            step += 1

        total_rewards.append(episode_reward)
        global_completion_rates.append(env.get_global_poi_completion_rate())
        logger.info(f"Evaluation Episode {episode + 1}/{num_episodes}, "
                    f"Reward: {episode_reward:.2f}, "
                    f"Completion Rate: {global_completion_rates[-1]:.2f}")

    avg_reward = np.mean(total_rewards)
    avg_completion_rate = np.mean(global_completion_rates)
    logger.info(f"Average Evaluation Reward: {avg_reward:.2f}")
    logger.info(f"Average Global Completion Rate: {avg_completion_rate:.2f}")

    plot_evaluation_results(total_rewards, global_completion_rates, model_dir, timestamp)


def visualize(env, hierarchical_maddpg, config, model_dir, timestamp):
    logger.info(f"Starting visualization at {timestamp}")

    model_path = os.path.join(model_dir, f"hmaddpg_final_{timestamp}.pth")
    try:
        hierarchical_maddpg.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        return

    visualizer = PygameVisualizer(env, config)
    state = env.reset()
    done = False
    step = 0

    while not done:
        action = hierarchical_maddpg.select_action(state, explore=False)
        next_state, reward, done, info = env.step(action)
        state = next_state

        if not visualizer.update(env, 0, step, sum(reward)):
            logger.info("Visualization window closed.")
            break

        step += 1

    visualizer.close()


def plot_training_curve(episode_rewards, global_completion_rates, model_dir, timestamp):
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
    plot_path = os.path.join(model_dir, f"training_curves_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Training curves saved to {plot_path}")


def plot_evaluation_results(total_rewards, global_completion_rates, model_dir, timestamp):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(total_rewards)), total_rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.bar(range(len(global_completion_rates)), global_completion_rates)
    plt.title('Evaluation Global Completion Rates')
    plt.xlabel('Episode')
    plt.ylabel('Completion Rate')

    plt.tight_layout()
    plot_path = os.path.join(model_dir, f"evaluation_results_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Evaluation results saved to {plot_path}")


if __name__ == "__main__":
    main()