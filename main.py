import argparse
import yaml
import numpy as np
from tqdm import tqdm
import pygame
import os
from environment import SAGINEnv
from agents import create_agents
from models import MADDPG
from visualization import PygameVisualizer


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(description="SAGIN Simulation")
    parser.add_argument("--config", type=str, default="config/base_config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train", help="Run mode")
    args = parser.parse_args()

    config = load_config(args.config)
    print("Configuration loaded successfully:")
    print(config)

    env = SAGINEnv(config)
    agents = create_agents(config, env)
    env.agents = agents

    state_dim = env.get_state_dim()
    action_dim = env.action_space.shape[0]

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    maddpg = MADDPG(num_agents=len(agents),
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dim=config['hidden_dim'],
                    actor_lr=config['actor_lr'],
                    critic_lr=config['critic_lr'],
                    gamma=config['gamma'],
                    tau=config['tau'])

    visualizer = PygameVisualizer(env, config)

    if args.mode == "train":
        train(env, maddpg, config, visualizer)
    else:
        evaluate(env, maddpg, config, visualizer)


def train(env, maddpg, config, visualizer):
    # 创建保存模型的目录
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)

    print("Starting training...")
    total_steps = 0
    episode_rewards = []

    for episode in range(config['num_episodes']):
        print(f"Starting episode {episode}")
        states = env.reset()
        states = [state for state in states]
        print(f"Initial states shape: {np.array(states).shape}")
        print(f"Initial states: {states}")
        episode_reward = 0

        for step in range(config['max_steps']):
            print(f"Episode {episode}, Step {step}")
            actions = maddpg.choose_action(states)
            print(f"Chosen actions: {actions}")
            next_states, rewards, dones, _ = env.step(actions)

            if not rewards:  # 如果 rewards 为空，打印警告并跳过此步
                print(f"Warning: Empty rewards at episode {episode}, step {step}")
                continue

            next_states = [state for state in next_states]
            print(f"Rewards: {rewards}")
            print(f"Dones: {dones}")

            maddpg.store_transition(states, actions, rewards, next_states, dones)
            episode_reward += sum(rewards)

            if total_steps % config['update_every'] == 0:
                print("Updating MADDPG")
                maddpg.learn()

            states = next_states
            total_steps += 1

            visualizer.update(env, episode, step, episode_reward)

            if visualizer.should_quit():
                print("Visualization window closed. Stopping training.")
                return

            if all(dones):
                print(f"Episode {episode} finished at step {step}")
                break

        episode_rewards.append(episode_reward)

        if episode % config['print_every'] == 0:
            avg_reward = np.mean(episode_rewards[-config['print_every']:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

        if episode % config['save_every'] == 0:
            maddpg.save(f"{config['model_save_path']}_episode_{episode}.pth")

    maddpg.save(f"{config['model_save_path']}_final.pth")
    print("Training completed")


def evaluate(env, maddpg, config, visualizer):
    maddpg.load(f"{config['model_save_path']}_final.pth")

    total_rewards = []
    for episode in range(config['eval_episodes']):
        states = env.reset()
        states = [state for state in states]
        episode_reward = 0

        for step in range(config['max_steps']):
            actions = maddpg.choose_action(states, noise=0.0)  # No exploration during evaluation
            next_states, rewards, dones, _ = env.step(actions)
            next_states = [state for state in next_states]

            episode_reward += sum(rewards)
            states = next_states

            visualizer.update(env, episode, step, episode_reward)

            if visualizer.should_quit():
                print("Visualization window closed. Stopping evaluation.")
                return

            if all(dones):
                break

        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode}, Total Reward: {episode_reward}")

    print(f"Average Reward over {config['eval_episodes']} episodes: {np.mean(total_rewards)}")


if __name__ == "__main__":
    main()