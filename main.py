import argparse
import yaml
import numpy as np
from gym import spaces
from environment import SAGINEnv
from agents import create_agents
from models import DQN, DDPG, MADDPG
from trainers import HybridTrainer
from visualization import PygameVisualizer

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_models(config, env):
    models = []
    state_dims = [15 for _ in env.agents]  # 每个智能体的状态维度
    action_dims = [5 for _ in env.agents]  # 每个智能体的动作维度

    if config['use_maddpg']:
        maddpg = MADDPG(num_agents=len(env.agents),
                        state_dims=state_dims,
                        action_dims=action_dims,
                        hidden_dim=config['hidden_dim'],
                        actor_lr=config['actor_lr'],
                        critic_lr=config['critic_lr'])
        models.append(maddpg)
    else:
        for i in range(len(env.agents)):
            if config['model_type'] == 'dqn':
                models.append(DQN(state_dims[i], action_dims[i], config['hidden_dim']))
            else:
                models.append(DDPG(state_dims[i], action_dims[i], config['hidden_dim']))

    return models
def main():
    parser = argparse.ArgumentParser(description="SAGIN Simulation")
    parser.add_argument("--config", type=str, default="config/base_config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train", help="Run mode")
    args = parser.parse_args()

    config = load_config(args.config)

    env = SAGINEnv(config)
    agents = create_agents(config, env)
    env.agents = agents
    models = create_models(config, env)

    trainer = HybridTrainer(env, agents, models, config)
    visualizer = PygameVisualizer(env, config)

    if args.mode == "train":
        trainer.train(config['num_episodes'], visualizer)
        trainer.save_models("saved_models")
    else:
        trainer.load_models("saved_models")
        trainer.evaluate(config['num_eval_episodes'], visualizer)

    visualizer.close()

if __name__ == "__main__":
    main()