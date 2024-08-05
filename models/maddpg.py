import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .ddpg import DDPG

class MADDPG:
    def __init__(self, num_agents, state_dims, action_dims, hidden_dim=64, actor_lr=1e-4, critic_lr=1e-3):
        self.agents = [DDPG(state_dims[i], action_dims[i], hidden_dim, actor_lr, critic_lr) for i in range(num_agents)]
        self.num_agents = num_agents

    def act(self, states, noise_std=0.1):
        return [agent.act(state, noise_std) for agent, state in zip(self.agents, states)]

    def update(self, states, actions, rewards, next_states, dones):
        for i, agent in enumerate(self.agents):
            agent.update(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def save(self, filename):
        state = {f'agent_{i}': agent.actor.state_dict() for i, agent in enumerate(self.agents)}
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(state[f'agent_{i}'])
            agent.actor_target.load_state_dict(state[f'agent_{i}'])