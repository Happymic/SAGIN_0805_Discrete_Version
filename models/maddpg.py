import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_agents):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim * num_agents + action_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

class MADDPG:
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, tau):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau

        self.actors = [Actor(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        self.critics = [Critic(state_dim, action_dim, hidden_dim, num_agents) for _ in range(num_agents)]
        self.actors_target = [Actor(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        self.critics_target = [Critic(state_dim, action_dim, hidden_dim, num_agents) for _ in range(num_agents)]

        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=actor_lr) for i in range(num_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=critic_lr) for i in range(num_agents)]

        for i in range(num_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

        self.memory = deque(maxlen=100000)
        self.batch_size = 64

    def choose_action(self, states, noise=0.1):
        actions = []
        for i, actor in enumerate(self.actors):
            state = torch.FloatTensor(states[i]).unsqueeze(0)
            action = actor(state).squeeze().detach().numpy()
            action += np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -1, 1)
            actions.append(action)
        return actions

    def store_transition(self, states, actions, rewards, next_states, dones):
        self.memory.append((states, actions, rewards, next_states, dones))

    def sample_memory(self):
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_memory()

        states = [torch.FloatTensor(states[:, i, :]) for i in range(self.num_agents)]
        actions = [torch.FloatTensor(actions[:, i, :]) for i in range(self.num_agents)]
        rewards = [torch.FloatTensor(rewards[:, i]).unsqueeze(1) for i in range(self.num_agents)]
        next_states = [torch.FloatTensor(next_states[:, i, :]) for i in range(self.num_agents)]
        dones = [torch.FloatTensor(dones[:, i]).unsqueeze(1) for i in range(self.num_agents)]

        all_states = torch.cat(states, dim=1)
        all_next_states = torch.cat(next_states, dim=1)
        all_actions = torch.cat(actions, dim=1)

        for i in range(self.num_agents):
            # Update critic
            next_actions = [self.actors_target[j](next_states[j]) for j in range(self.num_agents)]
            next_actions = torch.cat(next_actions, dim=1)
            target_q = rewards[i] + (1 - dones[i]) * self.gamma * self.critics_target[i](all_next_states, next_actions)
            current_q = self.critics[i](all_states, all_actions)
            critic_loss = nn.MSELoss()(current_q, target_q.detach())

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # Update actor
            current_actions = [self.actors[j](states[j]) if j == i else self.actors[j](states[j]).detach() for j in
                               range(self.num_agents)]
            current_actions = torch.cat(current_actions, dim=1)
            actor_loss = -self.critics[i](all_states, current_actions).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # Update target networks
        for i in range(self.num_agents):
            for target_param, param in zip(self.actors_target[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.critics_target[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        state = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'actors_target': [actor_target.state_dict() for actor_target in self.actors_target],
            'critics_target': [critic_target.state_dict() for critic_target in self.critics_target],
        }
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(state['actors'][i])
            self.critics[i].load_state_dict(state['critics'][i])
            self.actors_target[i].load_state_dict(state['actors_target'][i])
            self.critics_target[i].load_state_dict(state['critics_target'][i])