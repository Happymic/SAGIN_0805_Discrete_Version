import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)

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
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

class HighLevelPolicy(nn.Module):
    def __init__(self, state_dim, num_options, hidden_dim):
        super(HighLevelPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )

    def forward(self, state):
        return self.network(state)

class HierarchicalMADDPG:
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim, num_options, actor_lr, critic_lr, gamma, tau):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        self.gamma = gamma
        self.tau = tau
        self.noise_std = 0.1

        self.actors = [Actor(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        self.critics = [Critic(state_dim * num_agents, action_dim * num_agents, hidden_dim) for _ in range(num_agents)]
        self.actors_target = [Actor(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        self.critics_target = [Critic(state_dim * num_agents, action_dim * num_agents, hidden_dim) for _ in range(num_agents)]
        self.high_level_policies = [HighLevelPolicy(state_dim, num_options, hidden_dim) for _ in range(num_agents)]

        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=actor_lr) for i in range(num_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=critic_lr) for i in range(num_agents)]
        self.high_level_optimizers = [optim.Adam(self.high_level_policies[i].parameters(), lr=actor_lr) for i in range(num_agents)]

        for i in range(num_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

        self.memory = deque(maxlen=100000)
        self.batch_size = 64
    def select_action(self, state, explore=True):
        actions = []
        state = np.array(state).flatten()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        for i in range(self.num_agents):
            action = self.actors[i](state_tensor).squeeze(0).detach().numpy()
            if explore:
                action += np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action, -1, 1)
            # 确保动作不为零
            action += np.random.uniform(-0.1, 0.1, size=action.shape)
            actions.append(action)
        return actions

    def select_option(self, state, agent_index):
        option_probs = torch.softmax(self.high_level_policies[agent_index](torch.FloatTensor(state)), dim=-1)
        return torch.multinomial(option_probs, 1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*batch))

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)

        # Update critics
        all_target_actions = []
        for i in range(self.num_agents):
            all_target_actions.append(self.actors_target[i](next_state_batch[:, i]))
        all_target_actions = torch.cat(all_target_actions, dim=1)

        for i in range(self.num_agents):
            target_q = reward_batch[:, i].unsqueeze(1) + self.gamma * (1 - done_batch[:, i].unsqueeze(1)) * \
                       self.critics_target[i](next_state_batch, all_target_actions)
            current_q = self.critics[i](state_batch, action_batch)
            critic_loss = nn.MSELoss()(current_q, target_q.detach())
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # Update actors
        for i in range(self.num_agents):
            current_policy_actions = self.actors[i](state_batch[:, i])
            all_actions = action_batch.clone()
            all_actions[:, i * self.action_dim:(i + 1) * self.action_dim] = current_policy_actions
            actor_loss = -self.critics[i](state_batch, all_actions).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # Soft update target networks
        self.soft_update_targets()

    def soft_update_targets(self):
        for i in range(self.num_agents):
            for target_param, param in zip(self.actors_target[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.critics_target[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        state = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'high_level_policies': [policy.state_dict() for policy in self.high_level_policies],
            'actors_target': [actor_target.state_dict() for actor_target in self.actors_target],
            'critics_target': [critic_target.state_dict() for critic_target in self.critics_target],
        }
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(state['actors'][i])
            self.critics[i].load_state_dict(state['critics'][i])
            self.high_level_policies[i].load_state_dict(state['high_level_policies'][i])
            self.actors_target[i].load_state_dict(state['actors_target'][i])
            self.critics_target[i].load_state_dict(state['critics_target'][i])
        logger.info(f"Model loaded from {filename}")

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        self.update()

    def reset_noise(self):
        self.noise_std = 0.1  # 重置噪声标准差

    def decay_noise(self, decay_factor=0.99):
        self.noise_std *= decay_factor

    def get_high_level_action(self, state):
        high_level_actions = []
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        for i in range(self.num_agents):
            option_probs = torch.softmax(self.high_level_policies[i](state_tensor), dim=-1)
            high_level_action = torch.multinomial(option_probs, 1).item()
            high_level_actions.append(high_level_action)
        return high_level_actions

    def update_high_level_policy(self, states, high_level_actions, rewards):
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0)
            action = high_level_actions[i]
            reward = rewards[i]

            option_probs = self.high_level_policies[i](state)
            log_prob = torch.log_softmax(option_probs, dim=-1)[0, action]
            loss = -log_prob * reward

            self.high_level_optimizers[i].zero_grad()
            loss.backward()
            self.high_level_optimizers[i].step()

    def get_params(self):
        return {
            'num_agents': self.num_agents,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'num_options': self.num_options,
            'gamma': self.gamma,
            'tau': self.tau,
            'noise_std': self.noise_std,
            'batch_size': self.batch_size
        }

    def set_params(self, params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        logger.info("HMADDPG parameters updated")

    def get_memory_size(self):
        return len(self.memory)

    def clear_memory(self):
        self.memory.clear()
        logger.info("Replay memory cleared")

    def update_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
        logger.info(f"Batch size updated to {new_batch_size}")

    def get_critic_loss(self, agent_index, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0)
        done_tensor = torch.FloatTensor([float(done)]).unsqueeze(0)

        with torch.no_grad():
            next_action = self.actors_target[agent_index](next_state_tensor)
            target_q = reward_tensor + self.gamma * (1 - done_tensor) * self.critics_target[agent_index](
                next_state_tensor, next_action)

        current_q = self.critics[agent_index](state_tensor, action_tensor)
        critic_loss = nn.MSELoss()(current_q, target_q)
        return critic_loss.item()

    def get_actor_loss(self, agent_index, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actors[agent_index](state_tensor)
        actor_loss = -self.critics[agent_index](state_tensor, action).mean()
        return actor_loss.item()

    def save_checkpoint(self, filename):
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'high_level_policies': [policy.state_dict() for policy in self.high_level_policies],
            'actors_target': [actor_target.state_dict() for actor_target in self.actors_target],
            'critics_target': [critic_target.state_dict() for critic_target in self.critics_target],
            'actor_optimizers': [optimizer.state_dict() for optimizer in self.actor_optimizers],
            'critic_optimizers': [optimizer.state_dict() for optimizer in self.critic_optimizers],
            'high_level_optimizers': [optimizer.state_dict() for optimizer in self.high_level_optimizers],
            'params': self.get_params()
        }
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
            self.high_level_policies[i].load_state_dict(checkpoint['high_level_policies'][i])
            self.actors_target[i].load_state_dict(checkpoint['actors_target'][i])
            self.critics_target[i].load_state_dict(checkpoint['critics_target'][i])
            self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizers'][i])
            self.critic_optimizers[i].load_state_dict(checkpoint['critic_optimizers'][i])
            self.high_level_optimizers[i].load_state_dict(checkpoint['high_level_optimizers'][i])
        self.set_params(checkpoint['params'])
        logger.info(f"Checkpoint loaded from {filename}")