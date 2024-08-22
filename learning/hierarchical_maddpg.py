import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

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
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim, num_options, actor_lr, critic_lr, gamma, tau, device):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.noise_std = 0.1

        self.actors = [Actor(state_dim, action_dim, hidden_dim).to(device) for _ in range(num_agents)]
        self.critics = [Critic(state_dim, action_dim * num_agents, hidden_dim).to(device) for _ in range(num_agents)]
        self.actors_target = [Actor(state_dim, action_dim, hidden_dim).to(device) for _ in range(num_agents)]
        self.critics_target = [Critic(state_dim, action_dim * num_agents, hidden_dim).to(device) for _ in range(num_agents)]
        self.high_level_policies = [HighLevelPolicy(state_dim, num_options, hidden_dim).to(device) for _ in range(num_agents)]

        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=actor_lr) for i in range(num_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=critic_lr) for i in range(num_agents)]
        self.high_level_optimizers = [optim.Adam(self.high_level_policies[i].parameters(), lr=actor_lr) for i in range(num_agents)]

        for i in range(num_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

        self.memory = SumTree(100000)  # Replace deque with SumTree
        self.batch_size = 64
        self.PER_e = 0.01  # Small amount to avoid zero priority
        self.PER_a = 0.6  # Hyperparameter for prioritization
        self.PER_b = 0.4  # Initial importance-sampling weight
        self.PER_b_increment = 0.001  # Importance-sampling weight increment per sampling
        self.absolute_error_upper = 1.  # Clipping for error

    def select_action(self, state, explore=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        actions = []
        for i in range(self.num_agents):
            action = self.actors[i](state).squeeze(0).cpu().detach().numpy()
            if explore:
                action += np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action, -1, 1)
            actions.append(action)
        return actions

    def select_option(self, state, agent_index):
        option_probs = torch.softmax(self.high_level_policies[agent_index](torch.FloatTensor(state)), dim=-1)
        return torch.multinomial(option_probs, 1).item()

    def store_transition(self, state, action, reward, next_state, done):
        state = np.array(state).flatten()
        action = np.array(action).flatten()
        next_state = np.array(next_state).flatten()
        experience = (state, action, reward, next_state, done)
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.memory.add(max_priority, experience)

    def update(self):
        if self.memory.n_entries < self.batch_size:
            return

        batch, idxs, is_weights = self._sample_batch(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*batch))

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        # Update critics
        all_target_actions = torch.cat([self.actors_target[i](next_state_batch) for i in range(self.num_agents)], dim=1)

        for i in range(self.num_agents):
            target_q = reward_batch + self.gamma * (1 - done_batch) * \
                       self.critics_target[i](next_state_batch, all_target_actions)
            current_q = self.critics[i](state_batch, action_batch)
            td_error = torch.abs(target_q - current_q).detach().cpu().numpy()
            self._update_priorities(idxs, td_error)
            critic_loss = (is_weights * nn.MSELoss(reduction='none')(current_q, target_q.detach())).mean()
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # Update actors
        for i in range(self.num_agents):
            current_policy_actions = self.actors[i](state_batch)
            all_actions = action_batch.clone()
            all_actions[:, i * self.action_dim:(i + 1) * self.action_dim] = current_policy_actions
            actor_loss = -self.critics[i](state_batch, all_actions).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # Soft update target networks
        self.soft_update_targets()

    def _sample_batch(self, batch_size):
        batch = []
        idxs = []
        segment = self.memory.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.memory.get(s)
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.memory.total()
        is_weights = np.power(self.memory.n_entries * sampling_probabilities, -self.PER_b)
        is_weights /= is_weights.max()
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment])

        return batch, idxs, is_weights

    def _update_priorities(self, idxs, td_errors):
        td_errors += self.PER_e
        clipped_errors = np.minimum(td_errors, self.absolute_error_upper)
        priorities = np.power(clipped_errors, self.PER_a)
        for idx, priority in zip(idxs, priorities):
            self.memory.update(idx, priority)

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

    def train(self):
        self.update()

    def reset_noise(self):
        self.noise_std = 0.1

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
            'batch_size': self.batch_size,
            'PER_e': self.PER_e,
            'PER_a': self.PER_a,
            'PER_b': self.PER_b,
            'PER_b_increment': self.PER_b_increment,
            'absolute_error_upper': self.absolute_error_upper
        }

    def set_params(self, params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_memory_size(self):
        return self.memory.n_entries

    def clear_memory(self):
        self.memory = SumTree(self.memory.capacity)

    def update_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size

    def get_critic_loss(self, agent_index, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
        done_tensor = torch.FloatTensor([float(done)]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            next_action = self.actors_target[agent_index](next_state_tensor)
            target_q = reward_tensor + self.gamma * (1 - done_tensor) * self.critics_target[agent_index](
                next_state_tensor, next_action)

        current_q = self.critics[agent_index](state_tensor, action_tensor)
        critic_loss = nn.MSELoss()(current_q, target_q)
        return critic_loss.item()

    def get_actor_loss(self, agent_index, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
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

    def update_target_networks(self):
        for i in range(self.num_agents):
            for target_param, param in zip(self.actors_target[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.critics_target[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def adjust_learning_rates(self, actor_lr, critic_lr):
        for optimizer in self.actor_optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = actor_lr
        for optimizer in self.critic_optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = critic_lr

    def get_exploration_rate(self):
        return self.noise_std

    def set_exploration_rate(self, rate):
        self.noise_std = rate

    def update_per_parameters(self, PER_e=None, PER_a=None, PER_b=None, PER_b_increment=None):
        if PER_e is not None:
            self.PER_e = PER_e
        if PER_a is not None:
            self.PER_a = PER_a
        if PER_b is not None:
            self.PER_b = PER_b
        if PER_b_increment is not None:
            self.PER_b_increment = PER_b_increment

    def get_td_error(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([float(done)]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            next_actions = [self.actors_target[i](next_state) for i in range(self.num_agents)]
            next_actions = torch.cat(next_actions, dim=1)
            target_q = reward + self.gamma * (1 - done) * self.critics_target[0](next_state, next_actions)
            current_q = self.critics[0](state, action)
            td_error = torch.abs(target_q - current_q).item()

        return td_error

    def get_high_level_policy_loss(self, agent_index, state, option, reward):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        option_probs = self.high_level_policies[agent_index](state)
        log_prob = torch.log_softmax(option_probs, dim=-1)[0, option]
        loss = -log_prob * reward
        return loss.item()

    def reset(self):
        self.noise_std = 0.1
        self.PER_b = 0.4
        for i in range(self.num_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())