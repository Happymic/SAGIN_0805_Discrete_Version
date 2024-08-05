import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.optimizer = optim.Adam(self.parameters())
        self.loss_fn = nn.MSELoss()

    def forward(self, state):
        return self.network(state)

    def act(self, state, epsilon=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)  # 添加 batch 维度
        if np.random.random() < epsilon:
            return np.random.randint(0, self.network[-1].out_features)
        with torch.no_grad():
            q_values = self(state)
            return q_values.argmax().item()
    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])
        action = torch.LongTensor([action])

        q_values = self(state)
        next_q_values = self(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + 0.99 * next_q_value * (1 - done)

        loss = self.loss_fn(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()