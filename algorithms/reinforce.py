import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MetaPolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


class ReinforceAgent:
    def __init__(
        self,
        input_dim,
        n_actions,
        lr=1e-2,
        baseline_alpha=0.1,
        batch_size=16,
    ):
        self.policy = MetaPolicyNet(input_dim, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.baseline = 0.0
        self.baseline_alpha = baseline_alpha
        self.batch_size = batch_size
        self.buffer = []

    def select_action(self, state, temperature=1.0):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor)
        if temperature != 1.0:
            probs = torch.softmax(torch.log(probs) / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def record(self, log_prob, reward):
        self.buffer.append((log_prob, reward))
        if len(self.buffer) >= self.batch_size:
            return self.update()
        return None

    def update(self):
        rewards = np.array([r for _, r in self.buffer], dtype=np.float32)
        batch_mean = float(np.mean(rewards))
        self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * batch_mean

        loss = 0.0
        for log_prob, reward in self.buffer:
            advantage = reward - self.baseline
            loss -= log_prob * advantage

        loss = loss / len(self.buffer)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.buffer = []
        return float(loss.item())
