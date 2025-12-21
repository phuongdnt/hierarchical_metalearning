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

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


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
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_beta = 0.05
        self.entropy_coef = 0.01

    def select_action(self, state, temperature=1.0):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy(state_tensor)
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy

    def record(self, log_prob, entropy, reward):
        self.buffer.append((log_prob, entropy, reward))
        if len(self.buffer) >= self.batch_size:
            return self.update()
        return None

    def update(self):
        rewards = np.array([r for _, _, r in self.buffer], dtype=np.float32)
        batch_mean = float(np.mean(rewards))
        self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * batch_mean

        batch_var = float(np.var(rewards))
        self.reward_mean = (1 - self.reward_beta) * self.reward_mean + self.reward_beta * batch_mean
        self.reward_var = (1 - self.reward_beta) * self.reward_var + self.reward_beta * batch_var
        reward_std = float(np.sqrt(self.reward_var)) + 1e-6

        loss = 0.0
        for log_prob, entropy, reward in self.buffer:
            advantage = (reward - self.baseline) / reward_std
            loss -= log_prob * advantage
            loss -= self.entropy_coef * entropy

        loss = loss / len(self.buffer)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.buffer = []
        return float(loss.item())
