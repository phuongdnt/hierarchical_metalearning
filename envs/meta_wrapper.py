try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - fallback for older setups
    import gym
import numpy as np

from rules.inventory_rules import EOQRule, FOQRule, POQRule, SilverMealRule


class MetaSerialEnv(gym.Env):
    """
    Meta-learning wrapper:
    - Action: rule index
    - Step: runs a full episode of SerialSupplyChain using that rule
    - Reward: total episode reward
    """

    def __init__(self, serial_env):
        self.env = serial_env
        self.rules = [
            FOQRule(reorder_point=10, quantity=15),
            FOQRule(reorder_point=20, quantity=30),
            POQRule(target_level=25),
            POQRule(target_level=40),
            SilverMealRule(),
            EOQRule(),
        ]

        self.action_space = gym.spaces.Discrete(len(self.rules))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()
        return self._get_meta_state(obs)

    def step(self, rule_idx):
        selected_rule = self.rules[int(rule_idx)]

        total_reward = 0.0
        done = False
        obs = self.env._get_obs()
        history_inv = []

        while not done:
            retailer_inv = obs[0]
            demand_hist = self.env.demand_history

            action_qty = selected_rule.get_action(retailer_inv, demand_hist)

            obs, reward, done, info = self.env.step(action_qty)
            total_reward += float(reward)
            history_inv.append(info["inventory"])

        info = {
            "rule_name": selected_rule.name,
            "avg_inv": float(np.mean(history_inv)) if history_inv else 0.0,
            "final_demand_mean": self.env.base_demand,
        }

        return self._get_meta_state(obs), total_reward, True, info

    def _get_meta_state(self, obs):
        inventory = obs[:3]
        if len(self.env.demand_history) >= 2:
            demand_mean = float(np.mean(self.env.demand_history[-10:]))
            demand_std = float(np.std(self.env.demand_history[-10:]))
        else:
            demand_mean = float(self.env.base_demand)
            demand_std = 0.0

        return np.array(
            [
                inventory[0],
                inventory[1],
                inventory[2],
                demand_mean,
                demand_std,
            ],
            dtype=np.float32,
        )
