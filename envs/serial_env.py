try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - fallback for older setups
    import gym
import numpy as np
from gym import spaces


class SerialSupplyChain(gym.Env):
    """
    Linear 1-1-1 supply chain: Retailer <-> Wholesaler <-> Factory
    Meta-Agent controls Retailer. Upstream uses pass-through policy.
    """

    def __init__(self, config=None):
        if config is None:
            config = {}
        self.max_steps = int(config.get("max_steps", 52))

        self.h_cost = float(config.get("holding_cost", 1.0))
        self.p_cost = float(config.get("stockout_cost", 10.0))
        self.demand_std = float(config.get("demand_std", 2.0))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.steps = 0
        self.inventory = [20.0, 20.0, 100.0]
        self.pipeline = [0.0, 0.0]
        self.demand_history = []
        self.base_demand = float(np.random.uniform(5, 15))
        return self._get_obs()

    def _get_obs(self):
        return np.array(
            [self.inventory[0], self.inventory[1], self.inventory[2], self.base_demand],
            dtype=np.float32,
        )

    def step(self, action_retailer):
        action_retailer = float(np.squeeze(action_retailer))

        real_demand = max(0.0, float(np.random.normal(self.base_demand, self.demand_std)))
        self.demand_history.append(real_demand)

        sales = min(self.inventory[0], real_demand)
        backorder = real_demand - sales
        self.inventory[0] -= sales

        ship_to_retailer = min(self.inventory[1], action_retailer)
        self.inventory[1] -= ship_to_retailer
        self.inventory[0] += ship_to_retailer

        order_whole = ship_to_retailer
        ship_to_whole = min(self.inventory[2], order_whole)
        self.inventory[2] -= ship_to_whole
        self.inventory[1] += ship_to_whole

        self.inventory[2] += order_whole

        holding_cost = self.inventory[0] * self.h_cost
        stockout_cost = backorder * self.p_cost
        reward = -(holding_cost + stockout_cost)

        self.steps += 1
        done = self.steps >= self.max_steps

        info = {
            "demand": real_demand,
            "inventory": self.inventory[0],
            "action": action_retailer,
        }

        return self._get_obs(), reward, done, info
