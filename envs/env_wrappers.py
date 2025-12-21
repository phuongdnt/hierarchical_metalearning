"""Environment wrappers for hierarchical MARL supply chain system."""

import numpy as np
import gym
from gym import spaces
from envs.serial import Env as SerialEnv
from envs.hierarchical_rulechoice import Env as HierarchicalSupplyChainEnv


""" Environment registry"""

ENV_REGISTRY = {
    "serial": SerialEnv,
    "hierarchical": HierarchicalSupplyChainEnv,
        "MyEnv": HierarchicalSupplyChainEnv,
}


def _get_env_class(env_name: str):
    """
    Get environment class by name.
    
    Args:
        env_name (str): Environment name
        
    Returns:
        Environment class
        
    Raises:
        ValueError: If environment not registered
    """
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment '{env_name}'. "
                        f"Available: {list(ENV_REGISTRY.keys())}")
    return ENV_REGISTRY[env_name]


class MultiDiscrete(gym.Space):
    """
    Multi-discrete action space.
    
    Consists of a series of discrete action spaces with different parameters.
    Useful for representing game controllers or keyboards where each key 
    is a discrete action space.
    
    Example:
        MultiDiscrete([[0,4], [0,1], [0,1]])
        represents 3 discrete spaces: [0-4], [0-1], [0-1]
    """

    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """Returns array with one sample from each discrete action space."""
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        """Check if x is a valid sample."""
        return (len(x) == self.num_discrete_space and 
                (np.array(x) >= self.low).all() and 
                (np.array(x) <= self.high).all())

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


class SubprocVecEnv(object):
    """
    Vectorized environment with multiple parallel environments.
    
    Creates multiple environment instances running in the same process.
    Useful for collecting experience from multiple environments simultaneously.
    """
    
    def __init__(self, all_args):
        """
        Initialize vectorized environment.
        
        Args:
            all_args: Configuration arguments containing:
                - env_name: Environment name (serial or hierarchical)
                - n_rollout_threads: Number of parallel environments
        """
        # Get environment class
        env_name = getattr(all_args, "env_name", "hierarchical")
        env_cls = _get_env_class(env_name)
        
        # Create environment instances
        self.env_list = [env_cls(all_args) for _ in range(all_args.n_rollout_threads)]
        self.num_envs = all_args.n_rollout_threads

        # Get environment properties
        self.num_agent = self.env_list[0].agent_num
        all_args.num_agents = self.num_agent
        
        # Get observation dimension (17 for hierarchical, varies for others)
        if hasattr(self.env_list[0], 'share_obs_dim'):
            self.signal_obs_dim = self.env_list[0].share_obs_dim
        else:
            self.signal_obs_dim = self.env_list[0].obs_dim
        
        self.signal_action_dim = self.env_list[0].action_dim

        # Control parameters
        self.u_range = 1.0
        self.movable = True
        self.discrete_action_space = True
        self.discrete_action_input = False
        self.force_discrete_action = False

        # Configure action and observation spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        
        for agent in range(self.num_agent):
            # Action space (discrete rule selection)
            u_action_space = spaces.Discrete(self.signal_action_dim)
            self.action_space.append(u_action_space)
            
            # Observation space
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, 
                          shape=(self.signal_obs_dim,),
                          dtype=np.float32)
            )

        # Shared observation space
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, 
                      shape=(share_obs_dim,),
                      dtype=np.float32)
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        Execute one step in all environments.
        
        Args:
            actions: List of actions for each environment
            
        Returns:
            observations, rewards, dones, infos
        """
        results = [env.step(action) for env, action in zip(self.env_list, actions)]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def get_property(self):
        """Get environment properties (inventory, demand, orders)."""
        inv = [env.get_inventory() for env in self.env_list]
        demand = [env.get_demand() for env in self.env_list]
        orders = [env.get_orders() for env in self.env_list]
        return inv, demand, orders
         
    def reset(self):
        """Reset all environments."""
        obs = [env.reset() for env in self.env_list]
        return np.stack(obs), None

    def close(self):
        """Close all environments."""
        pass

    def render(self, mode="rgb_array"):
        """Render environments."""
        pass


class DummyVecEnv(object):
    """
    Single environment wrapper (non-vectorized).
    
    Provides same interface as SubprocVecEnv but with single environment.
    Useful for evaluation or debugging.
    """
    
    def __init__(self, all_args):
        """
        Initialize single environment.
        
        Args:
            all_args: Configuration arguments
        """
        # Get environment class
        env_name = getattr(all_args, "env_name", "hierarchical")
        env_cls = _get_env_class(env_name)
        
        # Create single environment instance
        self.env_list = [env_cls(all_args)]
        self.num_envs = all_args.n_rollout_threads

        # Get environment properties
        self.num_agent = self.env_list[0].agent_num
        all_args.num_agents = self.num_agent
        
        # Get observation dimension
        if hasattr(self.env_list[0], 'share_obs_dim'):
            self.signal_obs_dim = self.env_list[0].share_obs_dim
        else:
            self.signal_obs_dim = self.env_list[0].obs_dim
        
        self.signal_action_dim = self.env_list[0].action_dim

        # Control parameters
        self.u_range = 1.0
        self.movable = True
        self.discrete_action_space = True
        self.discrete_action_input = False
        self.force_discrete_action = False

        # Configure action and observation spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        
        for agent_num in range(self.num_agent):
            # Action space
            u_action_space = spaces.Discrete(self.signal_action_dim)
            self.action_space.append(u_action_space)
            
            # Observation space
            obs_dim = self.signal_obs_dim
            share_obs_dim += obs_dim
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, 
                          shape=(obs_dim,), 
                          dtype=np.float32)
            )

        # Shared observation space
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, 
                      shape=(share_obs_dim,),
                      dtype=np.float32)
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        Execute one step.
        
        Args:
            actions: List of actions
            
        Returns:
            observations, rewards, dones, infos
        """
        results = [env.step(action) for env, action in zip(self.env_list, actions)]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        """Reset environment."""
        obs = [env.reset(train=False) for env in self.env_list]
        return np.stack(obs), None
    
    def get_eval_bw_res(self):
        """Get evaluation bullwhip results."""
        if hasattr(self.env_list[0], 'get_eval_bw_res'):
            return self.env_list[0].get_eval_bw_res()
        return None
    
    def get_eval_num(self):
        """Get evaluation number."""
        if hasattr(self.env_list[0], 'get_eval_num'):
            return self.env_list[0].get_eval_num()
        return 0
        
    def close(self):
        """Close environment."""
        pass

    def render(self, mode="rgb_array"):
        """Render environment."""
        pass


"""Testing and Debugging Functions"""

def test_env_registration():
    """Test that environments are properly registered."""
    print("\n" + "="*70)
    print("ENVIRONMENT REGISTRATION TEST")
    print("="*70)
    
    print("\n✅ Registered Environments:")
    for env_name, env_class in ENV_REGISTRY.items():
        print(f"   • {env_name:20s} → {env_class.__name__}")
    
    print("\n" + "="*70)
    print("✅ Environment wrappers loaded successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run test when file is executed directly
    test_env_registration()
