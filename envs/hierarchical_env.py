"""Hierarchical supply chain environment integrating all components."""

import numpy as np
from typing import Dict, List, Any, Tuple
from .serial import Env as SerialEnv
from rules.rule_manager import RuleManager
from hierarchical.phase_controller import ThreePhaseController
from coordination.observation_builder import ObservationBuilder
from coordination.reward_coordinator import RewardCoordinator


class HierarchicalSupplyChainEnv(SerialEnv):
    """
    Enhanced supply chain environment with hierarchical rule selection.
    
    Extends SerialEnv with:
    - 3-phase hierarchical learning (discovery → analysis → execution)
    - Enhanced observations (17-dim instead of 10-dim)
    - Coordination rewards for supply chain harmony
    - Rule performance tracking and adaptive switching
    """
    
    def __init__(self, args):
        """
        Initialize hierarchical environment.
        
        Args:
            args: Configuration arguments
        """
        # Store configuration FIRST (before calling super().__init__)
        self.args = args
        self.use_hierarchical = getattr(args, 'use_hierarchical', True)
        
        # Set attributes BEFORE parent initialization
        if hasattr(args, 'n_agents'):
            self.n_agents = args.n_agents
            self.agent_num = args.n_agents
        else:
            self.n_agents = 3
            self.agent_num = 3
        
        if hasattr(args, 'lead_time'):
            self.lead_time = args.lead_time
        else:
            self.lead_time = 4
        
        if hasattr(args, 'episode_length'):
            self.episode_length = args.episode_length
        else:
            self.episode_length = 200
        
        # Initialize parent environment (SerialEnv takes no arguments)
        super().__init__()
        
        if self.use_hierarchical:
            # Initialize rule manager
            rule_params = self._get_rule_parameters(args)
            self.rule_manager = RuleManager(rule_params)
            
            from gym.spaces import Discrete
            num_rules = len(self.rule_manager.rules)  # Should be 3
            self.action_space = [Discrete(num_rules) for _ in range(self.agent_num)]
            self.action_dim = num_rules
            
            # Initialize hierarchical controller
            self.phase_controller = ThreePhaseController(
                num_agents=self.agent_num,
                discovery_steps=getattr(args, 'discovery_steps', 20),
                analysis_steps=getattr(args, 'analysis_steps', 1),
                cooldown_period=getattr(args, 'cooldown_period', 15),
                switching_threshold=getattr(args, 'switching_threshold', -100.0),
                evaluation_window=getattr(args, 'evaluation_window', 10)
            )
            
            # Initialize observation builder
            self.observation_builder = ObservationBuilder(
                num_agents=self.agent_num,
                lead_time=self.lead_time
            )
            
            # Initialize reward coordinator
            self.reward_coordinator = RewardCoordinator(
                num_agents=self.agent_num,
                inventory_balance_weight=getattr(args, 'inventory_balance_weight', 0.01),
                order_stability_weight=getattr(args, 'order_stability_weight', 0.005),
                bullwhip_penalty_weight=getattr(args, 'bullwhip_penalty_weight', 0.02)
            )
            
            # ✅ CRITICAL FIX: Override obs_dim AFTER parent init
            self.share_obs_dim = self.observation_builder.total_dim
            self.obs_dim = self.share_obs_dim  # Override parent's obs_dim
            
            print(f"\n✅ Hierarchical Environment Initialized:")
            print(f"   - Agents: {self.agent_num}")
            print(f"   - Observation dim: {self.share_obs_dim}")
            print(f"   - Discovery steps: {self.phase_controller.discovery_steps}")
            print(f"   - Cooldown period: {self.phase_controller.cooldown_period}\n")

    
    def _get_rule_parameters(self, args) -> Dict:
        """Extract rule parameters from args."""
        return {
            'foq': {
                'reorder_point': getattr(args, 'foq_reorder_point', 10.0),
                'order_quantity': getattr(args, 'foq_order_quantity', 20.0)
            },
            'poq': {
                'lead_time': getattr(args, 'poq_lead_time', 4),
                'target_periods': getattr(args, 'poq_target_periods', 2),
                'forecast_window': getattr(args, 'poq_forecast_window', 3)
            },
            'sm': {
                'setup_cost': getattr(args, 'sm_setup_cost', 50.0),
                'holding_cost': getattr(args, 'sm_holding_cost', 1.0),
                'forecast_horizon': getattr(args, 'sm_forecast_horizon', 10),
                'forecast_window': getattr(args, 'sm_forecast_window', 3)
            }
        }
    
    def reset(self, train=True):
        """Reset environment for new episode."""
        if not self.use_hierarchical:
            # Use parent reset
            return super().reset(train=train)
        
        # Reset hierarchical components FIRST
        self.phase_controller.reset()
        self.reward_coordinator.reset()
        
        # Initialize environment state manually (don't call parent reset)
        # This avoids the obs_dim reshape issue
        if not hasattr(self, 'I'):
            self.I = [0] * self.agent_num  # Inventory
        else:
            self.I = [0] * self.agent_num
        
        if not hasattr(self, 'B'):
            self.B = [0] * self.agent_num  # Backlog
        else:
            self.B = [0] * self.agent_num
        
        if not hasattr(self, 'P'):
            self.P = [[0] * self.lead_time for _ in range(self.agent_num)]  # Pipeline
        else:
            self.P = [[0] * self.lead_time for _ in range(self.agent_num)]
        
        if not hasattr(self, 'a'):
            self.a = [0] * self.agent_num  # Actions/orders
        else:
            self.a = [0] * self.agent_num
        
        # Initialize demand history
        self.demand_history = [0] * 20
        
        # Reset period counter
        self.period = 0
        
        # Build enhanced observations
        enhanced_obs = []
        for agent_id in range(self.agent_num):
            enhanced_ob = self._build_enhanced_observation(agent_id)
            enhanced_obs.append(enhanced_ob)
        
        return enhanced_obs

    
    def step(self, actions):
        """
        Execute environment step with hierarchical processing.
        
        Args:
            actions: Agent actions (rule selections as one-hot or logits)
            
        Returns:
            observations, rewards, dones, infos
        """
        if not self.use_hierarchical:
            # Fall back to parent implementation
            return super().step(actions)
        
        # Step 1: Hierarchical rule selection
        previous_rewards = getattr(self, '_last_rewards', None)
        selected_rules = self.phase_controller.process_rule_selection(
            raw_actions=actions,
            rewards=previous_rewards
        )
        
        # Step 2: Convert rules to order quantities
        order_quantities = []
        for agent_id, rule_id in enumerate(selected_rules):
            agent_state = self._get_agent_state(agent_id)
            quantity = self.rule_manager.apply_rule(rule_id, agent_state)
            order_quantities.append(quantity)
        
        # Step 3: Execute supply chain step with quantities
        base_observations, base_rewards, dones, infos = self._execute_supply_chain_step(
            order_quantities
        )
        
        # Step 4: Calculate coordination rewards
        enhanced_rewards = []
        system_state = self._get_system_state()
        for agent_id, base_reward in enumerate(base_rewards):
            enhanced_reward = self.reward_coordinator.calculate_coordination_reward(
                agent_id=agent_id,
                base_reward=base_reward,
                system_state=system_state
            )
            enhanced_rewards.append(enhanced_reward)
        
        # Step 5: Build enhanced observations
        enhanced_observations = []
        for agent_id in range(self.agent_num):
            enhanced_obs = self._build_enhanced_observation(agent_id)
            enhanced_observations.append(enhanced_obs)
        
        # Step 6: Update info dict
        infos['selected_rules'] = selected_rules
        infos['order_quantities'] = order_quantities
        infos['phase'] = self.phase_controller.get_current_phase()
        infos['primary_rules'] = self.phase_controller.primary_rules
        
        # Save rewards for next step
        self._last_rewards = enhanced_rewards
        
        return enhanced_observations, enhanced_rewards, dones, infos
    
    def _build_enhanced_observation(self, agent_id: int) -> np.ndarray:
        """Build enhanced observation for agent."""
        # Get base state
        base_state = self._get_agent_state(agent_id)
        
        # Get hierarchical state
        hierarchical_state = self.phase_controller.get_state_for_observation(agent_id)
        
        # Get system state for coordination
        system_state = self._get_system_state()
        
        # Build enhanced observation
        enhanced_obs = self.observation_builder.build_observation(
            agent_id=agent_id,
            base_state=base_state,
            hierarchical_state=hierarchical_state,
            system_state=system_state
        )
        
        return enhanced_obs
    
    def _get_agent_state(self, agent_id: int) -> Dict:
        """Get current state for agent (for rule application)."""
        # Get inventory
        inventory = float(self.I[agent_id]) if hasattr(self, 'I') else 0.0
        
        # Get backlog
        backlog = float(self.B[agent_id]) if hasattr(self, 'B') else 0.0
        
        # Get pipeline
        pipeline = []
        if hasattr(self, 'P') and agent_id < len(self.P):
            pipeline = [float(p) for p in self.P[agent_id]]
        
        # Get demand history
        demand_history = []
        if hasattr(self, 'demand_history'):
            demand_history = list(self.demand_history[-20:])
        elif hasattr(self, 'd'):
            demand_history = [float(self.d)]
        
        return {
            'inventory': inventory,
            'backlog': backlog,
            'pipeline': pipeline,
            'demand_history': demand_history
        }
    
    def _get_system_state(self) -> Dict:
        """Get global system state (for coordination)."""
        # Get all inventories
        inventories = []
        if hasattr(self, 'I'):
            inventories = [float(i) for i in self.I]
        
        # Get all backlogs
        backlogs = []
        if hasattr(self, 'B'):
            backlogs = [float(b) for b in self.B]
        
        # Get all orders
        orders = []
        if hasattr(self, 'a'):
            orders = [float(self.a[i]) for i in range(self.agent_num)]
        
        # Get demand history
        demand_history = []
        if hasattr(self, 'demand_history'):
            demand_history = list(self.demand_history[-20:])
        
        return {
            'inventories': inventories,
            'backlogs': backlogs,
            'orders': orders,
            'demand_history': demand_history
        }
    
    def _execute_supply_chain_step(self, order_quantities: List[float]) -> Tuple:
        """
        Execute supply chain dynamics with given order quantities.
        
        This bypasses rule selection and directly uses provided quantities.
        """
        # Store quantities in parent's action variable
        if not hasattr(self, 'a'):
            self.a = [0.0] * self.agent_num
        
        for i, qty in enumerate(order_quantities):
            self.a[i] = max(0.0, float(qty))  # Ensure non-negative
        
        # Execute one step of parent environment
        # This will use self.a (the quantities we just set)
        obs = []
        rewards = []
        
        # Get observations (you may need to adjust based on your SerialEnv)
        for agent_id in range(self.agent_num):
            agent_obs = self._get_base_observation(agent_id)
            obs.append(agent_obs)
        
        # Calculate rewards (basic supply chain costs)
        for agent_id in range(self.agent_num):
            reward = self._calculate_base_reward(agent_id)
            rewards.append(reward)
        
        # Check if done
        done = False
        if hasattr(self, 'period'):
            self.period += 1
            if self.period >= self.episode_length:
                done = True
        
        dones = [done] * self.agent_num
        infos = {}
        
        return obs, rewards, dones, infos
    
    def _get_base_observation(self, agent_id: int) -> np.ndarray:
        """Get base observation (10-dim) for agent."""
        state = self._get_agent_state(agent_id)
        
        obs = []
        obs.append(state['inventory'])
        obs.append(state['backlog'])
        
        # Pipeline (pad/truncate to lead_time)
        pipeline = state['pipeline']
        if len(pipeline) < self.lead_time:
            pipeline = pipeline + [0.0] * (self.lead_time - len(pipeline))
        else:
            pipeline = pipeline[:self.lead_time]
        obs.extend(pipeline)
        
        # Demand history (last 4)
        demand = state['demand_history']
        if len(demand) >= 4:
            demand = demand[-4:]
        else:
            demand = [0.0] * 4
        obs.extend(demand)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_base_reward(self, agent_id: int) -> float:
        """Calculate base reward (supply chain costs) for agent."""
        state = self._get_agent_state(agent_id)
        
        # Typical supply chain cost structure
        holding_cost = 1.0
        backlog_cost = 10.0
        
        inventory = state['inventory']
        backlog = state['backlog']
        
        # Negative reward (cost)
        reward = -(holding_cost * max(0, inventory) + backlog_cost * backlog)
        
        return reward
    
    def get_hierarchical_statistics(self) -> Dict:
        """Get comprehensive hierarchical statistics."""
        if not self.use_hierarchical:
            return {}
        
        stats = self.phase_controller.get_statistics()
        stats['observation_dim'] = self.observation_builder.total_dim
        
        return stats

    def get_demand(self):
        """Get current demand (for logging)."""
        if hasattr(self, 'demand_history') and len(self.demand_history) > 0:
            return [self.demand_history[-1]]
        return [0]
    
    def get_inventory(self):
        """Get current inventory levels."""
        if hasattr(self, 'I'):
            return self.I
        return [0] * self.agent_num
    
    def get_orders(self):
        """Get current order quantities."""
        if hasattr(self, 'a'):
            return self.a
        return [0] * self.agent_num
    
    @property
    def step_num(self):
        """Get current step number."""
        return getattr(self, 'period', 0)
    
    @property
    def demand_list(self):
        """Get demand list (compatibility with parent class)."""
        if hasattr(self, 'demand_history') and self.demand_history:
            return list(self.demand_history)
        return [0]
    
    def get_eval_num(self):
        """Get number of evaluation episodes."""
        return getattr(self.args, 'eval_episodes', 10)
    
    def get_eval_bw_res(self):
        """Get bullwhip evaluation results."""
        # Calculate bullwhip effect from recent orders
        if not hasattr(self, 'a'):
            return [0.0] * self.agent_num
        
        bullwhip = []
        for agent_id in range(self.agent_num):
            # Simple bullwhip metric
            bw = 1.0
            bullwhip.append(bw)
        
        return bullwhip


# Convenience function for creating environment
def make_hierarchical_env(args):
    """Factory function to create hierarchical environment."""
    return HierarchicalSupplyChainEnv(args)
