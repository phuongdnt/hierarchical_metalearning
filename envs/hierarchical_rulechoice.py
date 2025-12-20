"""
HIERARCHICAL RULE CHOICE ENVIRONMENT

Kết hợp:
1. 3-phase hierarchical learning (Discovery → Analysis → Execution) ✅
2. Liu's serial.py dynamics 100% ✅
3. Rule logic hợp lý (không order dư) ✅
4. Coordination rewards ✅

Flow:
Neural Network → Rule ID → Phase Controller → Rule Logic → Order Qty → Liu's step()
"""

import numpy as np
from gym import spaces
from typing import Dict, List, Any

# Import Liu's environment
try:
    from envs.serial import (
        Env as LiuSerialEnv,
        LEVEL_NUM, LEAD_TIME, H, B, S_I, S_O, ALPHA
    )
except ImportError:
    LEVEL_NUM = 3
    LEAD_TIME = 4
    H = [1, 1, 1]
    B = [10, 10, 10]
    S_I = 0
    S_O = 0
    ALPHA = 0.5
    from envs.serial import Env as LiuSerialEnv


class ThreePhaseController:
    """
    3-Phase Hierarchical Learning Controller.
    
    Phase 1: DISCOVERY - Explore all rules, collect performance data
    Phase 2: ANALYSIS - Select primary rule for each agent
    Phase 3: EXECUTION - Use primary rule with adaptive switching
    """
    
    def __init__(self, num_agents: int,
                 discovery_steps: int = 20,
                 cooldown_period: int = 15,
                 switching_threshold: float = -150.0,
                 evaluation_window: int = 10):
        
        self.num_agents = num_agents
        self.discovery_steps = discovery_steps
        self.cooldown_period = cooldown_period
        self.switching_threshold = switching_threshold
        self.evaluation_window = evaluation_window
        
        self.reset()
    
    def reset(self):
        """Reset controller for new episode."""
        self.current_step = 0
        self.phase = "discovery"  # discovery, analysis, execution
        
        # Performance tracking per rule per agent
        self.rule_rewards = [[[] for _ in range(3)] for _ in range(self.num_agents)]
        
        # Primary rules (selected after discovery)
        self.primary_rules = [0] * self.num_agents  # Default to FOQ
        
        # Execution phase tracking
        self.steps_since_switch = [0] * self.num_agents
        self.recent_rewards = [[] for _ in range(self.num_agents)]
        
        # Discovery: which rules to try
        self.discovery_schedule = self._create_discovery_schedule()
        self.discovery_idx = 0
    
    def _create_discovery_schedule(self) -> List[List[int]]:
        """
        Create schedule for rule exploration during discovery.
        Ensures each rule is tried multiple times.
        """
        schedule = []
        rules_per_cycle = 3
        cycles = self.discovery_steps // rules_per_cycle
        
        for cycle in range(cycles):
            for rule_id in range(3):
                # All agents try same rule (simpler coordination)
                schedule.append([rule_id] * self.num_agents)
        
        # Fill remaining steps
        while len(schedule) < self.discovery_steps:
            rule_id = len(schedule) % 3
            schedule.append([rule_id] * self.num_agents)
        
        return schedule
    
    def process_actions(self, raw_actions: List, rewards: List = None) -> List[int]:
        """
        Process actions through hierarchical controller.
        
        Args:
            raw_actions: Actions from neural network (rule selections)
            rewards: Rewards from previous step (for tracking)
            
        Returns:
            Final rule selections for each agent
        """
        # Update with previous rewards
        if rewards is not None:
            self._update_rewards(rewards)
        
        self.current_step += 1
        
        # Phase transitions
        if self.phase == "discovery" and self.current_step >= self.discovery_steps:
            self._transition_to_analysis()
        
        # Get actions based on phase
        if self.phase == "discovery":
            return self._discovery_action()
        elif self.phase == "analysis":
            self._transition_to_execution()
            return self.primary_rules.copy()
        else:  # execution
            return self._execution_action(raw_actions)
    
    def _update_rewards(self, rewards: List):
        """Track rewards for performance evaluation."""
        for agent_id, reward in enumerate(rewards):
            reward_val = float(reward[0]) if isinstance(reward, (list, np.ndarray)) else float(reward)
            
            # Track for recent window
            self.recent_rewards[agent_id].append(reward_val)
            if len(self.recent_rewards[agent_id]) > self.evaluation_window:
                self.recent_rewards[agent_id].pop(0)
            
            # Track per rule (during discovery)
            if self.phase == "discovery" and self.discovery_idx > 0:
                prev_rules = self.discovery_schedule[self.discovery_idx - 1]
                rule_id = prev_rules[agent_id]
                self.rule_rewards[agent_id][rule_id].append(reward_val)
    
    def _discovery_action(self) -> List[int]:
        """Get action during discovery phase."""
        if self.discovery_idx < len(self.discovery_schedule):
            actions = self.discovery_schedule[self.discovery_idx]
            self.discovery_idx += 1
            return actions
        return [0] * self.num_agents
    
    def _transition_to_analysis(self):
        """Transition from discovery to analysis phase."""
        self.phase = "analysis"
        
        # Select best rule for each agent based on discovery data
        for agent_id in range(self.num_agents):
            best_rule = 0
            best_avg = float('-inf')
            
            for rule_id in range(3):
                rewards = self.rule_rewards[agent_id][rule_id]
                if rewards:
                    avg = np.mean(rewards)
                    if avg > best_avg:
                        best_avg = avg
                        best_rule = rule_id
            
            self.primary_rules[agent_id] = best_rule
        
        print(f"\n🎯 PHASE TRANSITION: Discovery → Analysis")
        print(f"   Primary rules selected: {['FOQ', 'POQ', 'SM'][r] for r in self.primary_rules}")
    
    def _transition_to_execution(self):
        """Transition from analysis to execution phase."""
        self.phase = "execution"
        self.steps_since_switch = [0] * self.num_agents
        print(f"   Entering Execution Phase\n")
    
    def _execution_action(self, raw_actions: List) -> List[int]:
        """
        Get action during execution phase.
        
        Uses primary rule but can switch if performance is poor.
        """
        final_actions = []
        
        for agent_id in range(self.num_agents):
            # Increment cooldown counter
            self.steps_since_switch[agent_id] += 1
            
            # Check if switch is allowed (cooldown passed)
            if self.steps_since_switch[agent_id] < self.cooldown_period:
                # Still in cooldown, use primary rule
                final_actions.append(self.primary_rules[agent_id])
            else:
                # Check performance
                recent = self.recent_rewards[agent_id]
                if recent and np.mean(recent) < self.switching_threshold:
                    # Performance poor, allow network to choose
                    action = self._extract_action(raw_actions[agent_id])
                    if action != self.primary_rules[agent_id]:
                        self.steps_since_switch[agent_id] = 0  # Reset cooldown
                        self.primary_rules[agent_id] = action  # Update primary
                    final_actions.append(action)
                else:
                    # Performance OK, use primary rule
                    final_actions.append(self.primary_rules[agent_id])
        
        return final_actions
    
    def _extract_action(self, action) -> int:
        """Extract rule ID from action (handles various formats)."""
        if isinstance(action, np.ndarray):
            if action.shape == () or len(action.shape) == 0:
                return int(action) % 3
            else:
                return int(np.argmax(action))
        elif isinstance(action, (list, tuple)):
            return int(np.argmax(action))
        else:
            return int(action) % 3
    
    def get_phase(self) -> str:
        """Get current phase name."""
        return self.phase
    
    def get_state_for_observation(self, agent_id: int) -> Dict:
        """Get hierarchical state for enhanced observation."""
        phase_encoding = {
            "discovery": [1.0, 0.0, 0.0],
            "analysis": [0.0, 1.0, 0.0],
            "execution": [0.0, 0.0, 1.0]
        }
        
        primary_rule_onehot = [0.0, 0.0, 0.0]
        primary_rule_onehot[self.primary_rules[agent_id]] = 1.0
        
        return {
            "phase": phase_encoding.get(self.phase, [0.0, 0.0, 1.0]),
            "primary_rule": primary_rule_onehot,
            "steps_since_switch": self.steps_since_switch[agent_id] / self.cooldown_period,
        }


class Env(LiuSerialEnv):
    """
    Hierarchical Rule Choice Environment.
    
    Combines:
    - 3-phase hierarchical learning
    - Liu's serial.py dynamics (100% accurate)
    - Smart rule logic (no over-ordering)
    """
    
    def __init__(self):
        # Initialize parent (Liu's SerialEnv)
        super().__init__()
        
        # Override action space: 3 rules
        self.action_dim = 3
        self.action_space = [spaces.Discrete(3) for _ in range(self.agent_num)]
        
        # Rule parameters
        self.rule_params = {
            'foq': {
                'reorder_point': 15.0,
                'order_quantity': 25.0,
            },
            'poq': {
                'safety_periods': 1.5,
                'coverage_periods': 4,
            },
            'sm': {
                'safety_periods': 1.5,
                'coverage_periods': 3,
            }
        }
        
        # Initialize hierarchical controller
        self.phase_controller = ThreePhaseController(
            num_agents=self.agent_num,
            discovery_steps=20,
            cooldown_period=15,
            switching_threshold=-150.0,
            evaluation_window=10
        )
        
        # Tracking
        self.rule_counts = [[0, 0, 0] for _ in range(self.agent_num)]
        self._demand_history = [10.0] * 10
        self._last_rewards = None
        
        print(f"\n{'='*60}")
        print(f"✅ HIERARCHICAL RULE CHOICE ENVIRONMENT")
        print(f"{'='*60}")
        print(f"Agents: {self.agent_num}")
        print(f"Action space: Discrete(3) [FOQ, POQ, SM]")
        print(f"Dynamics: 100% Liu's serial.py")
        print(f"3-Phase: Discovery({self.phase_controller.discovery_steps}) → Analysis → Execution")
        print(f"Cooldown: {self.phase_controller.cooldown_period} steps")
        print(f"{'='*60}\n")
    
    def reset(self, train=True, normalize=True):
        """Reset environment."""
        obs = super().reset(train=train, normalize=normalize)
        
        # Reset hierarchical controller
        self.phase_controller.reset()
        
        # Reset tracking
        self.rule_counts = [[0, 0, 0] for _ in range(self.agent_num)]
        self._demand_history = [10.0] * 10
        self._last_rewards = None
        
        return obs
    
    def step(self, actions, one_hot=True):
        """
        Execute step with hierarchical rule selection.
        
        Flow:
        1. Neural network outputs → Phase controller → Final rule selection
        2. Rule logic → Order quantity
        3. Liu's state_update → Rewards, next state
        """
        # Convert one-hot to indices
        if one_hot:
            action_indices = [np.argmax(a) for a in actions]
        else:
            action_indices = list(actions)
        
        # Step 1: Process through hierarchical controller
        selected_rules = self.phase_controller.process_actions(
            raw_actions=action_indices,
            rewards=self._last_rewards
        )
        
        # Step 2: Update demand history
        if hasattr(self, 'demand_list') and self.step_num < len(self.demand_list):
            self._demand_history.append(float(self.demand_list[self.step_num]))
            if len(self._demand_history) > 20:
                self._demand_history = self._demand_history[-20:]
        
        # Step 3: Convert rules to order quantities
        order_quantities = self._rules_to_orders(selected_rules)
        
        # Track rule usage
        for agent_id, rule_id in enumerate(selected_rules):
            self.rule_counts[agent_id][rule_id] += 1
        
        # Step 4: Call Liu's state_update with our quantities
        rewards = self.state_update(order_quantities)
        
        # Store for logging
        self.current_orders = order_quantities
        
        # Get observations
        sub_agent_obs = self.get_step_obs(order_quantities)
        
        # Process rewards (Liu's method)
        sub_agent_reward = self.get_processed_rewards(rewards)
        
        # Save for next step
        self._last_rewards = sub_agent_reward
        
        # Check done
        if self.step_num >= self.episode_max_steps:
            sub_agent_done = [True] * self.agent_num
        else:
            sub_agent_done = [False] * self.agent_num
        
        # Build info
        sub_agent_info = []
        for agent_id in range(self.agent_num):
            info = {
                'chosen_rule': selected_rules[agent_id],
                'order_quantity': order_quantities[agent_id],
                'phase': self.phase_controller.get_phase(),
                'primary_rule': self.phase_controller.primary_rules[agent_id],
            }
            
            if sub_agent_done[0]:
                info['episode_metrics'] = self._build_episode_metrics()
            
            sub_agent_info.append(info)
        
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
    
    def _rules_to_orders(self, selected_rules: List[int]) -> List[float]:
        """Convert rule selections to order quantities."""
        orders = []
        
        for agent_id, rule_id in enumerate(selected_rules):
            inv_pos = self._get_inventory_position(agent_id)
            avg_demand = self._get_avg_demand()
            
            if rule_id == 0:
                qty = self._apply_foq(inv_pos, avg_demand)
            elif rule_id == 1:
                qty = self._apply_poq(inv_pos, avg_demand)
            else:
                qty = self._apply_sm(inv_pos, avg_demand)
            
            orders.append(qty)
        
        return orders
    
    def _get_inventory_position(self, agent_id: int) -> float:
        """Calculate inventory position = inventory + pipeline - backlog."""
        inventory = float(self.inventory[agent_id]) if hasattr(self, 'inventory') else 0.0
        backlog = float(self.backlog[agent_id]) if hasattr(self, 'backlog') else 0.0
        
        pipeline_sum = 0.0
        if hasattr(self, 'order') and self.order and agent_id < len(self.order):
            pipeline_sum = sum(float(p) for p in self.order[agent_id])
        
        return inventory + pipeline_sum - backlog
    
    def _get_avg_demand(self) -> float:
        """Get average demand from recent history."""
        if self._demand_history:
            return max(1.0, np.mean(self._demand_history[-5:]))
        return 10.0
    
    def _apply_foq(self, inv_pos: float, avg_demand: float) -> float:
        """Fixed Order Quantity rule."""
        reorder_point = self.rule_params['foq']['reorder_point']
        order_qty = self.rule_params['foq']['order_quantity']
        
        if inv_pos > reorder_point:
            return 0.0
        return order_qty
    
    def _apply_poq(self, inv_pos: float, avg_demand: float) -> float:
        """Periodic Order Quantity rule."""
        safety_periods = self.rule_params['poq']['safety_periods']
        coverage_periods = self.rule_params['poq']['coverage_periods']
        
        reorder_point = avg_demand * safety_periods
        
        if inv_pos > reorder_point:
            return 0.0
        
        target_inv = avg_demand * coverage_periods
        order_qty = max(0.0, target_inv - inv_pos)
        order_qty = round(order_qty / 5) * 5
        return min(100, max(0, order_qty))
    
    def _apply_sm(self, inv_pos: float, avg_demand: float) -> float:
        """Silver-Meal rule."""
        safety_periods = self.rule_params['sm']['safety_periods']
        coverage_periods = self.rule_params['sm']['coverage_periods']
        
        reorder_point = avg_demand * safety_periods
        
        if inv_pos > reorder_point:
            return 0.0
        
        target_inv = avg_demand * coverage_periods
        order_qty = max(0.0, target_inv - inv_pos)
        order_qty = round(order_qty / 5) * 5
        return min(100, max(0, order_qty))
    
    def _build_episode_metrics(self) -> Dict:
        """Build episode-level metrics."""
        rule_usage = []
        for agent_id in range(self.agent_num):
            counts = np.array(self.rule_counts[agent_id], dtype=np.float32)
            total = counts.sum()
            if total > 0:
                freq = (counts / total).tolist()
            else:
                freq = [1/3, 1/3, 1/3]
            rule_usage.append(freq)
        
        return {
            'rule_usage': rule_usage,
            'primary_rules': self.phase_controller.primary_rules.copy(),
            'final_phase': self.phase_controller.get_phase(),
        }


# Test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING HIERARCHICAL RULE CHOICE ENVIRONMENT")
    print("="*60)
    
    env = Env()
    obs = env.reset()
    
    print(f"Observation shape: {np.array(obs).shape}")
    print(f"Action dim: {env.action_dim}")
    
    # Run episode to test phase transitions
    print("\n--- Running episode to test phases ---")
    total_reward = 0
    
    for step in range(60):
        # Random actions (neural network would provide these)
        actions = [np.random.randint(0, 3) for _ in range(env.agent_num)]
        result = env.step(actions, one_hot=False)
        obs, rewards, dones, infos = result
        
        total_reward += sum(r[0] for r in rewards)
        
        # Print at phase boundaries
        if step in [0, 19, 20, 21, 30, 50]:
            phase = infos[0]['phase']
            primary = infos[0]['primary_rule']
            print(f"  Step {step}: Phase={phase}, Primary={['FOQ','POQ','SM'][primary]}, "
                  f"Order={env.current_orders}, Inv={[int(i) for i in env.inventory]}")
    
    print(f"\n  Total reward (60 steps): {total_reward:.2f}")
    print(f"  Rule usage: {env.rule_counts}")
    print(f"  Final primary rules: {[['FOQ','POQ','SM'][r] for r in env.phase_controller.primary_rules]}")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETE")
    print("="*60)