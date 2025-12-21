"""
HIERARCHICAL RULE CHOICE ENVIRONMENT - COMPATIBLE VERSION

Tương thích 100% với serial.py gốc của Liu.
KHÔNG cần thay đổi serial.py.

Flow:
Neural Network → Rule ID → Phase Controller → Rule Logic → Order Qty → Liu's step()
"""
import os
import sys
from typing import Dict, List, Any, Optional

import numpy as np
from gym import spaces

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import từ serial.py GỐC của bạn
from envs.serial import Env as LiuSerialEnv
from envs.serial import (
    LEVEL_NUM, 
    LEAD_TIME, 
    H, B, 
    S_I, S_O, 
    ALPHA, 
    EPISODE_LEN, 
    ACTION_DIM, 
    OBS_DIM
)


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
        self.phase = "discovery"
        
        # Performance tracking per rule per agent
        self.rule_rewards = [[[] for _ in range(3)] for _ in range(self.num_agents)]
        
        # Primary rules (selected after discovery)
        self.primary_rules = [0] * self.num_agents
        
        # Execution phase tracking
        self.steps_since_switch = [0] * self.num_agents
        self.recent_rewards = [[] for _ in range(self.num_agents)]
        
        # Discovery schedule
        self.discovery_schedule = self._create_discovery_schedule()
        self.discovery_idx = 0
        self._last_used_rules = [0] * self.num_agents
    
    def _create_discovery_schedule(self) -> List[List[int]]:
        """Create schedule for rule exploration during discovery."""
        schedule = []
        rules_per_cycle = 3
        cycles = self.discovery_steps // rules_per_cycle
        
        for cycle in range(cycles):
            for rule_id in range(3):
                schedule.append([rule_id] * self.num_agents)
        
        while len(schedule) < self.discovery_steps:
            rule_id = len(schedule) % 3
            schedule.append([rule_id] * self.num_agents)
        
        return schedule
    
    def process_actions(self, raw_actions: List, rewards: List = None) -> List[int]:
        """Process actions through hierarchical controller."""
        if rewards is not None:
            self._update_rewards(rewards)
        
        self.current_step += 1
        
        if self.phase == "discovery" and self.current_step >= self.discovery_steps:
            self._transition_to_analysis()
        
        if self.phase == "discovery":
            selected = self._discovery_action()
        elif self.phase == "analysis":
            self._transition_to_execution()
            selected = self.primary_rules.copy()
        else:
            selected = self._execution_action(raw_actions)
        
        self._last_used_rules = selected.copy()
        return selected
    
    def _update_rewards(self, rewards: List):
        """Track rewards for performance evaluation."""
        for agent_id, reward in enumerate(rewards):
            reward_val = float(reward[0]) if isinstance(reward, (list, np.ndarray)) else float(reward)
            
            self.recent_rewards[agent_id].append(reward_val)
            if len(self.recent_rewards[agent_id]) > self.evaluation_window:
                self.recent_rewards[agent_id].pop(0)
            
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
        
        print(f"\n{'='*60}")
        print("📊 ANALYSIS PHASE - Selecting Primary Rules")
        print(f"{'='*60}")
        
        rule_names = ['FOQ', 'POQ', 'SM']
        
        for agent_id in range(self.num_agents):
            avg_rewards = []
            for rule_id in range(3):
                rewards = self.rule_rewards[agent_id][rule_id]
                if rewards:
                    avg_rewards.append(np.mean(rewards))
                else:
                    avg_rewards.append(-float('inf'))
            
            best_rule = int(np.argmax(avg_rewards))
            self.primary_rules[agent_id] = best_rule
            
            print(f"  Agent {agent_id}: Primary={rule_names[best_rule]}")
            for rid, avg in enumerate(avg_rewards):
                samples = len(self.rule_rewards[agent_id][rid])
                if avg > -float('inf'):
                    print(f"    {rule_names[rid]}: Avg={avg:.2f} ({samples} samples)")
        
        print(f"{'='*60}\n")
    
    def _transition_to_execution(self):
        """Transition from analysis to execution phase."""
        self.phase = "execution"
        self.steps_since_switch = [0] * self.num_agents
    
    def _execution_action(self, raw_actions: List) -> List[int]:
        """Execution phase: Use primary rules with adaptive switching."""
        selected = []
        
        for agent_id in range(self.num_agents):
            self.steps_since_switch[agent_id] += 1
            nn_action = self._convert_action_to_rule(raw_actions[agent_id])
            
            can_switch = self.steps_since_switch[agent_id] >= self.cooldown_period
            recent = self.recent_rewards[agent_id]
            avg_recent = np.mean(recent) if recent else 0
            performance_poor = avg_recent < self.switching_threshold
            
            if can_switch and performance_poor and nn_action != self.primary_rules[agent_id]:
                selected.append(nn_action)
                self.steps_since_switch[agent_id] = 0
            else:
                selected.append(self.primary_rules[agent_id])
        
        return selected
    
    def _convert_action_to_rule(self, action) -> int:
        """Convert action to rule ID."""
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
        return self.phase


class Env(LiuSerialEnv):
    """
    Hierarchical Rule Choice Environment.
    Kế thừa 100% từ Liu's serial.py Env class.
    """
    
    def __init__(self, all_args=None):
        # Gọi __init__ của parent class
        super(Env, self).__init__()
        
        # Override action space: 3 rules thay vì 21 quantities
        self.action_dim = 3
        self.action_space = [spaces.Discrete(3) for _ in range(self.agent_num)]
        
        # Rule parameters
        self.rule_params = {
            'foq': {
                'reorder_point': 30.0,
                'order_quantity': 12.0,
            },
            'poq': {
                'safety_stock': 10.0,
                'coverage_periods': 1.5,
            },
            'sm': {
                'safety_stock': 8.0,
                'coverage_periods': 1.5,
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
        # Gọi reset của parent (Liu's Env)
        obs = super(Env, self).reset(train=train, normalize=normalize)
        
        # Reset hierarchical components
        self.phase_controller.reset()
        self.rule_counts = [[0, 0, 0] for _ in range(self.agent_num)]
        self._demand_history = [10.0] * 10
        self._last_rewards = None
        
        return obs
    
    def step(self, actions, one_hot=True):
        """Execute step with hierarchical rule selection."""
        # Convert to indices
        if one_hot:
            action_indices = [np.argmax(a) for a in actions]
        else:
            action_indices = list(actions)
        
        # Process through hierarchical controller
        selected_rules = self.phase_controller.process_actions(
            raw_actions=action_indices,
            rewards=self._last_rewards
        )
        
        # Update demand history
        if hasattr(self, 'demand_list') and self.step_num < len(self.demand_list):
            self._demand_history.append(float(self.demand_list[self.step_num]))
            if len(self._demand_history) > 20:
                self._demand_history = self._demand_history[-20:]
        
        # Convert rules to order quantities
        order_quantities = self._rules_to_orders(selected_rules)
        
        # Track rule usage
        for agent_id, rule_id in enumerate(selected_rules):
            self.rule_counts[agent_id][rule_id] += 1
        
        # Gọi state_update của Liu (QUAN TRỌNG)
        rewards = self.state_update(order_quantities)
        
        # Store orders
        self.current_orders = order_quantities
        
        # Get observations (Liu's method)
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
    
    def _rules_to_orders(self, selected_rules: List[int]) -> List:
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
            
            # Ensure integer for Liu's state_update
            orders.append(int(round(qty)))
        
        return orders
    
    def _get_inventory_position(self, agent_id: int) -> float:
        """Calculate inventory position."""
        inventory = float(self.inventory[agent_id])
        backlog = float(self.backlog[agent_id])
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
        return min(order_qty, max(0, reorder_point + order_qty - inv_pos))
    
    def _apply_poq(self, inv_pos: float, avg_demand: float) -> float:
        """Periodic Order Quantity rule."""
        safety_stock = self.rule_params['poq']['safety_stock']
        coverage_periods = self.rule_params['poq']['coverage_periods']
        
        target = safety_stock + avg_demand * coverage_periods
        
        if inv_pos >= target + avg_demand:
            return 0.0
        
        order_qty = max(0.0, target + avg_demand - inv_pos)
        order_qty = min(order_qty, avg_demand * 2)
        return round(order_qty)
    
    def _apply_sm(self, inv_pos: float, avg_demand: float) -> float:
        """Silver-Meal rule."""
        safety_stock = self.rule_params['sm']['safety_stock']
        coverage_periods = self.rule_params['sm']['coverage_periods']
        
        target = safety_stock + avg_demand * coverage_periods
        
        if inv_pos >= target + avg_demand:
            return 0.0
        
        order_qty = max(0.0, target + avg_demand - inv_pos)
        order_qty = min(order_qty, avg_demand * 2)
        return round(order_qty)
    
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


# =============================================================================
# TEST CODE
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING HIERARCHICAL RULE CHOICE ENVIRONMENT")
    print("="*60)
    
    env = Env()
    obs = env.reset()
    
    print(f"\nObservation shape: {np.array(obs).shape}")
    print(f"Action dim: {env.action_dim}")
    print(f"Number of agents: {env.agent_num}")
    
    print("\n--- Running full episode ---")
    total_reward = 0
    step_count = 0
    
    for step in range(min(200, env.episode_max_steps)):
        actions = [np.random.randint(0, 3) for _ in range(env.agent_num)]
        result = env.step(actions, one_hot=False)
        obs, rewards, dones, infos = result
        
        for r in rewards:
            reward_val = r[0] if isinstance(r, (list, np.ndarray)) else r
            total_reward += reward_val
        
        step_count += 1
        
        if step in [0, 19, 20, 21, 50, 100, 199]:
            phase = infos[0]['phase']
            primary = infos[0]['primary_rule']
            orders = [int(o) for o in env.current_orders]
            invs = [int(i) for i in env.inventory]
            rule_names = ['FOQ', 'POQ', 'SM']
            print(f"  Step {step:3d}: Phase={phase:9s}, Primary={rule_names[primary]}, "
                  f"Orders={orders}, Inv={invs}")
        
        if dones[0]:
            break
    
    print(f"\n--- Episode Complete ---")
    print(f"  Steps: {step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Avg reward/step: {total_reward/step_count:.2f}")
    
    print(f"\n--- Rule Usage ---")
    rule_names = ['FOQ', 'POQ', 'SM']
    for agent_id in range(env.agent_num):
        counts = env.rule_counts[agent_id]
        total = sum(counts)
        if total > 0:
            pcts = [c/total*100 for c in counts]
            print(f"  Agent {agent_id}: FOQ={pcts[0]:.1f}%, POQ={pcts[1]:.1f}%, SM={pcts[2]:.1f}%")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETE")
    print("="*60)
