"""Coordination reward calculator for multi-agent supply chain."""

import numpy as np
from typing import Dict, List, Any

class RewardCoordinator:
    """
    Calculate coordination bonuses/penalties for supply chain harmony.
    
    Coordination objectives:
    1. Inventory balance: Minimize inventory level differences between neighbors
    2. Order stability: Minimize variance in ordering patterns
    3. Bullwhip mitigation: Reduce demand amplification across chain
    """
    
    def __init__(self, num_agents: int,
                 inventory_balance_weight: float = 0.01,
                 order_stability_weight: float = 0.005,
                 bullwhip_penalty_weight: float = 0.02):
        """
        Initialize reward coordinator.
        
        Args:
            num_agents (int): Number of agents
            inventory_balance_weight (float): Weight for inventory balance bonus
            order_stability_weight (float): Weight for order stability bonus
            bullwhip_penalty_weight (float): Weight for bullwhip penalty
        """
        self.num_agents = num_agents
        self.inventory_balance_weight = inventory_balance_weight
        self.order_stability_weight = order_stability_weight
        self.bullwhip_penalty_weight = bullwhip_penalty_weight
        
        # Track order history for stability calculation
        self.order_history = [[] for _ in range(num_agents)]
        self.stability_window = 5  # Last 5 orders
    
    def calculate_coordination_reward(self, agent_id: int,
                                      base_reward: float,
                                      system_state: Dict[str, Any]) -> float:
        """
        Calculate total reward including coordination bonus.
        
        Args:
            agent_id (int): Agent identifier
            base_reward (float): Original reward from environment
            system_state (dict): Current system state
            
        Returns:
            float: Enhanced reward = base_reward + coordination_bonus
        """
        coordination_bonus = 0.0
        
        # 1. Inventory balance bonus
        balance_bonus = self._calculate_inventory_balance_bonus(agent_id, system_state)
        coordination_bonus += balance_bonus
        
        # 2. Order stability bonus
        stability_bonus = self._calculate_order_stability_bonus(agent_id, system_state)
        coordination_bonus += stability_bonus
        
        # 3. Bullwhip penalty
        bullwhip_penalty = self._calculate_bullwhip_penalty(agent_id, system_state)
        coordination_bonus += bullwhip_penalty
        
        # Enhanced reward
        enhanced_reward = base_reward + coordination_bonus
        
        return enhanced_reward
    
    def _calculate_inventory_balance_bonus(self, agent_id: int,
                                          system_state: Dict[str, Any]) -> float:
        """
        Calculate bonus for balanced inventory across supply chain.
        
        Penalizes large inventory differences between neighboring echelons.
        
        Args:
            agent_id (int): Agent identifier
            system_state (dict): System state
            
        Returns:
            float: Balance bonus (negative = penalty)
        """
        inventories = system_state.get('inventories', [])
        
        if len(inventories) < 2:
            return 0.0
        
        bonus = 0.0
        
        # Compare with upstream neighbor
        if agent_id > 0:
            inventory_diff = abs(inventories[agent_id] - inventories[agent_id-1])
            bonus -= self.inventory_balance_weight * inventory_diff
        
        # Compare with downstream neighbor
        if agent_id < self.num_agents - 1:
            inventory_diff = abs(inventories[agent_id] - inventories[agent_id+1])
            bonus -= self.inventory_balance_weight * inventory_diff
        
        return bonus
    
    def _calculate_order_stability_bonus(self, agent_id: int,
                                        system_state: Dict[str, Any]) -> float:
        """
        Calculate bonus for stable ordering patterns.
        
        Rewards low variance in recent orders.
        
        Args:
            agent_id (int): Agent identifier
            system_state (dict): System state
            
        Returns:
            float: Stability bonus (negative = penalty for high variance)
        """
        # Update order history
        current_order = system_state.get('orders', [0]*self.num_agents)[agent_id]
        self.order_history[agent_id].append(current_order)
        
        # Keep only recent orders
        if len(self.order_history[agent_id]) > self.stability_window:
            self.order_history[agent_id].pop(0)
        
        # Need sufficient history
        if len(self.order_history[agent_id]) < 3:
            return 0.0
        
        # Calculate order variance
        order_variance = np.var(self.order_history[agent_id])
        
        # Penalty for high variance (unstable ordering)
        bonus = -self.order_stability_weight * order_variance
        
        return bonus
    
    def _calculate_bullwhip_penalty(self, agent_id: int,
                                   system_state: Dict[str, Any]) -> float:
        """
        Calculate penalty for bullwhip effect contribution.
        
        Bullwhip effect: amplification of demand variability upstream.
        
        Args:
            agent_id (int): Agent identifier
            system_state (dict): System state
            
        Returns:
            float: Bullwhip penalty (negative if amplifying demand)
        """
        # Need sufficient order history
        if len(self.order_history[agent_id]) < self.stability_window:
            return 0.0
        
        # Get recent orders
        recent_orders = self.order_history[agent_id][-self.stability_window:]
        
        # Calculate coefficient of variation for orders
        order_mean = np.mean(recent_orders)
        if order_mean == 0:
            return 0.0
        
        order_std = np.std(recent_orders)
        order_cv = order_std / order_mean
        
        # Get demand coefficient of variation (if available)
        demand_history = system_state.get('demand_history', [])
        if len(demand_history) >= self.stability_window:
            recent_demand = demand_history[-self.stability_window:]
            demand_mean = np.mean(recent_demand)
            if demand_mean > 0:
                demand_std = np.std(recent_demand)
                demand_cv = demand_std / demand_mean
                
                # Bullwhip ratio: order CV / demand CV
                # > 1 means amplification (bad)
                if demand_cv > 0:
                    bullwhip_ratio = order_cv / demand_cv
                    
                    # Penalty if amplifying (ratio > 1)
                    if bullwhip_ratio > 1.0:
                        penalty = -self.bullwhip_penalty_weight * (bullwhip_ratio - 1.0)
                        return penalty
        
        return 0.0
    
    def reset(self):
        """Reset order history for new episode."""
        self.order_history = [[] for _ in range(self.num_agents)]
    
    def __str__(self) -> str:
        return (f"RewardCoordinator(agents={self.num_agents}, "
                f"balance_weight={self.inventory_balance_weight}, "
                f"stability_weight={self.order_stability_weight}, "
                f"bullwhip_weight={self.bullwhip_penalty_weight})")
