"""Enhanced observation builder for hierarchical system."""

import numpy as np
from typing import Dict, List, Any, Optional

class ObservationBuilder:
    """
    Build enhanced observations for hierarchical agents.
    
    Enhanced observation structure (17 dimensions):
    
    Original features (10-dim):
        - inventory_level
        - backlog_level
        - pipeline_inventory (4 values for lead_time=4)
        - demand_history (4 recent values)
    
    Hierarchical features (3-dim):
        - primary_rule (0, 1, or 2)
        - rule_confidence (0.0 to 1.0)
        - cooldown_remaining (0 to cooldown_period)
    
    Coordination features (4-dim):
        - upstream_inventory (0 if no upstream)
        - upstream_order (0 if no upstream)
        - downstream_inventory (0 if no downstream)
        - downstream_backlog (0 if no downstream)
    """
    
    def __init__(self, num_agents: int, lead_time: int = 4):
        """
        Initialize observation builder.
        
        Args:
            num_agents (int): Number of agents in supply chain
            lead_time (int): Lead time for pipeline (default: 4)
        """
        self.num_agents = num_agents
        self.lead_time = lead_time
        
        # Calculate observation dimensions
        self.original_dim = 2 + lead_time + 4  # inv, backlog, pipeline, demand
        self.hierarchical_dim = 3  # primary_rule, confidence, cooldown
        self.coordination_dim = 4  # upstream + downstream info
        self.total_dim = self.original_dim + self.hierarchical_dim + self.coordination_dim
    
    def build_observation(self, agent_id: int, 
                         base_state: Dict[str, Any],
                         hierarchical_state: Dict[str, float],
                         system_state: Dict[str, Any]) -> np.ndarray:
        """
        Build complete enhanced observation for an agent.
        
        Args:
            agent_id (int): Agent identifier
            base_state (dict): Base supply chain state (inventory, backlog, etc.)
            hierarchical_state (dict): Hierarchical features from phase controller
            system_state (dict): Global system state for coordination features
            
        Returns:
            np.ndarray: Enhanced observation vector (17-dim)
        """
        # Part 1: Original features (10-dim)
        original_features = self._build_original_features(agent_id, base_state)
        
        # Part 2: Hierarchical features (3-dim)
        hierarchical_features = self._build_hierarchical_features(hierarchical_state)
        
        # Part 3: Coordination features (4-dim)
        coordination_features = self._build_coordination_features(agent_id, system_state)
        
        # Concatenate all features
        enhanced_obs = np.concatenate([
            original_features,
            hierarchical_features,
            coordination_features
        ])
        
        return enhanced_obs.astype(np.float32)
    
    def _build_original_features(self, agent_id: int, 
                                 base_state: Dict[str, Any]) -> np.ndarray:
        """
        Build original supply chain features (unchanged from baseline).
        
        Args:
            agent_id (int): Agent identifier
            base_state (dict): Contains inventory, backlog, pipeline, demand
            
        Returns:
            np.ndarray: Original features (10-dim)
        """
        features = []
        
        # Inventory level
        inventory = float(base_state.get('inventory', 0.0))
        features.append(inventory)
        
        # Backlog level
        backlog = float(base_state.get('backlog', 0.0))
        features.append(backlog)
        
        # Pipeline inventory (lead_time values)
        pipeline = base_state.get('pipeline', [])
        if len(pipeline) < self.lead_time:
            # Pad with zeros if needed
            pipeline = list(pipeline) + [0.0] * (self.lead_time - len(pipeline))
        elif len(pipeline) > self.lead_time:
            # Truncate if too long
            pipeline = pipeline[:self.lead_time]
        features.extend([float(p) for p in pipeline])
        
        # Demand history (last 4 periods)
        demand_history = base_state.get('demand_history', [])
        if len(demand_history) < 4:
            demand = [0.0] * 4
        else:
            demand = demand_history[-4:]
        features.extend([float(d) for d in demand])
        
        return np.array(features, dtype=np.float32)
    
    def _build_hierarchical_features(self, 
                                    hierarchical_state: Dict[str, float]) -> np.ndarray:
        """
        Build hierarchical intelligence features.
        
        Args:
            hierarchical_state (dict): From phase_controller.get_state_for_observation()
            
        Returns:
            np.ndarray: Hierarchical features (3-dim)
        """
        features = [
            float(hierarchical_state.get('primary_rule', 0.0)),
            float(hierarchical_state.get('rule_confidence', 0.5)),
            float(hierarchical_state.get('cooldown_remaining', 0.0))
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _build_coordination_features(self, agent_id: int,
                                     system_state: Dict[str, Any]) -> np.ndarray:
        """
        Build coordination features for multi-agent coordination.
        
        Args:
            agent_id (int): Current agent
            system_state (dict): Contains inventory, orders, backlog for all agents
            
        Returns:
            np.ndarray: Coordination features (4-dim)
        """
        features = []
        
        # Upstream agent information
        if agent_id > 0:
            # Has upstream neighbor
            upstream_inventory = float(system_state.get('inventories', [0]*self.num_agents)[agent_id-1])
            upstream_order = float(system_state.get('orders', [0]*self.num_agents)[agent_id-1])
        else:
            # No upstream (manufacturer)
            upstream_inventory = 0.0
            upstream_order = 0.0
        
        features.append(upstream_inventory)
        features.append(upstream_order)
        
        # Downstream agent information
        if agent_id < self.num_agents - 1:
            # Has downstream neighbor
            downstream_inventory = float(system_state.get('inventories', [0]*self.num_agents)[agent_id+1])
            downstream_backlog = float(system_state.get('backlogs', [0]*self.num_agents)[agent_id+1])
        else:
            # No downstream (retailer)
            downstream_inventory = 0.0
            downstream_backlog = 0.0
        
        features.append(downstream_inventory)
        features.append(downstream_backlog)
        
        return np.array(features, dtype=np.float32)
    
    def get_observation_shape(self) -> tuple:
        """Get shape of enhanced observation."""
        return (self.total_dim,)
    
    def __str__(self) -> str:
        return (f"ObservationBuilder(total_dim={self.total_dim}: "
                f"original={self.original_dim}, "
                f"hierarchical={self.hierarchical_dim}, "
                f"coordination={self.coordination_dim})")
