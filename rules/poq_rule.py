"""Periodic Order Quantity (POQ) rule implementation - FIXED v3.

Logic đơn giản:
- Chỉ order khi inventory_position < demand × safety_periods
- safety_periods = 1.5 (đủ cho 1.5 periods)
- Nếu đủ hàng cho 1.5 periods → KHÔNG ORDER
"""

import numpy as np
from typing import Dict, Any, List
from .base_rule import InventoryRule


class POQRule(InventoryRule):
    """
    Periodic Order Quantity Policy - Conservative Version.
    
    Key principle: Only order when inventory is TRULY insufficient.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.rule_id = 1
        self.rule_name = "Periodic Order Quantity (POQ)"
        
        self.lead_time = int(parameters.get('lead_time', 4))
        self.target_periods = int(parameters.get('target_periods', 2))
        self.forecast_window = int(parameters.get('forecast_window', 3))
        
        # ✅ KEY PARAMETER: Only order if inv < demand * safety_periods
        # safety_periods = 1.5 means: order only if inventory covers less than 1.5 periods
        self.safety_periods = float(parameters.get('safety_periods', 1.5))
    
    def calculate_order_quantity(self, agent_state: Dict[str, Any]) -> float:
        """
        Calculate order quantity using conservative POQ logic.
        
        ✅ SIMPLE LOGIC:
        1. Estimate average demand
        2. Calculate reorder_point = avg_demand × safety_periods
        3. If inventory_position > reorder_point → DON'T ORDER
        4. If inventory_position <= reorder_point → Order to cover target
        """
        # Step 1: Estimate demand
        avg_demand = self._estimate_demand(agent_state)
        
        # Step 2: Calculate inventory position
        inventory_position = self._calculate_inventory_position(agent_state)
        
        # Step 3: Calculate reorder point (conservative)
        # Only order if inventory covers less than safety_periods of demand
        reorder_point = avg_demand * self.safety_periods
        
        # ✅ KEY CHECK: Don't order if we have enough inventory
        if inventory_position > reorder_point:
            return 0.0
        
        # Step 4: Calculate target and order quantity
        coverage_period = self.lead_time + self.target_periods
        target_inventory = avg_demand * coverage_period
        
        order_quantity = max(0.0, target_inventory - inventory_position)
        
        return order_quantity
    
    def _estimate_demand(self, agent_state: Dict[str, Any]) -> float:
        """Estimate average demand from history."""
        demand_history = agent_state.get('demand_history', [])
        
        if not demand_history:
            return 10.0  # Default demand
        
        window = min(self.forecast_window, len(demand_history))
        recent_demand = demand_history[-window:]
        
        return sum(recent_demand) / len(recent_demand) if recent_demand else 10.0
    
    def get_rule_name(self) -> str:
        return self.rule_name
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        return True
    
    def get_parameters_info(self) -> Dict[str, str]:
        return {
            'lead_time': f"Lead time (current: {self.lead_time})",
            'target_periods': f"Target periods (current: {self.target_periods})",
            'safety_periods': f"Only order if inv < demand × this (current: {self.safety_periods})"
        }