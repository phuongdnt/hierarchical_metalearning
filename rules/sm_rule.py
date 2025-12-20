"""Silver-Meal heuristic rule implementation - FIXED v3.

Logic đơn giản:
- Chỉ order khi inventory_position < demand × safety_periods
- safety_periods = 1.5 (đủ cho 1.5 periods)
- Nếu đủ hàng cho 1.5 periods → KHÔNG ORDER
"""

import numpy as np
from typing import Dict, Any, List
from .base_rule import InventoryRule


class SilverMealRule(InventoryRule):
    """
    Silver-Meal Heuristic - Conservative Version.
    
    Key principle: Only order when inventory is TRULY insufficient.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.rule_id = 2
        self.rule_name = "Silver-Meal (SM)"
        
        self.setup_cost = float(parameters.get('setup_cost', 50.0))
        self.holding_cost = float(parameters.get('holding_cost', 1.0))
        self.forecast_horizon = int(parameters.get('forecast_horizon', 10))
        self.forecast_window = int(parameters.get('forecast_window', 3))
        
        # ✅ KEY PARAMETER: Only order if inv < demand * safety_periods
        self.safety_periods = float(parameters.get('safety_periods', 1.5))
    
    def calculate_order_quantity(self, agent_state: Dict[str, Any]) -> float:
        """
        Calculate order quantity using conservative Silver-Meal logic.
        
        ✅ SIMPLE LOGIC:
        1. Estimate average demand
        2. Calculate reorder_point = avg_demand × safety_periods
        3. If inventory_position > reorder_point → DON'T ORDER
        4. If inventory_position <= reorder_point → Use Silver-Meal to find optimal lot size
        """
        # Step 1: Estimate demand
        avg_demand = self._estimate_demand(agent_state)
        
        # Step 2: Calculate inventory position
        inventory_position = self._calculate_inventory_position(agent_state)
        
        # Step 3: Calculate reorder point (conservative)
        reorder_point = avg_demand * self.safety_periods
        
        # ✅ KEY CHECK: Don't order if we have enough inventory
        if inventory_position > reorder_point:
            return 0.0
        
        # Step 4: Use Silver-Meal to find optimal lot size
        demand_forecast = [avg_demand] * self.forecast_horizon
        optimal_periods = self._find_optimal_coverage(demand_forecast)
        
        total_demand = avg_demand * optimal_periods
        order_quantity = max(0.0, total_demand - inventory_position)
        
        return order_quantity
    
    def _find_optimal_coverage(self, demand_forecast: List[float]) -> int:
        """Find optimal periods using Silver-Meal criterion."""
        if not demand_forecast:
            return 1
        
        K = self.setup_cost
        h = self.holding_cost
        
        best_periods = 1
        cumulative_holding = 0.0
        prev_avg_cost = float('inf')
        
        for k in range(1, min(len(demand_forecast) + 1, 10)):  # Max 10 periods
            if k > 1:
                cumulative_holding += h * demand_forecast[k-1] * (k - 1)
            
            total_cost = K + cumulative_holding
            avg_cost = total_cost / k
            
            if avg_cost > prev_avg_cost + 1e-8:
                break
            
            prev_avg_cost = avg_cost
            best_periods = k
        
        return best_periods
    
    def _estimate_demand(self, agent_state: Dict[str, Any]) -> float:
        """Estimate average demand from history."""
        demand_history = agent_state.get('demand_history', [])
        
        if not demand_history:
            return 10.0
        
        window = min(self.forecast_window, len(demand_history))
        recent_demand = demand_history[-window:]
        
        return sum(recent_demand) / len(recent_demand) if recent_demand else 10.0
    
    def get_rule_name(self) -> str:
        return self.rule_name
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        return True
    
    def get_parameters_info(self) -> Dict[str, str]:
        return {
            'setup_cost': f"Setup cost K (current: {self.setup_cost})",
            'holding_cost': f"Holding cost h (current: {self.holding_cost})",
            'safety_periods': f"Only order if inv < demand × this (current: {self.safety_periods})"
        }