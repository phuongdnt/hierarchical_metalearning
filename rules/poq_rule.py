"""Periodic Order Quantity (POQ) rule implementation."""

import numpy as np
from typing import Dict, Any, List
from .base_rule import InventoryRule

class POQRule(InventoryRule):
    """
    Periodic Order Quantity Policy Implementation.
    
    Logic: Order to cover forecasted demand for (lead_time + target_periods).
    This policy is effective for items with predictable demand patterns.
    
    Reference: Silver, E. A., Pyke, D. F., & Peterson, R. (1998).
               Inventory management and production planning and scheduling.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize POQ rule.
        
        Args:
            parameters (dict): Must contain:
                - 'lead_time': Lead time periods
                - 'target_periods': Number of periods to cover
                - 'forecast_window': Window for demand forecast (optional, default=3)
        """
        super().__init__(parameters)
        self.rule_id = 1
        self.rule_name = "Periodic Order Quantity (POQ)"
        
        # Extract parameters
        self.lead_time = int(parameters['lead_time'])
        self.target_periods = int(parameters['target_periods'])
        self.forecast_window = int(parameters.get('forecast_window', 3))
    
    def calculate_order_quantity(self, agent_state: Dict[str, Any]) -> float:
        """
        Calculate order quantity using POQ logic.
        
        Logic:
            coverage_period = lead_time + target_periods
            forecasted_demand = forecast(coverage_period)
            target_inventory = sum(forecasted_demand)
            order_quantity = max(0, target_inventory - inventory_position)
        
        Args:
            agent_state (dict): Current agent state
            
        Returns:
            float: Order quantity to cover target periods
        """
        # Calculate coverage period
        coverage_period = self.lead_time + self.target_periods
        
        # Forecast demand for coverage period
        demand_forecast = self._forecast_demand(agent_state, coverage_period)
        
        # Calculate target inventory level
        target_inventory = sum(demand_forecast)
        
        # Calculate current inventory position
        inventory_position = self._calculate_inventory_position(agent_state)
        
        # Order up to target level
        order_quantity = max(0.0, target_inventory - inventory_position)
        
        return order_quantity
    
    def _forecast_demand(self, agent_state: Dict[str, Any], periods: int) -> List[float]:
        """
        Forecast demand using moving average.
        
        Args:
            agent_state (dict): Contains demand_history
            periods (int): Number of periods to forecast
            
        Returns:
            list: Forecasted demand for each period
        """
        demand_history = agent_state.get('demand_history', [])
        
        if not demand_history:
            # No history - use zero forecast
            return [0.0] * periods
        
        # Use moving average for forecast
        window = min(self.forecast_window, len(demand_history))
        recent_demand = demand_history[-window:]
        average_demand = float(np.mean(recent_demand))
        
        # Simple forecast: constant average
        forecast = [average_demand] * periods
        
        return forecast
    
    def get_rule_name(self) -> str:
        """Return rule name."""
        return self.rule_name
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate POQ parameters.
        
        Args:
            params (dict): Parameters to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If parameters invalid
        """
        if 'lead_time' not in params:
            raise ValueError("POQ requires 'lead_time' parameter")
        if 'target_periods' not in params:
            raise ValueError("POQ requires 'target_periods' parameter")
        
        lead_time = int(params['lead_time'])
        target_periods = int(params['target_periods'])
        
        if lead_time <= 0:
            raise ValueError(f"lead_time must be positive, got {lead_time}")
        if target_periods <= 0:
            raise ValueError(f"target_periods must be positive, got {target_periods}")
        
        return True
    
    def get_parameters_info(self) -> Dict[str, str]:
        """Return information about rule parameters."""
        return {
            'lead_time': f"Supply lead time (current: {self.lead_time})",
            'target_periods': f"Periods to cover (current: {self.target_periods})",
            'coverage': f"Total coverage: {self.lead_time + self.target_periods} periods"
        }

