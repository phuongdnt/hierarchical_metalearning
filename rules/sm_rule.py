"""Silver-Meal dynamic lot-sizing rule implementation."""

import numpy as np
from typing import Dict, Any, List
from .base_rule import InventoryRule

class SilverMealRule(InventoryRule):
    """
    Silver-Meal Dynamic Lot-Sizing Algorithm.
    
    Logic: Minimize average cost per period over planning horizon.
    Balances setup costs and holding costs dynamically.
    
    Reference: Silver, E. A., & Meal, H. C. (1973). 
               A heuristic for selecting lot size quantities for the case of a 
               deterministic time-varying demand rate and discrete opportunities 
               for replenishment. Production and inventory management, 14(2), 64-74.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize Silver-Meal rule.
        
        Args:
            parameters (dict): Must contain:
                - 'setup_cost': Fixed cost per order
                - 'holding_cost': Cost per unit per period
                - 'forecast_horizon': Planning horizon periods
                - 'forecast_window': Window for demand forecast (optional, default=3)
        """
        super().__init__(parameters)
        self.rule_id = 2
        self.rule_name = "Silver-Meal (SM)"
        
        # Extract parameters
        self.setup_cost = float(parameters['setup_cost'])
        self.holding_cost = float(parameters['holding_cost'])
        self.forecast_horizon = int(parameters['forecast_horizon'])
        self.forecast_window = int(parameters.get('forecast_window', 3))
    
    def calculate_order_quantity(self, agent_state: Dict[str, Any]) -> float:
        """
        Calculate order quantity using Silver-Meal algorithm.
        
        Logic:
            For each potential lot size k (covering 1 to n periods):
                Calculate average cost per period = (setup_cost + holding_cost) / k
                Find k that minimizes average cost per period
        
        Args:
            agent_state (dict): Current agent state
            
        Returns:
            float: Optimal order quantity
        """
        # Get demand forecast
        demand_forecast = self._forecast_demand(agent_state, self.forecast_horizon)
        
        if not demand_forecast or sum(demand_forecast) == 0:
            return 0.0
        
        # Silver-Meal algorithm to find optimal lot size
        optimal_periods = self._find_optimal_periods(demand_forecast)
        
        # Calculate order quantity for optimal periods
        optimal_lot_size = sum(demand_forecast[:optimal_periods])
        
        # Adjust for current inventory position
        inventory_position = self._calculate_inventory_position(agent_state)
        order_quantity = max(0.0, optimal_lot_size - inventory_position)
        
        return order_quantity
    
    def _find_optimal_periods(self, demand_forecast: List[float]) -> int:
        """
        Find optimal number of periods to cover using Silver-Meal algorithm.
        
        The algorithm minimizes the average cost per period by balancing:
        - Setup cost (fixed per order)
        - Holding cost (increases with lot size and time)
        
        Args:
            demand_forecast (list): Forecasted demand for each period
            
        Returns:
            int: Optimal number of periods to cover
        """
        min_cost_per_period = float('inf')
        optimal_periods = 1
        
        cumulative_demand = 0.0
        cumulative_holding_cost = 0.0
        
        for period in range(1, len(demand_forecast) + 1):
            # Add demand for this period
            period_demand = demand_forecast[period - 1]
            cumulative_demand += period_demand
            
            # Calculate holding cost for this period's demand
            if period > 1:
                # Items ordered in period 1 are held for (period-1) periods
                cumulative_holding_cost += self.holding_cost * period_demand * (period - 1)
            
            # Total cost for ordering lot covering 'period' periods
            total_cost = self.setup_cost + cumulative_holding_cost
            cost_per_period = total_cost / period
            
            # Check if this is better than previous
            if cost_per_period < min_cost_per_period:
                min_cost_per_period = cost_per_period
                optimal_periods = period
            else:
                # Cost per period started increasing - stop searching
                break
        
        return optimal_periods
    
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
            return [0.0] * periods
        
        # Moving average forecast
        window = min(self.forecast_window, len(demand_history))
        recent_demand = demand_history[-window:]
        average_demand = float(np.mean(recent_demand))
        
        # Constant forecast (can be extended to trend-based forecast)
        forecast = [average_demand] * periods
        
        return forecast
    
    def get_rule_name(self) -> str:
        """Return rule name."""
        return self.rule_name
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate Silver-Meal parameters.
        
        Args:
            params (dict): Parameters to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If parameters invalid
        """
        required_params = ['setup_cost', 'holding_cost', 'forecast_horizon']
        
        for param in required_params:
            if param not in params:
                raise ValueError(f"SM requires '{param}' parameter")
        
        setup_cost = float(params['setup_cost'])
        holding_cost = float(params['holding_cost'])
        forecast_horizon = int(params['forecast_horizon'])
        
        if setup_cost < 0:
            raise ValueError(f"setup_cost must be non-negative, got {setup_cost}")
        if holding_cost < 0:
            raise ValueError(f"holding_cost must be non-negative, got {holding_cost}")
        if forecast_horizon <= 0:
            raise ValueError(f"forecast_horizon must be positive, got {forecast_horizon}")
        
        return True
    
    def get_parameters_info(self) -> Dict[str, str]:
        """Return information about rule parameters."""
        return {
            'setup_cost': f"Fixed ordering cost (current: {self.setup_cost})",
            'holding_cost': f"Holding cost per unit per period (current: {self.holding_cost})",
            'forecast_horizon': f"Planning horizon (current: {self.forecast_horizon})"
        }
