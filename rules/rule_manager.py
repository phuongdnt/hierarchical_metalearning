"""Rule management and factory for creating and applying inventory rules."""

from typing import Dict, Any, List
from .base_rule import InventoryRule
from .foq_rule import FOQRule
from .poq_rule import POQRule
from .sm_rule import SilverMealRule

class RuleManager:
    """
    Manages inventory rules and provides factory methods.
    Central point for rule creation and application.
    
    Provides easy interface to:
    - Create all rules with parameters
    - Apply any rule given rule_id
    - Get rule information and names
    """
    
    # Rule registry mapping IDs to classes
    RULE_REGISTRY = {
        0: FOQRule,
        1: POQRule,
        2: SilverMealRule
    }
    
    RULE_NAMES = {
        0: "Fixed Order Quantity (FOQ)",
        1: "Periodic Order Quantity (POQ)",
        2: "Silver-Meal (SM)"
    }
    
    def __init__(self, rule_parameters: Dict[str, Dict[str, Any]]):
        """
        Initialize rule manager with parameters for all rules.
        
        Args:
            rule_parameters (dict): Parameters for each rule
                {
                    'foq': {'reorder_point': 10, 'order_quantity': 20},
                    'poq': {'lead_time': 4, 'target_periods': 2},
                    'sm': {'setup_cost': 50, 'holding_cost': 1, 'forecast_horizon': 10}
                }
        """
        self.rule_parameters = rule_parameters
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict[int, InventoryRule]:
        """Create instances of all rules."""
        rules = {}
        
        # Create FOQ rule
        if 'foq' in self.rule_parameters:
            rules[0] = FOQRule(self.rule_parameters['foq'])
        
        # Create POQ rule
        if 'poq' in self.rule_parameters:
            rules[1] = POQRule(self.rule_parameters['poq'])
        
        # Create SM rule
        if 'sm' in self.rule_parameters:
            rules[2] = SilverMealRule(self.rule_parameters['sm'])
        
        return rules
    
    def apply_rule(self, rule_id: int, agent_state: Dict[str, Any]) -> float:
        """
        Apply specified rule to calculate order quantity.
        
        Args:
            rule_id (int): Rule identifier (0=FOQ, 1=POQ, 2=SM)
            agent_state (dict): Current agent state
            
        Returns:
            float: Calculated order quantity
            
        Raises:
            ValueError: If rule_id is invalid
        """
        if rule_id not in self.rules:
            raise ValueError(f"Invalid rule_id: {rule_id}. Available rules: {list(self.rules.keys())}")
        
        quantity = self.rules[rule_id].calculate_order_quantity(agent_state)
        
        # Ensure non-negative
        return max(0.0, float(quantity))
    
    def get_rule_name(self, rule_id: int) -> str:
        """
        Get human-readable name for rule.
        
        Args:
            rule_id (int): Rule identifier
            
        Returns:
            str: Rule name
        """
        return self.RULE_NAMES.get(rule_id, f"Unknown Rule {rule_id}")
    
    def get_all_rule_names(self) -> Dict[int, str]:
        """Get all available rule names."""
        return self.RULE_NAMES.copy()
    
    def get_rule_info(self, rule_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific rule.
        
        Args:
            rule_id (int): Rule identifier
            
        Returns:
            dict: Rule information including parameters
        """
        if rule_id not in self.rules:
            return {}
        
        rule = self.rules[rule_id]
        return {
            'id': rule_id,
            'name': rule.get_rule_name(),
            'parameters': rule.parameters,
            'parameters_info': rule.get_parameters_info() if hasattr(rule, 'get_parameters_info') else {}
        }
    
    @staticmethod
    def get_default_parameters() -> Dict[str, Dict[str, Any]]:
        """
        Get default parameters for all rules.
        
        Returns:
            dict: Default parameters for FOQ, POQ, SM
        """
        return {
            'foq': {
                'reorder_point': 10.0,
                'order_quantity': 20.0
            },
            'poq': {
                'lead_time': 4,
                'target_periods': 2,
                'forecast_window': 3
            },
            'sm': {
                'setup_cost': 50.0,
                'holding_cost': 1.0,
                'forecast_horizon': 10,
                'forecast_window': 3
            }
        }
    
    def __str__(self) -> str:
        return f"RuleManager with {len(self.rules)} rules: {self.get_all_rule_names()}"
