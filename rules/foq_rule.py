"""Fixed Order Quantity (FOQ) rule implementation."""

from typing import Dict, Any
from .base_rule import InventoryRule

class FOQRule(InventoryRule):
    """
    Fixed Order Quantity (s,Q) Policy Implementation.
    
    Logic: When inventory position <= reorder point (s), order fixed quantity (Q).
    This is one of the simplest and most widely used inventory policies.
    
    Reference: Silver, E. A., Pyke, D. F., & Peterson, R. (1998). 
               Inventory management and production planning and scheduling.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize FOQ rule.
        
        Args:
            parameters (dict): Must contain:
                - 'reorder_point': Threshold inventory level (s)
                - 'order_quantity': Fixed amount to order (Q)
        """
        super().__init__(parameters)
        self.rule_id = 0
        self.rule_name = "Fixed Order Quantity (FOQ)"
        
        # Extract parameters
        self.reorder_point = float(parameters['reorder_point'])
        self.order_quantity = float(parameters['order_quantity'])
    
    def calculate_order_quantity(self, agent_state: Dict[str, Any]) -> float:
        """
        Calculate order quantity using FOQ logic.
        
        Logic:
            inventory_position = inventory + pipeline - backlog
            if inventory_position <= reorder_point:
                return order_quantity
            else:
                return 0
        
        Args:
            agent_state (dict): Current agent state
            
        Returns:
            float: Order quantity (Q if position <= s, else 0)
        """
        # Calculate current inventory position
        inventory_position = self._calculate_inventory_position(agent_state)
        
        # FOQ logic: order Q when position drops to or below s
        if inventory_position <= self.reorder_point:
            return self.order_quantity
        else:
            return 0.0
    
    def get_rule_name(self) -> str:
        """Return rule name."""
        return self.rule_name
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate FOQ parameters.
        
        Args:
            params (dict): Parameters to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If parameters invalid
        """
        if 'reorder_point' not in params:
            raise ValueError("FOQ requires 'reorder_point' parameter")
        if 'order_quantity' not in params:
            raise ValueError("FOQ requires 'order_quantity' parameter")
        
        reorder_point = float(params['reorder_point'])
        order_quantity = float(params['order_quantity'])
        
        if reorder_point < 0:
            raise ValueError(f"reorder_point must be non-negative, got {reorder_point}")
        if order_quantity <= 0:
            raise ValueError(f"order_quantity must be positive, got {order_quantity}")
        
        return True
    
    def get_parameters_info(self) -> Dict[str, str]:
        """Return information about rule parameters."""
        return {
            'reorder_point': f"Threshold for ordering (current: {self.reorder_point})",
            'order_quantity': f"Fixed amount to order (current: {self.order_quantity})"
        }
