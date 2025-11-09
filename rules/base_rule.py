"""Abstract base class for inventory management rules."""

from abc import ABC, abstractmethod
from typing import Dict, Any

class InventoryRule(ABC):
    """
    Abstract interface that all inventory management rules must implement.
    Defines standard contract for rule-based ordering decisions.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize rule with parameters.
        
        Args:
            parameters (dict): Rule-specific parameters
        """
        self.parameters = parameters
        self.validate_parameters(parameters)
        self.rule_id = None  # Set by subclass
        self.rule_name = None  # Set by subclass
    
    @abstractmethod
    def calculate_order_quantity(self, agent_state: Dict[str, Any]) -> float:
        """
        Calculate order quantity based on rule logic.
        
        Args:
            agent_state (dict): Current agent state containing:
                - 'inventory': current inventory level
                - 'backlog': current backlog level
                - 'pipeline': list of incoming orders
                - 'demand_history': historical demand data
                
        Returns:
            float: Order quantity (non-negative)
        """
        pass
    
    @abstractmethod
    def get_rule_name(self) -> str:
        """Return human-readable rule name."""
        pass
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate rule-specific parameters.
        
        Args:
            params (dict): Parameters to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        pass
    
    def _calculate_inventory_position(self, agent_state: Dict[str, Any]) -> float:
        """
        Calculate inventory position = inventory + pipeline - backlog.
        
        Args:
            agent_state (dict): Agent state information
            
        Returns:
            float: Current inventory position
        """
        inventory = float(agent_state.get('inventory', 0.0))
        backlog = float(agent_state.get('backlog', 0.0))
        pipeline = agent_state.get('pipeline', [])
        pipeline_sum = float(sum(pipeline)) if pipeline else 0.0
        
        return inventory + pipeline_sum - backlog
    
    def __str__(self) -> str:
        return f"{self.get_rule_name()} with parameters: {self.parameters}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.parameters})"
