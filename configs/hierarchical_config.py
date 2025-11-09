"""Configuration management for hierarchical system."""

from typing import Dict, Any

class HierarchicalConfig:
    """
    Centralized configuration for hierarchical rule selection system.
    """
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default hierarchical configuration."""
        return {
            # Hierarchical learning
            'use_hierarchical': True,
            'discovery_steps': 20,
            'analysis_steps': 1,
            'switching_threshold': -100.0,
            'cooldown_period': 15,
            'evaluation_window': 10,
            
            # Coordination
            'coordination_weight': 0.05,
            'inventory_balance_weight': 0.01,
            'order_stability_weight': 0.005,
            'bullwhip_penalty_weight': 0.02,
            
            # FOQ rule parameters
            'foq_reorder_point': 10.0,
            'foq_order_quantity': 20.0,
            
            # POQ rule parameters
            'poq_lead_time': 4,
            'poq_target_periods': 2,
            'poq_forecast_window': 3,
            
            # SM rule parameters
            'sm_setup_cost': 50.0,
            'sm_holding_cost': 1.0,
            'sm_forecast_horizon': 10,
            'sm_forecast_window': 3
        }
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        required_keys = [
            'discovery_steps', 'analysis_steps', 'cooldown_period',
            'switching_threshold', 'coordination_weight'
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validation rules
        if config['discovery_steps'] < 1:
            raise ValueError("discovery_steps must be >= 1")
        if config['cooldown_period'] < 0:
            raise ValueError("cooldown_period must be >= 0")
        
        return True
