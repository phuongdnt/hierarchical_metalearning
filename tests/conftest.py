"""Pytest configuration and fixtures for hierarchical system."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from types import SimpleNamespace


@pytest.fixture
def basic_args():
    """Basic arguments for testing hierarchical system."""
    return SimpleNamespace(
        # Environment
        env_name='hierarchical',
        n_agents=3,
        lead_time=4,
        episode_length=200,
        n_rollout_threads=1,
        seed=1,
        
        # Hierarchical system
        use_hierarchical=True,
        discovery_steps=20,
        analysis_steps=1,
        cooldown_period=15,
        switching_threshold=-100.0,
        evaluation_window=10,
        
        # Coordination
        inventory_balance_weight=0.01,
        order_stability_weight=0.005,
        bullwhip_penalty_weight=0.02,
        
        # FOQ parameters
        foq_reorder_point=10.0,
        foq_order_quantity=20.0,
        
        # POQ parameters
        poq_lead_time=4,
        poq_target_periods=2,
        poq_forecast_window=3,
        
        # SM parameters
        sm_setup_cost=50.0,
        sm_holding_cost=1.0,
        sm_forecast_horizon=10,
        sm_forecast_window=3
    )


@pytest.fixture
def hierarchical_env(basic_args):
    """Create hierarchical environment for testing."""
    from envs.hierarchical_env import HierarchicalSupplyChainEnv
    return HierarchicalSupplyChainEnv(basic_args)


@pytest.fixture
def rule_manager():
    """Create rule manager for testing."""
    from rules.rule_manager import RuleManager
    params = RuleManager.get_default_parameters()
    return RuleManager(params)


@pytest.fixture
def phase_controller():
    """Create phase controller for testing."""
    from hierarchical.phase_controller import ThreePhaseController
    return ThreePhaseController(num_agents=3, discovery_steps=5, analysis_steps=1)


@pytest.fixture
def performance_tracker():
    """Create performance tracker for testing."""
    from hierarchical.performance_tracker import RulePerformanceTracker
    return RulePerformanceTracker(num_agents=3, num_rules=3)


@pytest.fixture
def observation_builder():
    """Create observation builder for testing."""
    from coordination.observation_builder import ObservationBuilder
    return ObservationBuilder(num_agents=3, lead_time=4)


@pytest.fixture
def reward_coordinator():
    """Create reward coordinator for testing."""
    from coordination.reward_coordinator import RewardCoordinator
    return RewardCoordinator(num_agents=3)
