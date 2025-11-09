"""Integration test for complete hierarchical system."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from types import SimpleNamespace


def create_test_args():
    """Create test arguments for hierarchical environment."""
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
        discovery_steps=5,  # Short for testing
        analysis_steps=1,
        cooldown_period=3,  # Short for testing
        switching_threshold=-100.0,
        evaluation_window=3,  # Short for testing
        
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


def test_hierarchical_system():
    """Test complete hierarchical system end-to-end."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Complete Hierarchical System")
    print("="*70)
    
    # Create environment
    args = create_test_args()
    
    try:
        from envs.hierarchical_env import HierarchicalSupplyChainEnv
        
        print("\n✅ Step 1: Creating environment...")
        env = HierarchicalSupplyChainEnv(args)
        print(f"   Environment: {type(env).__name__}")
        print(f"   Agents: {env.n_agents}")
        print(f"   Observation dim: {env.share_obs_dim}")
        
        # Reset
        print("\n✅ Step 2: Resetting environment...")
        obs = env.reset()
        print(f"   Observation shapes: {[o.shape for o in obs]}")
        print(f"   Expected shape: (17,) for each agent")
        
        # Verify observation dimensions
        for i, o in enumerate(obs):
            assert o.shape == (17,), f"Agent {i}: Expected (17,), got {o.shape}"
        print(f"   ✅ All observations have correct shape!")
        
        # Run through phases
        print("\n✅ Step 3: Running through 3 phases...")
        
        # Discovery phase (steps 0-4)
        # Discovery phase (steps 0-3, then step 4 starts analysis)
        print(f"\n   📊 Discovery Phase (steps 0-3):")
        for step in range(4):  # ✅ Changed from range(5) to range(4)
            actions = [np.random.rand(3) for _ in range(env.n_agents)]
            obs, rewards, dones, infos = env.step(actions)
            
            phase = infos.get('phase', 'unknown')
            rules = infos.get('selected_rules', [])
            
            assert phase == 'discovery', f"Expected 'discovery', got '{phase}'"
            print(f"      Step {step}: phase={phase}, rules={rules}")

        
        # Analysis phase (step 5)
        # Analysis phase (step 4)  # ✅ Updated comment
        print(f"\n   📊 Analysis Phase (step 4):")
        actions = [np.random.rand(3) for _ in range(env.n_agents)]
        obs, rewards, dones, infos = env.step(actions)

        phase = infos.get('phase', 'unknown')
        primary_rules = infos.get('primary_rules', [])

        assert phase == 'analysis', f"Expected 'analysis', got '{phase}'"
        print(f"      Step 4: phase={phase}")  # ✅ Changed from "Step 5" to "Step 4"
        print(f"      Primary rules determined: {primary_rules}")

        # Execution phase (steps 6-10)
        # Execution phase (steps 5-10)  # ✅ Updated comment
        print(f"\n   📊 Execution Phase (steps 5-10):")
        for step in range(5, 11):  # ✅ Changed from (6, 11) to (5, 11)
            actions = [np.random.rand(3) for _ in range(env.n_agents)]
            obs, rewards, dones, infos = env.step(actions)
            
            phase = infos.get('phase', 'unknown')
            rules = infos.get('selected_rules', [])
            
            assert phase == 'execution', f"Expected 'execution', got '{phase}'"
            print(f"      Step {step}: phase={phase}, rules={rules}, "
                f"rewards=[{', '.join([f'{r:.1f}' for r in rewards])}]")
            
            if any(dones):
                break

        
        # Verify all components working
        print("\n✅ Step 4: Verifying components...")
        
        # Check hierarchical controller exists
        assert hasattr(env, 'phase_controller'), "Missing phase_controller"
        print(f"   ✅ Phase controller: OK")
        
        # Check rule manager exists
        assert hasattr(env, 'rule_manager'), "Missing rule_manager"
        print(f"   ✅ Rule manager: OK")
        
        # Check coordination exists
        assert hasattr(env, 'observation_builder'), "Missing observation_builder"
        assert hasattr(env, 'reward_coordinator'), "Missing reward_coordinator"
        print(f"   ✅ Coordination: OK")
        
        # Success!
        print("\n" + "="*70)
        print("✅✅✅ INTEGRATION TEST PASSED!")
        print("="*70)
        print("\n🎉 Hierarchical system is working correctly!")
        print("   - 3-phase learning operational")
        print("   - Enhanced observations (17-dim)")
        print("   - Coordination rewards active")
        print("   - All components integrated")
        print("\n" + "="*70)
        
        return True
        
    except Exception as e:
        print(f"\n" + "="*70)
        print("❌ INTEGRATION TEST FAILED!")
        print("="*70)
        print(f"\n❌ ERROR: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("\n" + "="*70)
        return False


def test_environment_properties():
    """Test environment properties and methods."""
    print("\n" + "="*70)
    print("TEST: Environment Properties")
    print("="*70)
    
    args = create_test_args()
    
    try:
        from envs.hierarchical_env import HierarchicalSupplyChainEnv
        env = HierarchicalSupplyChainEnv(args)
        
        # Test properties
        print(f"\n✅ Testing environment properties:")
        print(f"   n_agents: {env.n_agents}")
        print(f"   agent_num: {env.agent_num}")
        print(f"   share_obs_dim: {env.share_obs_dim}")
        print(f"   action_dim: {env.action_dim}")
        
        # Test methods exist
        assert hasattr(env, 'reset'), "Missing reset method"
        assert hasattr(env, 'step'), "Missing step method"
        print(f"\n✅ All required methods exist")
        
        print("\n" + "="*70)
        print("✅ Environment properties test PASSED!")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("HIERARCHICAL MARL SYSTEM - INTEGRATION TESTS")
    print("🚀"*35)
    
    # Run tests
    test1_passed = test_hierarchical_system()
    print("\n" + "-"*70 + "\n")
    test2_passed = test_environment_properties()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Integration Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Properties Test:  {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("="*70)
    
    if test1_passed and test2_passed:
        print("\n🎉🎉🎉 ALL TESTS PASSED! WEEK 3 COMPLETE! 🎉🎉🎉\n")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - CHECK ERRORS ABOVE\n")
        sys.exit(1)
