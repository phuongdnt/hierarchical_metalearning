"""Quick test to verify all rules work correctly."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.rule_manager import RuleManager
import numpy as np

def test_all_rules():
    """Test all rules with sample data"""
    
    print("="*70)
    print("TESTING HIERARCHICAL RULE SYSTEM")
    print("="*70)
    
    # Get default parameters
    params = RuleManager.get_default_parameters()
    print(f"\n✅ Step 1: Loading default parameters")
    for rule_name, rule_params in params.items():
        print(f"   {rule_name.upper()}: {rule_params}")
    
    # Create rule manager
    try:
        manager = RuleManager(params)
        print(f"\n✅ Step 2: Rule manager created successfully")
        print(f"   Available rules: {manager.get_all_rule_names()}")
    except Exception as e:
        print(f"\n❌ ERROR creating rule manager: {e}")
        return False
    
    # Create test scenarios
    scenarios = [
        {
            'name': 'Low Inventory (Should trigger orders)',
            'state': {
                'inventory': 5.0,
                'backlog': 2.0,
                'pipeline': [3.0, 3.0, 2.0, 2.0],
                'demand_history': [10.0, 12.0, 11.0, 13.0, 12.0]
            }
        },
        {
            'name': 'High Inventory (Might not order)',
            'state': {
                'inventory': 50.0,
                'backlog': 0.0,
                'pipeline': [5.0, 5.0, 5.0, 5.0],
                'demand_history': [10.0, 11.0, 9.0, 10.0, 11.0]
            }
        },
        {
            'name': 'High Backlog (Critical situation)',
            'state': {
                'inventory': 10.0,
                'backlog': 15.0,
                'pipeline': [8.0, 8.0, 7.0, 7.0],
                'demand_history': [15.0, 18.0, 20.0, 17.0, 19.0]
            }
        }
    ]
    
    # Test all rules on all scenarios
    print("\n" + "="*70)
    print("RULE APPLICATION RESULTS")
    print("="*70)
    
    all_passed = True
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        state = scenario['state']
        
        # Calculate inventory position
        inv_pos = state['inventory'] + sum(state['pipeline']) - state['backlog']
        avg_demand = np.mean(state['demand_history'][-3:])
        
        print(f"   State: Inv={state['inventory']:.1f}, Backlog={state['backlog']:.1f}, "
              f"Pipeline={sum(state['pipeline']):.1f}")
        print(f"   → Inventory Position: {inv_pos:.1f}")
        print(f"   → Recent Avg Demand: {avg_demand:.1f}")
        print(f"\n   Rule Decisions:")
        
        for rule_id in [0, 1, 2]:
            try:
                quantity = manager.apply_rule(rule_id, state)
                rule_name = manager.get_rule_name(rule_id)
                print(f"   • {rule_name:30s}: Order {quantity:6.2f} units")
                
                # Basic sanity check
                if quantity < 0:
                    print(f"      ❌ ERROR: Negative quantity!")
                    all_passed = False
                    
            except Exception as e:
                print(f"   • Rule {rule_id:30s}: ❌ ERROR: {e}")
                all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - RULE SYSTEM WORKING CORRECTLY!")
    else:
        print("❌ SOME TESTS FAILED - CHECK ERRORS ABOVE")
    print("="*70)
    
    return all_passed

if __name__ == "__main__":
    success = test_all_rules()
    sys.exit(0 if success else 1)
