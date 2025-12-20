"""
DEBUG SCRIPT v3: Kiểm tra CHÍNH XÁC state từ hierarchical_env
Mục tiêu: Xem _get_agent_state() trả về gì và có khớp với I, B, P không
"""

import sys
import numpy as np
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

def test_get_agent_state():
    """Kiểm tra _get_agent_state() trả về đúng không."""
    print("\n" + "="*70)
    print("🔍 TEST: _get_agent_state() có trả về đúng không?")
    print("="*70)
    
    from envs.hierarchical_env import HierarchicalSupplyChainEnv
    
    args = SimpleNamespace(
        n_agents=3, lead_time=4, episode_length=200, use_hierarchical=True,
        discovery_steps=20, analysis_steps=1, cooldown_period=15,
        switching_threshold=-100.0, evaluation_window=10,
        inventory_balance_weight=0.01, order_stability_weight=0.005,
        bullwhip_penalty_weight=0.02,
        foq_reorder_point=10.0, foq_order_quantity=20.0,
        poq_lead_time=4, poq_target_periods=2, poq_forecast_window=3,
        sm_setup_cost=50.0, sm_holding_cost=1.0, sm_forecast_horizon=10, sm_forecast_window=3,
    )
    
    env = HierarchicalSupplyChainEnv(args)
    env.reset()
    
    # Manually set state để test
    print("\n📊 MANUALLY SET STATE:")
    env.I = [17.86, 27.59, 28.04]
    env.B = [0.0, 0.0, 0.0]
    env.P = [[5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]]  # Pipeline = 20 mỗi agent
    env.demand_history = [10.0] * 10
    
    print(f"   env.I = {env.I}")
    print(f"   env.B = {env.B}")
    print(f"   env.P = {env.P}")
    
    # Gọi _get_agent_state() và xem nó trả về gì
    print("\n📊 _get_agent_state() TRẢ VỀ:")
    for agent_id in range(3):
        state = env._get_agent_state(agent_id)
        print(f"\n   Agent {agent_id}:")
        print(f"      inventory: {state.get('inventory', 'MISSING')}")
        print(f"      backlog: {state.get('backlog', 'MISSING')}")
        print(f"      pipeline: {state.get('pipeline', 'MISSING')}")
        print(f"      demand_history (last 3): {state.get('demand_history', [])[-3:]}")
        
        # Tính inventory position
        inv = state.get('inventory', 0)
        backlog = state.get('backlog', 0)
        pipeline = state.get('pipeline', [])
        inv_pos = inv + sum(pipeline) - backlog
        print(f"      → Inventory Position: {inv} + {sum(pipeline)} - {backlog} = {inv_pos:.2f}")


def test_rule_with_real_state():
    """Test rule với state THỰC TẾ từ environment."""
    print("\n" + "="*70)
    print("🔍 TEST: Rule với state THỰC TẾ từ environment")
    print("="*70)
    
    from envs.hierarchical_env import HierarchicalSupplyChainEnv
    from rules.rule_manager import RuleManager
    
    args = SimpleNamespace(
        n_agents=3, lead_time=4, episode_length=200, use_hierarchical=True,
        discovery_steps=20, analysis_steps=1, cooldown_period=15,
        switching_threshold=-100.0, evaluation_window=10,
        inventory_balance_weight=0.01, order_stability_weight=0.005,
        bullwhip_penalty_weight=0.02,
        foq_reorder_point=10.0, foq_order_quantity=20.0,
        poq_lead_time=4, poq_target_periods=2, poq_forecast_window=3,
        sm_setup_cost=50.0, sm_holding_cost=1.0, sm_forecast_horizon=10, sm_forecast_window=3,
    )
    
    env = HierarchicalSupplyChainEnv(args)
    env.reset()
    
    # Set state CAO
    env.I = [17.86, 27.59, 28.04]
    env.B = [0.0, 0.0, 0.0]
    env.P = [[5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]]
    env.demand_history = [10.0] * 10
    
    print("\n📊 STATE ĐÃ SET:")
    print(f"   Inventory: {env.I}")
    print(f"   Expected: POQ và SM phải return 0 vì inventory cao!")
    
    # Lấy rule_manager từ env
    rule_manager = env.rule_manager
    
    print("\n📊 TEST TỪNG AGENT VỚI TỪNG RULE:")
    for agent_id in range(3):
        state = env._get_agent_state(agent_id)
        inv_pos = state['inventory'] + sum(state['pipeline']) - state['backlog']
        
        print(f"\n   Agent {agent_id}: InvPos = {inv_pos:.2f}")
        
        for rule_id in [0, 1, 2]:
            rule_name = ['FOQ', 'POQ', 'SM'][rule_id]
            order = rule_manager.apply_rule(rule_id, state)
            
            # Kiểm tra
            if inv_pos > 15 and order > 0:
                status = "❌ SAI! Inventory cao nhưng vẫn order"
            elif inv_pos > 15 and order == 0:
                status = "✅ ĐÚNG! Không order khi inventory cao"
            elif inv_pos <= 15 and order > 0:
                status = "✅ ĐÚNG! Order khi inventory thấp"
            else:
                status = "⚠️ Cần xem xét"
            
            print(f"      {rule_name}: Order = {order:.2f} - {status}")


def test_step_with_high_inventory():
    """Test step khi inventory đã CAO."""
    print("\n" + "="*70)
    print("🔍 TEST: Step khi inventory ĐÃ CAO")
    print("="*70)
    
    from envs.hierarchical_env import HierarchicalSupplyChainEnv
    
    args = SimpleNamespace(
        n_agents=3, lead_time=4, episode_length=200, use_hierarchical=True,
        discovery_steps=20, analysis_steps=1, cooldown_period=15,
        switching_threshold=-100.0, evaluation_window=10,
        inventory_balance_weight=0.01, order_stability_weight=0.005,
        bullwhip_penalty_weight=0.02,
        foq_reorder_point=10.0, foq_order_quantity=20.0,
        poq_lead_time=4, poq_target_periods=2, poq_forecast_window=3,
        sm_setup_cost=50.0, sm_holding_cost=1.0, sm_forecast_horizon=10, sm_forecast_window=3,
    )
    
    env = HierarchicalSupplyChainEnv(args)
    env.reset()
    
    # Set INVENTORY CAO
    env.I = [30.0, 30.0, 30.0]
    env.B = [0.0, 0.0, 0.0]
    env.P = [[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10]]  # Pipeline = 40
    env.demand_history = [10.0] * 10
    
    print("\n📊 STATE TRƯỚC STEP:")
    print(f"   Inventory: {env.I}")
    print(f"   Backlog: {env.B}")
    print(f"   Pipeline sums: {[sum(p) for p in env.P]}")
    
    for agent_id in range(3):
        inv_pos = env.I[agent_id] + sum(env.P[agent_id]) - env.B[agent_id]
        print(f"   Agent {agent_id} InvPos: {inv_pos:.2f}")
    
    # Test với POQ (rule 1)
    actions = [[0.0, 1.0, 0.0] for _ in range(3)]
    
    print("\n📊 EXECUTING STEP với POQ...")
    obs, rewards, dones, infos = env.step(actions)
    
    print("\n📊 KẾT QUẢ:")
    print(f"   Selected Rules: {infos.get('selected_rules', 'N/A')}")
    print(f"   Order Quantities: {infos.get('order_quantities', 'N/A')}")
    
    orders = infos.get('order_quantities', [])
    for agent_id, order in enumerate(orders):
        if order > 0:
            print(f"   ❌ Agent {agent_id}: Order = {order:.2f} - SAI! Inventory cao nhưng vẫn order!")
        else:
            print(f"   ✅ Agent {agent_id}: Order = {order:.2f} - ĐÚNG!")


def check_demand_history_in_state():
    """Kiểm tra demand_history có được pass đúng vào state không."""
    print("\n" + "="*70)
    print("🔍 CHECK: demand_history trong state")
    print("="*70)
    
    from envs.hierarchical_env import HierarchicalSupplyChainEnv
    
    args = SimpleNamespace(
        n_agents=3, lead_time=4, episode_length=200, use_hierarchical=True,
        discovery_steps=20, analysis_steps=1, cooldown_period=15,
        switching_threshold=-100.0, evaluation_window=10,
        inventory_balance_weight=0.01, order_stability_weight=0.005,
        bullwhip_penalty_weight=0.02,
        foq_reorder_point=10.0, foq_order_quantity=20.0,
        poq_lead_time=4, poq_target_periods=2, poq_forecast_window=3,
        sm_setup_cost=50.0, sm_holding_cost=1.0, sm_forecast_horizon=10, sm_forecast_window=3,
    )
    
    env = HierarchicalSupplyChainEnv(args)
    env.reset()
    
    print(f"\n📊 env.demand_history: {env.demand_history}")
    
    state = env._get_agent_state(0)
    print(f"📊 state['demand_history']: {state.get('demand_history', 'MISSING')}")
    
    if not state.get('demand_history'):
        print("❌ demand_history RỖNG trong state!")
        print("   → POQ/SM sẽ dùng default demand = 10")
    else:
        print("✅ demand_history có trong state")


if __name__ == "__main__":
    test_get_agent_state()
    check_demand_history_in_state()
    test_rule_with_real_state()
    test_step_with_high_inventory()
    
    print("\n" + "="*70)
    print("🏁 DEBUG COMPLETE")
    print("="*70)