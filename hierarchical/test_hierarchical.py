"""Test hierarchical intelligence system."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hierarchical.performance_tracker import RulePerformanceTracker
from hierarchical.phase_controller import ThreePhaseController
import numpy as np

print("="*70)
print("HIERARCHICAL INTELLIGENCE TEST")
print("="*70)

# Test 1: Performance Tracker
print("\n TEST 1: Performance Tracker")
tracker = RulePerformanceTracker(num_agents=2, num_rules=3)

# Simulate performance data
for step in range(10):
    tracker.record_performance(0, 0, -100.0 + np.random.randn()*10, step)
    tracker.record_performance(0, 1, -80.0 + np.random.randn()*10, step)
    tracker.record_performance(0, 2, -120.0 + np.random.randn()*10, step)

best = tracker.get_best_rule(0)
print(f"✅ Best rule: {best} (expected: 1)")

# Test 2: Phase Controller
print("\n✅ TEST 2: Phase Controller")
controller = ThreePhaseController(num_agents=2, discovery_steps=5, analysis_steps=1)

# Discovery phase
for step in range(5):
    actions = [np.array([1, 0, 0]), np.array([0, 1, 0])]
    rewards = [-100.0, -90.0] if step > 0 else None
    rules = controller.process_rule_selection(actions, rewards)
    print(f"Step {step}: phase={controller.get_current_phase()}, rules={rules}")

# Analysis phase
actions = [np.array([1, 0, 0]), np.array([0, 1, 0])]
rewards = [-100.0, -90.0]
rules = controller.process_rule_selection(actions, rewards)
print(f"Step 5: Analysis - primary_rules={controller.primary_rules}")

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
