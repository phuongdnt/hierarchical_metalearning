"""Three-phase learning controller for hierarchical rule selection - FIXED."""

import numpy as np
from typing import List, Dict, Any, Optional
from .performance_tracker import RulePerformanceTracker


class ThreePhaseController:
    """
    Manages 3-phase hierarchical learning process.
    
    Phase 1 - Discovery (Steps 1-20):
        Allow free exploration, track all rule performances
        
    Phase 2 - Analysis (Step 21):
        Statistical analysis, determine primary rules
        
    Phase 3 - Execution (Steps 22-200):
        Use primary rules with adaptive switching
    """
    
    def __init__(self, num_agents: int, 
                 discovery_steps: int = 20,
                 analysis_steps: int = 1,
                 cooldown_period: int = 15,
                 switching_threshold: float = -100.0,
                 evaluation_window: int = 10):
        """
        Initialize three-phase controller.
        """
        self.num_agents = num_agents
        self.discovery_steps = discovery_steps
        self.analysis_steps = analysis_steps
        self.cooldown_period = cooldown_period
        self.switching_threshold = switching_threshold
        self.evaluation_window = evaluation_window
        
        # Performance tracker
        self.performance_tracker = RulePerformanceTracker(num_agents)
        
        # Hierarchical state
        self.primary_rules = [0] * num_agents
        self.rule_confidences = [0.5] * num_agents
        self.cooldown_timers = [0] * num_agents
        
        # Phase tracking
        self.current_step = 0
        self.discovery_complete = False
        self.analysis_complete = False
        
        # Switching statistics
        self.switch_counts = [0] * num_agents
        self.switch_history = []
    
    def reset(self):
        """Reset controller for new episode."""
        self.performance_tracker.reset()
        self.primary_rules = [0] * self.num_agents
        self.rule_confidences = [0.5] * self.num_agents
        self.cooldown_timers = [0] * self.num_agents
        self.current_step = 0
        self.discovery_complete = False
        self.analysis_complete = False
        self.switch_counts = [0] * self.num_agents
        self.switch_history = []
    
    def get_current_phase(self) -> str:
        """Get current learning phase."""
        if self.current_step < self.discovery_steps:
            return 'discovery'
        elif self.current_step < self.discovery_steps + self.analysis_steps:
            return 'analysis'
        else:
            return 'execution'
    
    def process_rule_selection(self, raw_actions: List,
                               rewards: Optional[List[float]] = None) -> List[int]:
        """
        Main entry point - process rule selection based on current phase.
        
        Args:
            raw_actions (list): Raw actions from agents (can be one-hot, logits, or scalars)
            rewards (list, optional): Rewards from previous step
            
        Returns:
            list: Final selected rules for each agent
        """
        # ✅ FIX: Convert actions to rule IDs - handle all formats
        raw_rule_ids = []
        for action in raw_actions:
            rule_id = self._convert_action_to_rule_id(action)
            raw_rule_ids.append(rule_id)
        
        # Update performance tracking
        if rewards is not None and self.current_step > 0:
            self._update_performance_tracking(rewards)
        
        # Phase-specific processing
        phase = self.get_current_phase()
        
        if phase == 'discovery':
            selected_rules = self._discovery_phase_processing(raw_rule_ids)
        elif phase == 'analysis':
            selected_rules = self._analysis_phase_processing(raw_rule_ids)
        else:  # execution
            selected_rules = self._execution_phase_processing(raw_rule_ids)
        
        # Save for next step
        self._last_selected_rules = selected_rules
        
        # Increment step
        self.current_step += 1
        
        return selected_rules
    
    def _convert_action_to_rule_id(self, action) -> int:
        """
        ✅ FIXED: Convert action to rule ID - handles all formats.
        
        Supports:
        - Scalar int/float: 0, 1, 2
        - Numpy scalar: np.array(1)
        - One-hot numpy array: np.array([0, 1, 0])
        - One-hot Python list: [0.0, 1.0, 0.0]
        - Logits: [0.1, 0.8, 0.1]
        """
        # Case 1: Numpy array
        if isinstance(action, np.ndarray):
            if action.shape == () or len(action.shape) == 0:
                # Scalar numpy
                return int(action)
            else:
                # One-hot or logits array
                return int(np.argmax(action))
        
        # Case 2: Python list or tuple
        elif isinstance(action, (list, tuple)):
            # One-hot or logits list
            return int(np.argmax(action))
        
        # Case 3: Scalar (int or float)
        else:
            return int(action)
    
    def _update_performance_tracking(self, rewards: List[float]):
        """Update performance tracker with rewards."""
        if not hasattr(self, '_last_selected_rules'):
            return
        
        for agent_id, reward in enumerate(rewards):
            rule_used = self._last_selected_rules[agent_id]
            self.performance_tracker.record_performance(
                agent_id=agent_id,
                rule_id=rule_used,
                reward=reward,
                step=self.current_step - 1
            )
    
    def _discovery_phase_processing(self, raw_rule_ids: List[int]) -> List[int]:
        """Discovery: Allow free exploration."""
        return raw_rule_ids
    
    def _analysis_phase_processing(self, raw_rule_ids: List[int]) -> List[int]:
        """Analysis: Determine primary rules."""
        if not self.discovery_complete:
            print(f"\n{'='*70}")
            print("ANALYSIS PHASE - Determining Primary Rules")
            print(f"{'='*70}")
            
            for agent_id in range(self.num_agents):
                best_rule = self.performance_tracker.get_best_rule(agent_id)
                confidence = self.performance_tracker.calculate_rule_confidence(agent_id, best_rule)
                
                self.primary_rules[agent_id] = best_rule
                self.rule_confidences[agent_id] = confidence
                
                stats = self.performance_tracker.get_rule_statistics(agent_id, best_rule)
                comparison = self.performance_tracker.get_performance_comparison(agent_id)
                
                rule_names = ['FOQ', 'POQ', 'SM']
                print(f"\nAgent {agent_id}:")
                print(f"  Primary Rule: {best_rule} ({rule_names[best_rule]})")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Performance: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
                
                print(f"  Comparison:")
                for rule_id in range(3):
                    comp = comparison[rule_id]
                    print(f"    {rule_names[rule_id]}: Mean={comp['mean_reward']:.2f}, "
                          f"Samples={comp['sample_count']}, Conf={comp['confidence']:.3f}")
            
            self.discovery_complete = True
            print(f"\n{'='*70}\n")
        
        self.analysis_complete = True
        return self.primary_rules.copy()
    
    def _execution_phase_processing(self, raw_rule_ids: List[int]) -> List[int]:
        """
        Execution: Use primary rules with adaptive switching.
        """
        selected_rules = []
        
        for agent_id, suggested_rule in enumerate(raw_rule_ids):
            # Update cooldown
            if self.cooldown_timers[agent_id] > 0:
                self.cooldown_timers[agent_id] -= 1
            
            # Switching decision logic
            if self.cooldown_timers[agent_id] > 0:
                # In cooldown - use primary
                selected_rule = self.primary_rules[agent_id]
                
            elif self._should_switch_rule(agent_id, suggested_rule):
                # Switch approved
                old_rule = self.primary_rules[agent_id]
                selected_rule = suggested_rule
                
                # Update state
                self.primary_rules[agent_id] = suggested_rule
                self.cooldown_timers[agent_id] = self.cooldown_period
                self.switch_counts[agent_id] += 1
                
                # Log switch
                self.switch_history.append({
                    'step': self.current_step,
                    'agent': agent_id,
                    'from_rule': old_rule,
                    'to_rule': suggested_rule
                })
                
                rule_names = ['FOQ', 'POQ', 'SM']
                print(f"[Step {self.current_step}] Agent {agent_id}: "
                      f"{rule_names[old_rule]} → {rule_names[suggested_rule]}")
            else:
                # Keep primary
                selected_rule = self.primary_rules[agent_id]
            
            selected_rules.append(selected_rule)
        
        return selected_rules
    
    def _should_switch_rule(self, agent_id: int, suggested_rule: int) -> bool:
        """
        Decide if should switch rules.
        """
        # Don't switch to same rule
        if suggested_rule == self.primary_rules[agent_id]:
            return False
        
        # Get recent performance
        recent = self.performance_tracker.get_recent_performance(
            agent_id, window=self.evaluation_window
        )
        
        # Need sufficient data
        if len(recent) < self.evaluation_window // 2:
            return False
        
        # Check if performance below threshold
        recent_avg = np.mean(recent)
        return recent_avg < self.switching_threshold
    
    def get_state_for_observation(self, agent_id: int) -> Dict[str, float]:
        """Get hierarchical state for observation building."""
        return {
            'primary_rule': float(self.primary_rules[agent_id]),
            'rule_confidence': float(self.rule_confidences[agent_id]),
            'cooldown_remaining': float(self.cooldown_timers[agent_id])
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'current_step': self.current_step,
            'current_phase': self.get_current_phase(),
            'primary_rules': self.primary_rules.copy(),
            'rule_confidences': self.rule_confidences.copy(),
            'switch_counts': self.switch_counts.copy(),
            'total_switches': sum(self.switch_counts),
            'switch_history': self.switch_history.copy()
        }