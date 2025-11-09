"""Performance tracking system for rule selection intelligence."""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

class RulePerformanceTracker:
    """
    Track and analyze rule performance across different contexts.
    Provides intelligence for primary rule selection and switching decisions.
    
    INCLUDES:
    - Performance recording and database management
    - Statistical analysis (mean, std, confidence intervals)
    - Rule comparison and ranking
    - Confidence score calculation
    """
    
    def __init__(self, num_agents: int, num_rules: int = 3):
        """
        Initialize performance tracker.
        
        Args:
            num_agents (int): Number of agents in supply chain
            num_rules (int): Number of available rules (default: 3)
        """
        self.num_agents = num_agents
        self.num_rules = num_rules
        
        # Performance database: agent -> rule -> list of (reward, context, step)
        self.performance_db = {
            agent_id: {rule_id: [] for rule_id in range(num_rules)}
            for agent_id in range(num_agents)
        }
        
        # Recent performance windows for quick access
        self.recent_windows = {
            agent_id: [] for agent_id in range(num_agents)
        }
        
        # Statistics cache for efficiency
        self.statistics_cache = {
            agent_id: {rule_id: None for rule_id in range(num_rules)}
            for agent_id in range(num_agents)
        }
        
        # Track when statistics were last updated
        self.cache_valid = {
            agent_id: {rule_id: False for rule_id in range(num_rules)}
            for agent_id in range(num_agents)
        }
    
    def record_performance(self, agent_id: int, rule_id: int, 
                          reward: float, step: int, 
                          context: Optional[Dict[str, Any]] = None):
        """
        Record performance of a rule for an agent.
        
        Args:
            agent_id (int): Agent identifier
            rule_id (int): Rule identifier (0, 1, 2)
            reward (float): Reward received
            step (int): Current timestep
            context (dict, optional): Contextual information
        """
        record = {
            'reward': reward,
            'step': step,
            'context': context or {}
        }
        
        # Add to performance database
        self.performance_db[agent_id][rule_id].append(record)
        
        # Invalidate statistics cache
        self.cache_valid[agent_id][rule_id] = False
        
        # Update recent window (last 50 rewards for this agent)
        self.recent_windows[agent_id].append(reward)
        if len(self.recent_windows[agent_id]) > 50:
            self.recent_windows[agent_id].pop(0)
    
    def get_best_rule(self, agent_id: int) -> int:
        """
        Get best performing rule for agent based on discovery phase data.
        
        Args:
            agent_id (int): Agent identifier
            
        Returns:
            int: Best rule ID (0, 1, or 2)
        """
        best_rule = 0
        best_performance = float('-inf')
        
        for rule_id in range(self.num_rules):
            records = self.performance_db[agent_id][rule_id]
            
            if len(records) > 0:
                avg_performance = np.mean([r['reward'] for r in records])
                
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_rule = rule_id
        
        return best_rule
    
    def get_rule_statistics(self, agent_id: int, rule_id: int) -> Dict[str, float]:
        """
        Get statistical summary of rule performance.
        
        INCLUDES STATISTICAL ANALYSIS - No need for statistical_analyzer.py
        
        Args:
            agent_id (int): Agent identifier
            rule_id (int): Rule identifier
            
        Returns:
            dict: Statistics including mean, std, min, max, count
        """
        # Check cache first
        if self.cache_valid[agent_id][rule_id]:
            return self.statistics_cache[agent_id][rule_id]
        
        # Calculate fresh statistics
        records = self.performance_db[agent_id][rule_id]
        
        if len(records) == 0:
            stats = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0,
                'confidence_interval': (0.0, 0.0)
            }
        else:
            rewards = [r['reward'] for r in records]
            
            # Basic statistics
            mean = float(np.mean(rewards))
            std = float(np.std(rewards))
            
            # Confidence interval (95%)
            if len(rewards) > 1:
                se = std / np.sqrt(len(rewards))
                margin = 1.96 * se  # 95% confidence
                ci = (mean - margin, mean + margin)
            else:
                ci = (mean, mean)
            
            stats = {
                'mean': mean,
                'std': std,
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'count': len(rewards),
                'confidence_interval': ci
            }
        
        # Update cache
        self.statistics_cache[agent_id][rule_id] = stats
        self.cache_valid[agent_id][rule_id] = True
        
        return stats
    
    def get_recent_performance(self, agent_id: int, window: int = 10) -> List[float]:
        """
        Get recent performance for agent.
        
        Args:
            agent_id (int): Agent identifier
            window (int): Number of recent rewards to return
            
        Returns:
            list: Recent rewards (up to window size)
        """
        recent = self.recent_windows[agent_id]
        return recent[-window:] if len(recent) >= window else recent
    
    def calculate_rule_confidence(self, agent_id: int, rule_id: int) -> float:
        """
        Calculate confidence in rule performance (0-1 scale).
        
        STATISTICAL CONFIDENCE CALCULATION - No need for statistical_analyzer.py
        
        Confidence factors:
        - Sample size (more samples = higher confidence)
        - Performance level (better performance = higher confidence)
        - Consistency (lower variance = higher confidence)
        
        Args:
            agent_id (int): Agent identifier
            rule_id (int): Rule identifier
            
        Returns:
            float: Confidence score between 0 and 1
        """
        stats = self.get_rule_statistics(agent_id, rule_id)
        
        if stats['count'] < 3:
            return 0.5  # Neutral confidence with insufficient data
        
        # Sample size factor: confidence increases with more samples
        sample_size_factor = min(1.0, stats['count'] / 20.0)
        
        # Performance factor: normalize reward to [0, 1]
        # Assuming typical rewards in range [-300, 0]
        if stats['mean'] >= 0:
            performance_factor = 1.0
        else:
            performance_factor = max(0.0, min(1.0, (stats['mean'] + 300) / 300))
        
        # Consistency factor: lower variance = higher confidence
        if stats['std'] > 0:
            consistency_factor = 1.0 / (1.0 + stats['std'] / 100.0)
        else:
            consistency_factor = 1.0
        
        # Weighted combination
        confidence = (0.4 * sample_size_factor + 
                     0.4 * performance_factor + 
                     0.2 * consistency_factor)
        
        return max(0.0, min(1.0, confidence))
    
    def get_performance_comparison(self, agent_id: int) -> Dict[int, Dict[str, float]]:
        """
        Compare performance of all rules for an agent.
        
        Args:
            agent_id (int): Agent identifier
            
        Returns:
            dict: Comparative statistics for all rules
        """
        comparison = {}
        
        for rule_id in range(self.num_rules):
            stats = self.get_rule_statistics(agent_id, rule_id)
            confidence = self.calculate_rule_confidence(agent_id, rule_id)
            
            comparison[rule_id] = {
                'mean_reward': stats['mean'],
                'std_reward': stats['std'],
                'sample_count': stats['count'],
                'confidence': confidence
            }
        
        return comparison
    
    def reset(self):
        """Reset all tracking data (for new episode)."""
        self.performance_db = {
            agent_id: {rule_id: [] for rule_id in range(self.num_rules)}
            for agent_id in range(self.num_agents)
        }
        
        self.recent_windows = {
            agent_id: [] for agent_id in range(self.num_agents)
        }
        
        self.statistics_cache = {
            agent_id: {rule_id: None for rule_id in range(self.num_rules)}
            for agent_id in range(self.num_agents)
        }
        
        self.cache_valid = {
            agent_id: {rule_id: False for rule_id in range(self.num_rules)}
            for agent_id in range(self.num_agents)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all tracked data."""
        summary = {}
        
        for agent_id in range(self.num_agents):
            agent_summary = {}
            
            for rule_id in range(self.num_rules):
                stats = self.get_rule_statistics(agent_id, rule_id)
                confidence = self.calculate_rule_confidence(agent_id, rule_id)
                
                agent_summary[f'rule_{rule_id}'] = {
                    'statistics': stats,
                    'confidence': confidence
                }
            
            best_rule = self.get_best_rule(agent_id)
            agent_summary['best_rule'] = best_rule
            
            summary[f'agent_{agent_id}'] = agent_summary
        
        return summary
    
    def __str__(self) -> str:
        total_records = sum(
            sum(len(self.performance_db[agent_id][rule_id]) 
                for rule_id in range(self.num_rules))
            for agent_id in range(self.num_agents)
        )
        return f"RulePerformanceTracker(agents={self.num_agents}, rules={self.num_rules}, records={total_records})"
