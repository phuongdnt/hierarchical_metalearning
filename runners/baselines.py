#!/usr/bin/env python
"""Baseline runners for fixed rule policies (FOQ, POQ, SM)."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.serial import Env as SerialEnv
from envs.hierarchical_env import HierarchicalSupplyChainEnv

# ✅ UPDATED: Add hierarchical env
ENV_REGISTRY = {
    'serial': SerialEnv,
    'hierarchical': HierarchicalSupplyChainEnv,
}

POLICY_TO_ACTION = {
    'FOQ': 0,
    'POQ': 1,
    'SM': 2,
}


def run_baseline(env_name: str, policy: str, episodes: int, seed: int, scenario_name: str):
    """
    Run fixed-rule baseline (pure FOQ, POQ, or SM).
    
    Args:
        env_name: 'serial' or 'hierarchical'
        policy: 'FOQ', 'POQ', or 'SM'
        episodes: Number of episodes to run
        seed: Random seed
        scenario_name: Name for results folder
    """
    policy_key = policy.upper()
    if policy_key not in POLICY_TO_ACTION:
        raise ValueError(f"Unsupported policy {policy}")
    
    # Get environment class
    env_cls = ENV_REGISTRY.get(env_name, SerialEnv)
    
    # Create environment
    if env_name == 'hierarchical':
        # Need to create args for hierarchical env
        from types import SimpleNamespace
        args = SimpleNamespace(
            n_agents=3,
            lead_time=4,
            episode_length=200,
            use_hierarchical=False,  # Don't use hierarchical learning for baseline
            seed=seed,
            # Add other required params...
        )
        env = env_cls(args)
    else:
        env = env_cls()
    
    np.random.seed(seed)
    
    results: List[Dict[str, float]] = []
    
    print(f"\n{'='*70}")
    print(f"RUNNING FIXED RULE BASELINE: {policy_key}")
    print(f"{'='*70}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {episodes}")
    print(f"Seed: {seed}")
    print(f"{'='*70}\n")
    
    for episode in range(episodes):
        env.reset(train=False) if hasattr(env, 'reset') else env.reset()
        total_reward = np.zeros(env.agent_num, dtype=np.float32)
        
        step = 0
        while step < 200:  # Episode length
            # Fixed rule - always use same action
            action_index = POLICY_TO_ACTION[policy_key]
            actions = [[action_index] for _ in range(env.agent_num)]
            
            # One-hot encoding
            actions_one_hot = []
            for _ in range(env.agent_num):
                one_hot = np.zeros(3)
                one_hot[action_index] = 1
                actions_one_hot.append(one_hot)
            
            obs, reward, done, infos = env.step(actions_one_hot)
            
            # Convert reward to array
            if isinstance(reward, list):
                reward_arr = np.array([r[0] if isinstance(r, list) else r for r in reward], dtype=np.float32)
            else:
                reward_arr = np.array(reward, dtype=np.float32)
            
            total_reward += reward_arr
            step += 1
            
            if any(done):
                break
        
        # Collect episode metrics
        episode_result = {
            "episode": episode,
            "total_reward": float(np.sum(total_reward)),
            "avg_reward": float(np.mean(total_reward)),
        }
        
        # Add metrics if available
        if len(infos) > 0 and isinstance(infos[0], dict):
            metrics = infos[0].get("episode_metrics", {})
            episode_result["service_level"] = float(metrics.get("service_level", 0.0))
            episode_result["bullwhip_mean"] = float(np.mean(metrics.get("bullwhip_index", []))) \
                if metrics.get("bullwhip_index") else 0.0
            episode_result["order_variance_mean"] = float(np.mean(metrics.get("order_variance", []))) \
                if metrics.get("order_variance") else 0.0
        
        results.append(episode_result)
        
        print(f"Episode {episode}: Total reward = {episode_result['total_reward']:.2f}")
    
    # Save results
    base_dir = Path(__file__).resolve().parents[0]
    results_dir = base_dir / "results" / env_name / scenario_name / "baselines"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{policy_key.lower()}_baseline.csv"
    
    fieldnames = list(results[0].keys()) if results else ["episode", "total_reward"]
    
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary
    avg_total_reward = np.mean([r["total_reward"] for r in results])
    print(f"\n{'='*70}")
    print(f"BASELINE {policy_key} COMPLETE")
    print(f"{'='*70}")
    print(f"Average total reward: {avg_total_reward:.2f}")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*70}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-rule baselines")
    parser.add_argument("--env_name", type=str, default="hierarchical",
                       choices=['serial', 'hierarchical'])
    parser.add_argument("--scenario_name", type=str, default="RuleSelection")
    parser.add_argument("--policy", type=str, default="FOQ", 
                       choices=list(POLICY_TO_ACTION.keys()))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline(args.env_name, args.policy, args.episodes, args.seed, args.scenario_name)
