import argparse
import numpy as np

from algorithms.reinforce import ReinforceAgent
from envs.serial_env import SerialSupplyChain
from envs.meta_wrapper import MetaSerialEnv


def train_meta_agent(
    num_episodes=1000,
    max_steps=52,
    lr=1e-2,
    reward_scale=0.01,
    log_interval=50,
    batch_size=16,
    temp_start=2.0,
    temp_end=0.5,
    eval_interval=100,
    eval_episodes=10,
    seed=42,
):
    np.random.seed(seed)
    base_env = SerialSupplyChain(config={"max_steps": max_steps})
    meta_env = MetaSerialEnv(base_env)

    input_dim = meta_env.observation_space.shape[0]
    n_actions = meta_env.action_space.n

    agent = ReinforceAgent(input_dim, n_actions, lr=lr, batch_size=batch_size)

    all_rewards = []
    rule_counts = {}

    print("Bắt đầu training Meta-Agent trên Serial 1-1-1...")
    print(f"Danh sách Rules: {[r.name for r in meta_env.rules]}")

    for episode in range(num_episodes):
        state = meta_env.reset()
        progress = episode / max(1, num_episodes - 1)
        temperature = temp_start + (temp_end - temp_start) * progress

        action_idx, log_prob, entropy = agent.select_action(state, temperature=temperature)
        _, reward, _, info = meta_env.step(action_idx)
        scaled_reward = reward * reward_scale
        loss = agent.record(log_prob, entropy, scaled_reward)

        all_rewards.append(reward)
        rule_name = info["rule_name"]
        rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1

        if episode % log_interval == 0:
            avg_r = np.mean(all_rewards[-log_interval:]) if all_rewards else 0.0
            print(
                f"Ep {episode}: Reward {reward:.1f} | Rule: {rule_name} | "
                f"Temp: {temperature:.2f}"
            )
            print(f"  Avg Reward({log_interval}): {avg_r:.1f}")
            if loss is not None:
                print(f"  Batch Loss: {loss:.4f}")

        if eval_interval and (episode + 1) % eval_interval == 0:
            eval_rewards = []
            for _ in range(eval_episodes):
                eval_state = meta_env.reset()
                eval_action, _, _ = agent.select_action(eval_state, temperature=0.1)
                _, eval_reward, _, _ = meta_env.step(eval_action)
                eval_rewards.append(eval_reward)
            eval_avg = float(np.mean(eval_rewards)) if eval_rewards else 0.0
            print(f"  Eval Avg Reward({eval_episodes}): {eval_avg:.1f}")

    print("\n--- Training Complete ---")
    print("Phân phối Rule đã chọn:", rule_counts)


def build_parser():
    parser = argparse.ArgumentParser(description="Meta-learning serial baseline")
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=52)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_meta_agent(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        lr=args.lr,
        reward_scale=args.reward_scale,
        log_interval=args.log_interval,
        batch_size=args.batch_size,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )
