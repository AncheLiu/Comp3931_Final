import argparse
import csv
import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from experiment_runner import ActorCritic, DuelingQNetwork, RewardShapingWrapper
from q_network import QNetwork


@dataclass(frozen=True)
class ModelSpec:
    name: str
    env_name: str
    algorithm: str
    model_type: str
    hidden_dim: int
    reward_shaping: bool = False


MODEL_SPECS = [
    ModelSpec("lunarlander_dqn_baseline", "LunarLander-v3", "dqn", "q", 64),
    ModelSpec("lunarlander_dqn_per", "LunarLander-v3", "dqn_per", "q", 128),
    ModelSpec("lunarlander_ddqn_per", "LunarLander-v3", "ddqn_per", "q", 128),
    ModelSpec("lunarlander_dueling_dqn_per", "LunarLander-v3", "dueling_dqn_per", "dueling", 128),
    ModelSpec("lunarlander_ppo_improved", "LunarLander-v3", "ppo_improved", "ppo", 128),
    ModelSpec("mountaincar_dqn_baseline_shaped", "MountainCar-v0", "dqn", "q", 64, reward_shaping=True),
    ModelSpec("mountaincar_dqn_per_shaped", "MountainCar-v0", "dqn_per", "q", 128, reward_shaping=True),
    ModelSpec("mountaincar_ddqn_per_shaped", "MountainCar-v0", "ddqn_per", "q", 128, reward_shaping=True),
    ModelSpec("mountaincar_ppo_improved_shaped", "MountainCar-v0", "ppo_improved", "ppo", 128, reward_shaping=True),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sanitize_env_name(env_name: str) -> str:
    return env_name.lower().replace("-", "").replace("_", "")


def make_env(env_name: str, reward_shaping: bool):
    env = gym.make(env_name)
    if reward_shaping:
        env = RewardShapingWrapper(env)
    return env


def build_model(spec: ModelSpec, state_dim: int, action_dim: int):
    if spec.model_type == "dueling":
        return DuelingQNetwork(state_dim, action_dim, hidden_dim=spec.hidden_dim)
    if spec.model_type == "ppo":
        return ActorCritic(state_dim, action_dim, hidden_dim=spec.hidden_dim)
    return QNetwork(state_dim, action_dim, hidden_dim=spec.hidden_dim)


def model_path(results_dir: str, spec: ModelSpec, seed: int) -> str:
    stem = f"{spec.algorithm}_{sanitize_env_name(spec.env_name)}"
    return os.path.join(results_dir, f"{stem}_seed{seed}.pth")


def evaluate_value_model(env_name: str, reward_shaping: bool, model, device, episodes: int, seed_offset: int):
    rewards = []
    model.eval()
    for episode in range(episodes):
        env = make_env(env_name, reward_shaping)
        state, _ = env.reset(seed=seed_offset + episode)
        done = False
        total_reward = 0.0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(model(state_tensor).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        env.close()
        rewards.append(total_reward)
    return rewards


def evaluate_ppo_model(env_name: str, reward_shaping: bool, model, device, episodes: int, seed_offset: int, mode: str):
    rewards = []
    model.eval()
    for episode in range(episodes):
        env = make_env(env_name, reward_shaping)
        state, _ = env.reset(seed=seed_offset + episode)
        done = False
        total_reward = 0.0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                probs = model.actor(state_tensor)
            if mode == "greedy":
                action = int(probs.argmax(dim=1).item())
            else:
                action = int(Categorical(probs).sample().item())
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        env.close()
        rewards.append(total_reward)
    return rewards


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.results_dir, "final_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(args.device)

    summary_rows = []
    episode_rows = []
    for spec in MODEL_SPECS:
        probe_env = make_env(spec.env_name, spec.reward_shaping)
        state_dim = probe_env.observation_space.shape[0]
        action_dim = probe_env.action_space.n
        probe_env.close()

        for seed in args.seeds:
            path = model_path(args.results_dir, spec, seed)
            if not os.path.exists(path):
                print(f"[missing] {path}")
                continue

            model = build_model(spec, state_dim, action_dim).to(device)
            model.load_state_dict(torch.load(path, map_location=device))

            modes = ["greedy", "stochastic"] if spec.model_type == "ppo" else ["greedy"]
            for mode in modes:
                seed_offset = 300_000 + seed * 10_000
                if spec.model_type == "ppo":
                    rewards = evaluate_ppo_model(
                        spec.env_name, spec.reward_shaping, model, device, args.eval_episodes, seed_offset, mode
                    )
                else:
                    rewards = evaluate_value_model(
                        spec.env_name, spec.reward_shaping, model, device, args.eval_episodes, seed_offset
                    )

                mean_reward = float(np.mean(rewards))
                std_reward = float(np.std(rewards))
                summary_rows.append(
                    {
                        "model": spec.name,
                        "environment": spec.env_name,
                        "algorithm": spec.algorithm,
                        "seed": seed,
                        "mode": mode,
                        "episodes": args.eval_episodes,
                        "mean_reward": round(mean_reward, 4),
                        "std_reward": round(std_reward, 4),
                    }
                )
                for index, reward in enumerate(rewards, start=1):
                    episode_rows.append(
                        {
                            "model": spec.name,
                            "seed": seed,
                            "mode": mode,
                            "episode": index,
                            "reward": round(float(reward), 4),
                        }
                    )
                print(f"[ok] {spec.name} seed={seed} mode={mode} mean={mean_reward:.2f}")

    write_csv(
        os.path.join(output_dir, "final_evaluation_summary.csv"),
        summary_rows,
        ["model", "environment", "algorithm", "seed", "mode", "episodes", "mean_reward", "std_reward"],
    )
    write_csv(
        os.path.join(output_dir, "final_evaluation_episodes.csv"),
        episode_rows,
        ["model", "seed", "mode", "episode", "reward"],
    )
    print(f"final evaluation saved to {output_dir}")


if __name__ == "__main__":
    main()
