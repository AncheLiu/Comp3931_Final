import argparse
import os
from dataclasses import dataclass

import gymnasium as gym
import torch
from PIL import Image
from torch.distributions import Categorical

from experiment_runner import ActorCritic, DuelingQNetwork, RewardShapingWrapper
from q_network import QNetwork


@dataclass(frozen=True)
class DemoSpec:
    name: str
    env_name: str
    algorithm: str
    model_type: str
    hidden_dim: int
    seed: int = 0
    reward_shaping: bool = False
    mode: str = "greedy"


DEMO_SPECS = [
    DemoSpec("lunarlander_baseline_dqn", "LunarLander-v3", "dqn", "q", 64),
    DemoSpec("lunarlander_dueling_dqn_per", "LunarLander-v3", "dueling_dqn_per", "dueling", 128),
    DemoSpec("mountaincar_unshaped_dqn", "MountainCar-v0", "dqn_unshaped", "q", 64),
    DemoSpec("mountaincar_ddqn_per_shaped", "MountainCar-v0", "ddqn_per", "q", 128, reward_shaping=True),
    DemoSpec("mountaincar_ppo_improved_failure", "MountainCar-v0", "ppo_improved", "ppo", 128, reward_shaping=True),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/demos")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def sanitize_env_name(env_name: str) -> str:
    return env_name.lower().replace("-", "").replace("_", "")


def make_render_env(env_name: str, reward_shaping: bool):
    env = gym.make(env_name, render_mode="rgb_array")
    if reward_shaping:
        env = RewardShapingWrapper(env)
    return env


def build_model(spec: DemoSpec, state_dim: int, action_dim: int):
    if spec.model_type == "dueling":
        return DuelingQNetwork(state_dim, action_dim, hidden_dim=spec.hidden_dim)
    if spec.model_type == "ppo":
        return ActorCritic(state_dim, action_dim, hidden_dim=spec.hidden_dim)
    return QNetwork(state_dim, action_dim, hidden_dim=spec.hidden_dim)


def model_path(results_dir: str, spec: DemoSpec) -> str:
    stem = f"{spec.algorithm}_{sanitize_env_name(spec.env_name)}"
    return os.path.join(results_dir, f"{stem}_seed{spec.seed}.pth")


def select_action(model, model_type: str, state, device, mode: str) -> int:
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if model_type == "ppo":
            probs = model.actor(state_tensor)
            if mode == "stochastic":
                return int(Categorical(probs).sample().item())
            return int(probs.argmax(dim=1).item())
        return int(model(state_tensor).argmax(dim=1).item())


def save_gif(frames: list, output_path: str, fps: int) -> None:
    images = [Image.fromarray(frame) for frame in frames]
    duration_ms = int(1000 / fps)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    for spec in DEMO_SPECS:
        path = model_path(args.results_dir, spec)
        if not os.path.exists(path):
            print(f"[missing] {path}")
            continue

        env = make_render_env(spec.env_name, spec.reward_shaping)
        model = build_model(spec, env.observation_space.shape[0], env.action_space.n).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        state, _ = env.reset(seed=400_000 + spec.seed)
        frames = []
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < args.max_steps:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            action = select_action(model, spec.model_type, state, device, spec.mode)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        env.close()
        if not frames:
            print(f"[skipped] no frames captured for {spec.name}")
            continue

        output_path = os.path.join(args.output_dir, f"{spec.name}_seed{spec.seed}.gif")
        save_gif(frames, output_path, args.fps)
        print(f"[ok] {output_path} steps={steps} reward={total_reward:.2f}")

    print(f"demos saved to {args.output_dir}")


if __name__ == "__main__":
    main()
