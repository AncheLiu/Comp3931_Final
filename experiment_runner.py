import argparse
import csv
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from q_network import QNetwork
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


@dataclass
class ValueExperimentConfig:
    env_name: str
    algorithm: str
    episodes: int
    reward_shaping: bool = False
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    memory_capacity: int = 50_000
    target_update: int = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.985
    hidden_dim: int = 64
    dueling_hidden_dim: int = 128
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_epsilon: float = 1e-5
    loss_function: str = "mse"
    grad_clip_norm: float | None = None


@dataclass
class PPOExperimentConfig:
    env_name: str
    algorithm: str
    episodes: int
    reward_shaping: bool = False
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    k_epochs: int = 4
    eps_clip: float = 0.2
    hidden_dim: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    use_gae: bool = False
    gae_lambda: float = 0.95
    normalize_advantages: bool = False
    minibatch_size: int | None = None
    max_grad_norm: float | None = None


class RewardShapingWrapper(gym.RewardWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.max_position = -0.4

    def reward(self, reward):
        position = self.env.unwrapped.state[0]
        if position > self.max_position:
            self.max_position = position
            reward += (position + 0.5) * 10
        return reward

    def reset(self, **kwargs):
        self.max_position = -0.4
        return self.env.reset(**kwargs)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Linear(state_dim, hidden_dim)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        features = torch.relu(self.feature(x))
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def act(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, states, actions):
        probs = self.actor(states)
        dist = Categorical(probs)
        return dist.log_prob(actions), self.critic(states).squeeze(-1), dist.entropy()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args(default_seeds: list[int] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=default_seeds or [0, 1, 2])
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def make_env_factory(env_name: str, reward_shaping: bool) -> Callable[[], gym.Env]:
    def factory():
        env = gym.make(env_name)
        if reward_shaping:
            env = RewardShapingWrapper(env)
        return env

    return factory


def get_value_method(algorithm: str) -> str:
    if algorithm.startswith("dueling_dqn"):
        return "dueling_dqn"
    if algorithm.startswith("ddqn"):
        return "ddqn"
    return "dqn"


def sanitize_env_name(env_name: str) -> str:
    return env_name.lower().replace("-", "").replace("_", "")


def compute_per_beta(config: ValueExperimentConfig, episode: int) -> float:
    if config.episodes <= 1:
        return config.per_beta_end
    progress = (episode - 1) / (config.episodes - 1)
    return config.per_beta_start + progress * (config.per_beta_end - config.per_beta_start)


def evaluate_value_model(
    make_env: Callable[[], gym.Env], model: nn.Module, device: torch.device, episodes: int
) -> float:
    rewards = []
    model.eval()
    for eval_idx in range(episodes):
        env = make_env()
        state, _ = env.reset(seed=10_000 + eval_idx)
        total_reward = 0.0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(model(state_tensor).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        env.close()
    model.train()
    return float(np.mean(rewards))


def evaluate_ppo_model(
    make_env: Callable[[], gym.Env], model: ActorCritic, device: torch.device, episodes: int
) -> tuple[float, float]:
    greedy_rewards = []
    stochastic_rewards = []
    model.eval()
    for eval_idx in range(episodes):
        for mode in ("greedy", "stochastic"):
            env = make_env()
            state, _ = env.reset(seed=20_000 + eval_idx)
            total_reward = 0.0
            done = False
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
            if mode == "greedy":
                greedy_rewards.append(total_reward)
            else:
                stochastic_rewards.append(total_reward)
            env.close()
    model.train()
    return float(np.mean(greedy_rewards)), float(np.mean(stochastic_rewards))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_seed_curves(
    rows: list[dict], metric_key: str, title: str, ylabel: str, output_path: str
) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seed"]].append(row)

    plt.figure(figsize=(10, 6))
    for seed, seed_rows in sorted(grouped.items()):
        seed_rows = sorted(seed_rows, key=lambda item: item["episode"])
        episodes = [item["episode"] for item in seed_rows]
        values = [item[metric_key] for item in seed_rows]
        plt.plot(episodes, values, label=f"seed={seed}")

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def summarize_results(rows: list[dict], metric_key: str) -> list[dict]:
    if not rows:
        return []

    final_values = defaultdict(list)
    for row in rows:
        final_values[row["seed"]].append(row)

    summary_rows = []
    seed_scores = []
    train_scores = []
    for seed, seed_rows in sorted(final_values.items()):
        final_row = max(seed_rows, key=lambda item: item["episode"])
        train_scores.append(final_row["train_reward"])
        seed_scores.append(final_row[metric_key])
        summary_rows.append(
            {
                "seed": seed,
                "final_train_reward": round(final_row["train_reward"], 4),
                "final_eval_reward": round(final_row["eval_reward"], 4),
            }
        )

    summary_rows.append(
        {
            "seed": "mean",
            "final_train_reward": round(float(np.mean(train_scores)), 4),
            "final_eval_reward": round(float(np.mean(seed_scores)), 4),
        }
    )
    summary_rows.append(
        {
            "seed": "std",
            "final_train_reward": round(float(np.std(train_scores)), 4),
            "final_eval_reward": round(float(np.std(seed_scores)), 4),
        }
    )
    return summary_rows


def train_value_experiment(config: ValueExperimentConfig, args) -> None:
    device = torch.device(args.device)
    ensure_dir(args.output_dir)
    make_env = make_env_factory(config.env_name, config.reward_shaping)
    all_rows = []

    for seed in args.seeds:
        set_seed(seed)
        env = make_env()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        value_method = get_value_method(config.algorithm)

        if value_method == "dueling_dqn":
            policy_net = DuelingQNetwork(state_dim, action_dim, hidden_dim=config.dueling_hidden_dim).to(device)
            target_net = DuelingQNetwork(state_dim, action_dim, hidden_dim=config.dueling_hidden_dim).to(device)
        else:
            policy_net = QNetwork(state_dim, action_dim, hidden_dim=config.hidden_dim).to(device)
            target_net = QNetwork(state_dim, action_dim, hidden_dim=config.hidden_dim).to(device)

        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate)
        if config.use_per:
            memory = PrioritizedReplayBuffer(
                config.memory_capacity,
                alpha=config.per_alpha,
                epsilon=config.per_epsilon,
            )
        else:
            memory = ReplayBuffer(config.memory_capacity)
        epsilon = config.epsilon_start
        seed_rows = []

        for episode in range(1, config.episodes + 1):
            state, _ = env.reset(seed=seed * 1000 + episode)
            done = False
            total_reward = 0.0

            while not done:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = int(policy_net(state_tensor).argmax(dim=1).item())

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(memory) > config.batch_size:
                    if config.use_per:
                        beta = compute_per_beta(config, episode)
                        batch = memory.sample(config.batch_size, beta=beta)
                        indices = batch[5]
                        weights = torch.tensor(batch[6], dtype=torch.float32, device=device).unsqueeze(1)
                    else:
                        batch = memory.sample(config.batch_size)
                        indices = None
                        weights = torch.ones((config.batch_size, 1), dtype=torch.float32, device=device)

                    states = torch.tensor(batch[0], dtype=torch.float32, device=device)
                    actions = torch.tensor(batch[1], dtype=torch.long, device=device).unsqueeze(1)
                    rewards = torch.tensor(batch[2], dtype=torch.float32, device=device).unsqueeze(1)
                    next_states = torch.tensor(batch[3], dtype=torch.float32, device=device)
                    dones = torch.tensor(batch[4], dtype=torch.float32, device=device).unsqueeze(1)

                    current_q = policy_net(states).gather(1, actions)
                    with torch.no_grad():
                        if value_method == "dqn":
                            max_next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
                        else:
                            next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
                            max_next_q = target_net(next_states).gather(1, next_actions)
                        target_q = rewards + (1 - dones) * config.gamma * max_next_q

                    if config.loss_function == "huber":
                        td_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
                    else:
                        td_loss = F.mse_loss(current_q, target_q, reduction="none")
                    loss = (td_loss * weights).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    if config.grad_clip_norm is not None:
                        nn.utils.clip_grad_norm_(policy_net.parameters(), config.grad_clip_norm)
                    optimizer.step()
                    if config.use_per and indices is not None:
                        td_errors = (target_q - current_q).detach().abs().squeeze(1).cpu().numpy()
                        memory.update_priorities(indices, td_errors)

            epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
            if episode % config.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            eval_reward = np.nan
            if episode % args.eval_every == 0 or episode == config.episodes:
                eval_reward = evaluate_value_model(make_env, policy_net, device, args.eval_episodes)

            row = {
                "environment": config.env_name,
                "algorithm": config.algorithm,
                "seed": seed,
                "episode": episode,
                "train_reward": round(total_reward, 4),
                "eval_reward": round(eval_reward, 4) if not np.isnan(eval_reward) else "",
                "epsilon": round(epsilon, 6),
            }
            seed_rows.append(row)

            if episode % 10 == 0:
                eval_text = row["eval_reward"] if row["eval_reward"] != "" else "pending"
                print(
                    f"[{config.algorithm}] env={config.env_name} seed={seed} "
                    f"episode={episode}/{config.episodes} train={total_reward:.2f} eval={eval_text}"
                )

        env.close()
        all_rows.extend(seed_rows)
        model_name = f"{config.algorithm}_{sanitize_env_name(config.env_name)}_seed{seed}.pth"
        torch.save(policy_net.state_dict(), os.path.join(args.output_dir, model_name))

    stem = f"{config.algorithm}_{sanitize_env_name(config.env_name)}"
    save_csv(
        os.path.join(args.output_dir, f"{stem}_metrics.csv"),
        all_rows,
        ["environment", "algorithm", "seed", "episode", "train_reward", "eval_reward", "epsilon"],
    )
    summary_rows = summarize_results(all_rows, "eval_reward")
    save_csv(
        os.path.join(args.output_dir, f"{stem}_summary.csv"),
        summary_rows,
        ["seed", "final_train_reward", "final_eval_reward"],
    )
    plot_seed_curves(
        all_rows,
        "train_reward",
        f"{config.algorithm.upper()} Training Curve on {config.env_name}",
        "Training Reward",
        os.path.join(args.output_dir, f"{stem}_training_curve.png"),
    )
    eval_rows = [row for row in all_rows if row["eval_reward"] != ""]
    if eval_rows:
        plot_seed_curves(
            eval_rows,
            "eval_reward",
            f"{config.algorithm.upper()} Evaluation Curve on {config.env_name}",
            "Evaluation Reward",
            os.path.join(args.output_dir, f"{stem}_evaluation_curve.png"),
        )


def train_ppo_experiment(config: PPOExperimentConfig, args) -> None:
    device = torch.device(args.device)
    ensure_dir(args.output_dir)
    make_env = make_env_factory(config.env_name, config.reward_shaping)
    all_rows = []

    for seed in args.seeds:
        set_seed(seed)
        env = make_env()
        model = ActorCritic(env.observation_space.shape[0], env.action_space.n, hidden_dim=config.hidden_dim).to(device)
        optimizer = optim.Adam(
            [
                {"params": model.actor.parameters(), "lr": config.lr_actor},
                {"params": model.critic.parameters(), "lr": config.lr_critic},
            ]
        )
        seed_rows = []

        for episode in range(1, config.episodes + 1):
            states, actions, logprobs, rewards, terminals = [], [], [], [], []
            state, _ = env.reset(seed=seed * 1000 + episode)
            done = False
            total_reward = 0.0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                action, logprob = model.act(state_tensor)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state_tensor)
                actions.append(torch.tensor(action, device=device))
                logprobs.append(logprob)
                rewards.append(reward)
                terminals.append(done)

                state = next_state
                total_reward += reward

            old_states = torch.stack(states).detach()
            old_actions = torch.stack(actions).detach()
            old_logprobs = torch.stack(logprobs).detach()

            with torch.no_grad():
                old_values = model.critic(old_states).squeeze(-1)

            if config.use_gae:
                advantages_list = []
                gae = 0.0
                next_value = 0.0
                for step in reversed(range(len(rewards))):
                    mask = 0.0 if terminals[step] else 1.0
                    delta = rewards[step] + config.gamma * next_value * mask - old_values[step].item()
                    gae = delta + config.gamma * config.gae_lambda * mask * gae
                    advantages_list.insert(0, gae)
                    next_value = old_values[step].item()
                advantages = torch.tensor(advantages_list, dtype=torch.float32, device=device)
                returns = advantages + old_values
            else:
                returns_list = []
                discounted = 0.0
                for reward, terminal in zip(reversed(rewards), reversed(terminals)):
                    if terminal:
                        discounted = 0.0
                    discounted = reward + config.gamma * discounted
                    returns_list.insert(0, discounted)
                returns = torch.tensor(returns_list, dtype=torch.float32, device=device)
                advantages = returns - old_values

            if config.normalize_advantages and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            minibatch_size = config.minibatch_size or len(old_states)
            minibatch_size = max(1, min(minibatch_size, len(old_states)))

            for _ in range(config.k_epochs):
                permutation = torch.randperm(len(old_states), device=device)
                for start in range(0, len(old_states), minibatch_size):
                    batch_indices = permutation[start : start + minibatch_size]
                    new_logprobs, state_values, entropy = model.evaluate(
                        old_states[batch_indices], old_actions[batch_indices]
                    )
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                    ratios = torch.exp(new_logprobs - old_logprobs[batch_indices])
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - config.eps_clip, 1 + config.eps_clip) * batch_advantages
                    value_loss = nn.MSELoss()(state_values, batch_returns)
                    loss = -torch.min(surr1, surr2) + config.value_coef * value_loss - config.entropy_coef * entropy

                    optimizer.zero_grad()
                    loss.mean().backward()
                    if config.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

            eval_reward_greedy = np.nan
            eval_reward_stochastic = np.nan
            if episode % args.eval_every == 0 or episode == config.episodes:
                eval_reward_greedy, eval_reward_stochastic = evaluate_ppo_model(
                    make_env, model, device, args.eval_episodes
                )

            row = {
                "environment": config.env_name,
                "algorithm": config.algorithm,
                "seed": seed,
                "episode": episode,
                "train_reward": round(total_reward, 4),
                "eval_reward": round(eval_reward_greedy, 4) if not np.isnan(eval_reward_greedy) else "",
                "eval_reward_greedy": round(eval_reward_greedy, 4) if not np.isnan(eval_reward_greedy) else "",
                "eval_reward_stochastic": (
                    round(eval_reward_stochastic, 4) if not np.isnan(eval_reward_stochastic) else ""
                ),
            }
            seed_rows.append(row)

            if episode % 10 == 0:
                greedy_text = row["eval_reward_greedy"] if row["eval_reward_greedy"] != "" else "pending"
                stochastic_text = (
                    row["eval_reward_stochastic"] if row["eval_reward_stochastic"] != "" else "pending"
                )
                print(
                    f"[{config.algorithm}] env={config.env_name} seed={seed} "
                    f"episode={episode}/{config.episodes} train={total_reward:.2f} "
                    f"eval_greedy={greedy_text} eval_stochastic={stochastic_text}"
                )

        env.close()
        all_rows.extend(seed_rows)
        model_name = f"{config.algorithm}_{sanitize_env_name(config.env_name)}_seed{seed}.pth"
        torch.save(model.state_dict(), os.path.join(args.output_dir, model_name))

    stem = f"{config.algorithm}_{sanitize_env_name(config.env_name)}"
    save_csv(
        os.path.join(args.output_dir, f"{stem}_metrics.csv"),
        all_rows,
        [
            "environment",
            "algorithm",
            "seed",
            "episode",
            "train_reward",
            "eval_reward",
            "eval_reward_greedy",
            "eval_reward_stochastic",
        ],
    )
    summary_rows = summarize_results(all_rows, "eval_reward")
    save_csv(
        os.path.join(args.output_dir, f"{stem}_summary.csv"),
        summary_rows,
        ["seed", "final_train_reward", "final_eval_reward"],
    )
    stochastic_summary_rows = summarize_results(all_rows, "eval_reward_stochastic")
    save_csv(
        os.path.join(args.output_dir, f"{stem}_summary_stochastic.csv"),
        stochastic_summary_rows,
        ["seed", "final_train_reward", "final_eval_reward"],
    )
    plot_seed_curves(
        all_rows,
        "train_reward",
        f"{config.algorithm.upper()} Training Curve on {config.env_name}",
        "Training Reward",
        os.path.join(args.output_dir, f"{stem}_training_curve.png"),
    )
    eval_rows = [row for row in all_rows if row["eval_reward"] != ""]
    if eval_rows:
        plot_seed_curves(
            eval_rows,
            "eval_reward_greedy",
            f"{config.algorithm.upper()} Greedy Evaluation Curve on {config.env_name}",
            "Greedy Evaluation Reward",
            os.path.join(args.output_dir, f"{stem}_evaluation_curve.png"),
        )
        plot_seed_curves(
            eval_rows,
            "eval_reward_stochastic",
            f"{config.algorithm.upper()} Stochastic Evaluation Curve on {config.env_name}",
            "Stochastic Evaluation Reward",
            os.path.join(args.output_dir, f"{stem}_evaluation_curve_stochastic.png"),
        )
