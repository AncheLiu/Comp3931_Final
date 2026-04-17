import argparse
import os
import subprocess
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASELINE_EXPERIMENTS = [
    "experiments/cartpole_train_dqn.py",
    "experiments/cartpole_train_ddqn.py",
    "experiments/cartpole_train_duelingdqn.py",
    "experiments/cartpole_train_ppo.py",
    "experiments/mountaincar_train_dqn.py",
    "experiments/mountaincar_train_ddqn.py",
    "experiments/mountaincar_train_duelingdqn.py",
    "experiments/mountaincar_train_ppo.py",
    "experiments/acrobot_train_dqn.py",
    "experiments/acrobot_train_ddqn.py",
    "experiments/acrobot_train_duelingdqn.py",
    "experiments/acrobot_train_ppo.py",
    "experiments/lunarlander_train_dqn.py",
    "experiments/lunarlander_train_ddqn.py",
    "experiments/lunarlander_train_duelingdqn.py",
    "experiments/lunarlander_train_ppo.py",
]

IMPROVEMENT_EXPERIMENTS = [
    "experiments/mountaincar_train_dqn_per.py",
    "experiments/mountaincar_train_ddqn_per.py",
    "experiments/mountaincar_train_duelingdqn_per.py",
    "experiments/mountaincar_train_ppo_improved.py",
    "experiments/lunarlander_train_dqn_per.py",
    "experiments/lunarlander_train_ddqn_per.py",
    "experiments/lunarlander_train_duelingdqn_per.py",
    "experiments/lunarlander_train_ppo_improved.py",
]

PER_EXPERIMENTS = [
    "experiments/mountaincar_train_dqn_per.py",
    "experiments/mountaincar_train_ddqn_per.py",
    "experiments/mountaincar_train_duelingdqn_per.py",
    "experiments/lunarlander_train_dqn_per.py",
    "experiments/lunarlander_train_ddqn_per.py",
    "experiments/lunarlander_train_duelingdqn_per.py",
]

PPO_IMPROVED_EXPERIMENTS = [
    "experiments/mountaincar_train_ppo_improved.py",
    "experiments/lunarlander_train_ppo_improved.py",
]

ABLATION_EXPERIMENTS = [
    "experiments/mountaincar_train_dqn_unshaped.py",
    "experiments/mountaincar_train_ddqn_unshaped.py",
    "experiments/mountaincar_train_dqn_per_unshaped.py",
    "experiments/mountaincar_train_ddqn_per_unshaped.py",
]

SENSITIVITY_EXPERIMENTS = [
    "experiments/lunarlander_train_duelingdqn_per_alpha04.py",
    "experiments/lunarlander_train_duelingdqn_per_alpha08.py",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--suite",
        choices=["baseline", "improved", "per", "ppo_improved", "ablation", "sensitivity", "all"],
        default="baseline",
        help="Choose whether to run original baselines, improved experiments, or both.",
    )
    return parser.parse_args()


def select_experiments(suite: str) -> list[str]:
    if suite == "baseline":
        return BASELINE_EXPERIMENTS
    if suite == "improved":
        return IMPROVEMENT_EXPERIMENTS
    if suite == "per":
        return PER_EXPERIMENTS
    if suite == "ppo_improved":
        return PPO_IMPROVED_EXPERIMENTS
    if suite == "ablation":
        return ABLATION_EXPERIMENTS
    if suite == "sensitivity":
        return SENSITIVITY_EXPERIMENTS
    return BASELINE_EXPERIMENTS + IMPROVEMENT_EXPERIMENTS + ABLATION_EXPERIMENTS + SENSITIVITY_EXPERIMENTS


def build_command(script_path: str, args) -> list[str]:
    return [
        sys.executable,
        script_path,
        "--output-dir",
        args.output_dir,
        "--eval-every",
        str(args.eval_every),
        "--eval-episodes",
        str(args.eval_episodes),
        "--device",
        args.device,
        "--seeds",
        *[str(seed) for seed in args.seeds],
    ]


def main():
    args = parse_args()
    experiments = select_experiments(args.suite)
    print("=" * 80)
    print("Reinforcement learning experiment batch runner")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {args.seeds}")
    print(f"Suite: {args.suite}")
    print(f"Results directory: {os.path.join(BASE_DIR, args.output_dir)}")
    print("=" * 80)
    print()

    success_count = 0
    failed_experiments = []

    for index, script in enumerate(experiments, start=1):
        script_path = os.path.join(BASE_DIR, script)
        if not os.path.exists(script_path):
            failed_experiments.append(script)
            print(f"[missing] {script}")
            continue

        print(f"[{index}/{len(experiments)}] Running {script}")
        print("-" * 80)
        try:
            subprocess.run(build_command(script_path, args), cwd=BASE_DIR, check=True)
            success_count += 1
            print(f"[ok] {script}")
        except subprocess.CalledProcessError as exc:
            failed_experiments.append(script)
            print(f"[failed] {script} returned {exc.returncode}")
        print("=" * 80)
        print()

    print("=" * 80)
    print("Experiment summary")
    print(f"Total experiments: {len(experiments)}")
    print(f"Succeeded: {success_count}")
    print(f"Failed: {len(failed_experiments)}")
    if failed_experiments:
        print("Failed scripts:")
        for script in failed_experiments:
            print(f"  - {script}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
