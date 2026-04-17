import argparse
import csv
import os
from collections import defaultdict


ENVIRONMENT_STEMS = {
    "CartPole-v1": "cartpolev1",
    "MountainCar-v0": "mountaincarv0",
    "Acrobot-v1": "acrobotv1",
    "LunarLander-v3": "lunarlanderv3",
}

BASELINE_ALGORITHMS = {"dqn", "ddqn", "dueling_dqn", "ppo"}
IMPROVED_ALGORITHMS = {"dqn_per", "ddqn_per", "dueling_dqn_per", "ppo_improved"}
SAMPLE_THRESHOLDS = {
    "cartpolev1": [475.0],
    "acrobotv1": [-100.0],
    "mountaincarv0": [0.0, 50.0],
    "lunarlanderv3": [100.0, 200.0],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def parse_result_stem(stem: str) -> tuple[str, str]:
    for environment in sorted(ENVIRONMENT_STEMS.values(), key=len, reverse=True):
        suffix = f"_{environment}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], environment
    parts = stem.split("_")
    return parts[0], "_".join(parts[1:])


def load_summary(results_dir: str) -> dict[tuple[str, str], dict]:
    summary = {}
    for filename in sorted(os.listdir(results_dir)):
        if not filename.endswith("_summary.csv") or filename.endswith("_summary_stochastic.csv"):
            continue

        algorithm, environment = parse_result_stem(filename.replace("_summary.csv", ""))
        path = os.path.join(results_dir, filename)
        with open(path, encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

        mean_row = next((row for row in rows if row["seed"] == "mean"), None)
        std_row = next((row for row in rows if row["seed"] == "std"), None)
        if mean_row is None or std_row is None:
            continue

        summary[(environment, algorithm)] = {
            "mean_train": float(mean_row["final_train_reward"]),
            "mean_eval": float(mean_row["final_eval_reward"]),
            "std_train": float(std_row["final_train_reward"]),
            "std_eval": float(std_row["final_eval_reward"]),
            "rows": rows,
        }
    return summary


def load_metrics(results_dir: str) -> dict[tuple[str, str], list[dict]]:
    metrics = {}
    for filename in sorted(os.listdir(results_dir)):
        if not filename.endswith("_metrics.csv"):
            continue

        algorithm, environment = parse_result_stem(filename.replace("_metrics.csv", ""))
        path = os.path.join(results_dir, filename)
        with open(path, encoding="utf-8") as csv_file:
            metrics[(environment, algorithm)] = list(csv.DictReader(csv_file))
    return metrics


def best_algorithm(summary: dict, environment: str, candidates: set[str]) -> tuple[str | None, dict | None]:
    available = [
        (algorithm, values)
        for (env, algorithm), values in summary.items()
        if env == environment and algorithm in candidates
    ]
    if not available:
        return None, None
    return max(available, key=lambda item: item[1]["mean_eval"])


def mean_eval_by_episode(rows: list[dict]) -> dict[int, float]:
    grouped = defaultdict(list)
    for row in rows:
        value = row.get("eval_reward", "")
        if value == "":
            continue
        grouped[int(row["episode"])].append(float(value))
    return {
        episode: sum(values) / len(values)
        for episode, values in grouped.items()
        if values
    }


def first_episode_reaching(rows: list[dict], threshold: float) -> int | None:
    episode_means = mean_eval_by_episode(rows)
    for episode in sorted(episode_means):
        if episode_means[episode] >= threshold:
            return episode
    return None


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_baseline_vs_improved(summary: dict) -> list[dict]:
    environments = sorted({environment for environment, _ in summary})
    rows = []
    for environment in environments:
        baseline_algorithm, baseline = best_algorithm(summary, environment, BASELINE_ALGORITHMS)
        improved_algorithm, improved = best_algorithm(summary, environment, IMPROVED_ALGORITHMS)
        if baseline is None or improved is None:
            continue

        absolute_gain = improved["mean_eval"] - baseline["mean_eval"]
        if abs(baseline["mean_eval"]) > 1e-8:
            relative_gain = absolute_gain / abs(baseline["mean_eval"])
        else:
            relative_gain = None

        rows.append(
            {
                "environment": environment,
                "best_baseline": baseline_algorithm,
                "baseline_mean_eval": round(baseline["mean_eval"], 4),
                "baseline_std_eval": round(baseline["std_eval"], 4),
                "best_improved": improved_algorithm,
                "improved_mean_eval": round(improved["mean_eval"], 4),
                "improved_std_eval": round(improved["std_eval"], 4),
                "absolute_gain": round(absolute_gain, 4),
                "relative_gain": "" if relative_gain is None else round(relative_gain, 4),
            }
        )
    return rows


def build_sample_efficiency(metrics: dict) -> list[dict]:
    rows = []
    for (environment, algorithm), metric_rows in sorted(metrics.items()):
        for threshold in SAMPLE_THRESHOLDS.get(environment, []):
            episode = first_episode_reaching(metric_rows, threshold)
            rows.append(
                {
                    "environment": environment,
                    "algorithm": algorithm,
                    "threshold": threshold,
                    "first_episode": "not reached" if episode is None else episode,
                }
            )
    return rows


def build_stability(summary: dict) -> list[dict]:
    rows = []
    for (environment, algorithm), values in sorted(summary.items()):
        rows.append(
            {
                "environment": environment,
                "algorithm": algorithm,
                "mean_eval": round(values["mean_eval"], 4),
                "std_eval": round(values["std_eval"], 4),
            }
        )
    return sorted(rows, key=lambda row: (row["environment"], row["std_eval"]))


def build_reward_shaping_ablation(summary: dict) -> list[dict]:
    pairs = [
        ("dqn", "dqn_unshaped"),
        ("ddqn", "ddqn_unshaped"),
        ("dqn_per", "dqn_per_unshaped"),
        ("ddqn_per", "ddqn_per_unshaped"),
    ]
    rows = []
    environment = "mountaincarv0"
    for shaped_algorithm, unshaped_algorithm in pairs:
        shaped = summary.get((environment, shaped_algorithm))
        unshaped = summary.get((environment, unshaped_algorithm))
        if shaped is None and unshaped is None:
            continue
        rows.append(
            {
                "algorithm_pair": shaped_algorithm,
                "shaped_mean_eval": "" if shaped is None else round(shaped["mean_eval"], 4),
                "unshaped_mean_eval": "" if unshaped is None else round(unshaped["mean_eval"], 4),
                "shaping_gain": (
                    ""
                    if shaped is None or unshaped is None
                    else round(shaped["mean_eval"] - unshaped["mean_eval"], 4)
                ),
            }
        )
    return rows


def build_per_alpha_sensitivity(summary: dict) -> list[dict]:
    environment = "lunarlanderv3"
    variants = [
        ("0.4", "dueling_dqn_per_alpha04"),
        ("0.6", "dueling_dqn_per"),
        ("0.8", "dueling_dqn_per_alpha08"),
    ]
    rows = []
    for alpha, algorithm in variants:
        values = summary.get((environment, algorithm))
        rows.append(
            {
                "environment": environment,
                "algorithm": algorithm,
                "per_alpha": alpha,
                "mean_eval": "" if values is None else round(values["mean_eval"], 4),
                "std_eval": "" if values is None else round(values["std_eval"], 4),
            }
        )
    return rows


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    if not rows:
        return ["No data available yet."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return lines


def write_report(
    output_dir: str,
    baseline_rows: list[dict],
    efficiency_rows: list[dict],
    stability_rows: list[dict],
    ablation_rows: list[dict],
    sensitivity_rows: list[dict],
) -> None:
    lines = [
        "# Improvement Analysis",
        "",
        "This report summarizes the next-stage empirical analysis: baseline-vs-improved performance, sample efficiency, stability across seeds, and reward-shaping ablation readiness.",
        "",
        "## Baseline vs Improved",
        "",
    ]
    lines.extend(markdown_table(baseline_rows, list(baseline_rows[0].keys()) if baseline_rows else []))
    lines.extend(
        [
            "",
            "## Sample Efficiency",
            "",
            "The table reports the first evaluation episode where the mean evaluation reward reaches the selected threshold.",
            "",
        ]
    )
    lines.extend(markdown_table(efficiency_rows, list(efficiency_rows[0].keys()) if efficiency_rows else []))
    lines.extend(
        [
            "",
            "## Stability Across Seeds",
            "",
            "Lower standard deviation means the algorithm is less sensitive to random seed variation.",
            "",
        ]
    )
    lines.extend(markdown_table(stability_rows, list(stability_rows[0].keys()) if stability_rows else []))
    lines.extend(
        [
            "",
            "## MountainCar Reward-Shaping Ablation",
            "",
            "Rows with blank unshaped values indicate that the ablation experiments have not been run yet.",
            "",
        ]
    )
    lines.extend(markdown_table(ablation_rows, list(ablation_rows[0].keys()) if ablation_rows else []))
    lines.extend(
        [
            "",
            "## PER Alpha Sensitivity",
            "",
            "The table compares prioritization strength for Dueling DQN-PER on LunarLander.",
            "",
        ]
    )
    lines.extend(markdown_table(sensitivity_rows, list(sensitivity_rows[0].keys()) if sensitivity_rows else []))
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- A strong report should emphasize both final performance and stability rather than only the maximum reward.",
            "- MountainCar PPO remains a useful failure case because the improved PPO still struggles with sparse reward and long-horizon exploration.",
            "- LunarLander shows the clearest benefit from PER-based value learning, especially Dueling DQN with PER.",
            "- Reward-shaping ablation should be reported as a separate setting because shaped and unshaped MountainCar are not identical tasks.",
        ]
    )

    path = os.path.join(output_dir, "IMPROVEMENT_ANALYSIS.md")
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.results_dir, "comparisons")
    os.makedirs(output_dir, exist_ok=True)

    summary = load_summary(args.results_dir)
    metrics = load_metrics(args.results_dir)
    baseline_rows = build_baseline_vs_improved(summary)
    efficiency_rows = build_sample_efficiency(metrics)
    stability_rows = build_stability(summary)
    ablation_rows = build_reward_shaping_ablation(summary)
    sensitivity_rows = build_per_alpha_sensitivity(summary)

    write_csv(
        os.path.join(output_dir, "baseline_vs_improved.csv"),
        baseline_rows,
        list(baseline_rows[0].keys()) if baseline_rows else ["environment"],
    )
    write_csv(
        os.path.join(output_dir, "sample_efficiency.csv"),
        efficiency_rows,
        list(efficiency_rows[0].keys()) if efficiency_rows else ["environment"],
    )
    write_csv(
        os.path.join(output_dir, "stability_by_seed_std.csv"),
        stability_rows,
        list(stability_rows[0].keys()) if stability_rows else ["environment"],
    )
    write_csv(
        os.path.join(output_dir, "mountaincar_reward_shaping_ablation.csv"),
        ablation_rows,
        list(ablation_rows[0].keys()) if ablation_rows else ["algorithm_pair"],
    )
    write_csv(
        os.path.join(output_dir, "per_alpha_sensitivity.csv"),
        sensitivity_rows,
        list(sensitivity_rows[0].keys()) if sensitivity_rows else ["environment"],
    )
    write_report(output_dir, baseline_rows, efficiency_rows, stability_rows, ablation_rows, sensitivity_rows)
    print(f"improvement analysis saved to {output_dir}")


if __name__ == "__main__":
    main()
