import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


RESULTS_DIR = "results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "comparisons")
ENVIRONMENT_STEMS = {
    "CartPole-v1": "cartpolev1",
    "MountainCar-v0": "mountaincarv0",
    "Acrobot-v1": "acrobotv1",
    "LunarLander-v3": "lunarlanderv3",
}


def normalize_environment(environment: str) -> str:
    return ENVIRONMENT_STEMS.get(environment, environment.lower().replace("-", "").replace("_", ""))


def parse_result_stem(stem: str) -> tuple[str, str]:
    for environment in sorted(ENVIRONMENT_STEMS.values(), key=len, reverse=True):
        suffix = f"_{environment}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], environment
    parts = stem.split("_")
    return parts[0], "_".join(parts[1:])


def load_metrics():
    datasets = []
    for filename in sorted(os.listdir(RESULTS_DIR)):
        if not filename.endswith("_metrics.csv"):
            continue

        path = os.path.join(RESULTS_DIR, filename)
        with open(path, encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

        stem = filename.replace("_metrics.csv", "")
        parsed_algorithm, parsed_environment = parse_result_stem(stem)
        if rows:
            algorithm = rows[0].get("algorithm") or parsed_algorithm
            environment = normalize_environment(rows[0].get("environment") or parsed_environment)
        else:
            algorithm = parsed_algorithm
            environment = parsed_environment

        datasets.append(
            {
                "environment": environment,
                "algorithm": algorithm,
                "rows": rows,
            }
        )
    return datasets


def load_summary():
    summary = defaultdict(dict)
    for filename in sorted(os.listdir(RESULTS_DIR)):
        if not filename.endswith("_summary.csv"):
            continue

        path = os.path.join(RESULTS_DIR, filename)
        with open(path, encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

        stem = filename.replace("_summary.csv", "")
        algorithm, environment = parse_result_stem(stem)

        mean_row = next(row for row in rows if row["seed"] == "mean")
        std_row = next(row for row in rows if row["seed"] == "std")
        summary[environment][algorithm] = {
            "mean_eval": float(mean_row["final_eval_reward"]),
            "std_eval": float(std_row["final_eval_reward"]),
        }
    return summary


def mean_by_episode(rows, value_key, keep_empty=False):
    grouped = defaultdict(list)
    for row in rows:
        value = row[value_key]
        if value == "":
            if keep_empty:
                continue
            continue
        grouped[int(row["episode"])].append(float(value))

    episodes = sorted(grouped)
    means = [sum(grouped[episode]) / len(grouped[episode]) for episode in episodes]
    stds = []
    for episode in episodes:
        values = grouped[episode]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stds.append(variance ** 0.5)
    return episodes, means, stds


def plot_environment_comparison(environment, env_datasets):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for dataset in sorted(env_datasets, key=lambda item: item["algorithm"]):
        train_episodes, train_means, train_stds = mean_by_episode(dataset["rows"], "train_reward")
        eval_episodes, eval_means, eval_stds = mean_by_episode(dataset["rows"], "eval_reward")
        label = dataset["algorithm"]

        axes[0].plot(train_episodes, train_means, label=label)
        axes[0].fill_between(
            train_episodes,
            [mean - std for mean, std in zip(train_means, train_stds)],
            [mean + std for mean, std in zip(train_means, train_stds)],
            alpha=0.12,
        )
        axes[1].plot(eval_episodes, eval_means, label=label)
        axes[1].fill_between(
            eval_episodes,
            [mean - std for mean, std in zip(eval_means, eval_stds)],
            [mean + std for mean, std in zip(eval_means, eval_stds)],
            alpha=0.12,
        )

    axes[0].set_title(f"{environment} Training Comparison")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Mean Train Reward")

    axes[1].set_title(f"{environment} Evaluation Comparison")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Mean Eval Reward")

    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"{environment}_comparison.png")
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def detect_extension_candidates(datasets):
    notes = []
    for dataset in datasets:
        eval_rows = [row for row in dataset["rows"] if row["eval_reward"] != ""]
        if len(eval_rows) < 5:
            continue

        last_five = [float(row["eval_reward"]) for row in eval_rows[-5:]]
        delta_total = last_five[-1] - last_five[0]
        delta_last = last_five[-1] - last_five[-2]

        if delta_total > 30 and delta_last >= 0:
            verdict = "likely_benefit"
        elif delta_total > 0 and delta_last > 0:
            verdict = "possible_benefit"
        elif abs(delta_total) < 5 and abs(delta_last) < 5:
            verdict = "mostly_plateaued"
        else:
            verdict = "unclear_or_unstable"

        notes.append(
            {
                "environment": dataset["environment"],
                "algorithm": dataset["algorithm"],
                "last_five": last_five,
                "delta_total": round(delta_total, 3),
                "delta_last": round(delta_last, 3),
                "verdict": verdict,
            }
        )
    return notes


def write_report(summary, comparison_paths, extension_notes):
    lines = [
        "# Comparison Report",
        "",
        "This file groups the main comparison plots and notes whether some experiments may still improve with more episodes.",
        "",
        "## Combined Comparison Plots",
        "",
    ]

    for environment in sorted(comparison_paths):
        image_path = os.path.basename(comparison_paths[environment])
        lines.append(f"### {environment}")
        lines.append("")
        lines.append(f"![{environment} comparison]({image_path})")
        lines.append("")
        lines.append("| Algorithm | Mean Final Eval Reward | Std |")
        lines.append("| --- | ---: | ---: |")
        for algorithm, values in sorted(
            summary[environment].items(), key=lambda item: item[1]["mean_eval"], reverse=True
        ):
            lines.append(
                f"| {algorithm} | {values['mean_eval']:.2f} | {values['std_eval']:.2f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Episode Extension Notes",
            "",
            "| Environment | Algorithm | Last 5 Eval Points | Delta (first to last) | Delta (last step) | Verdict |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )

    for note in sorted(extension_notes, key=lambda item: (item["environment"], item["algorithm"])):
        lines.append(
            f"| {note['environment']} | {note['algorithm']} | {note['last_five']} | "
            f"{note['delta_total']:.3f} | {note['delta_last']:.3f} | {note['verdict']} |"
        )

    lines.extend(
        [
            "",
            "## Quick Interpretation",
            "",
            "- `likely_benefit`: the evaluation curve is still rising at the end, so more episodes may help.",
            "- `possible_benefit`: there is some late-stage improvement, but evidence is weaker.",
            "- `mostly_plateaued`: the last evaluations are nearly flat, so extra episodes are less likely to matter much.",
            "- `unclear_or_unstable`: the ending curve is noisy or dropping, so simply increasing episodes may not be enough; tuning may be more useful.",
        ]
    )

    report_path = os.path.join(OUTPUT_DIR, "COMPARISON_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    global RESULTS_DIR, OUTPUT_DIR
    args = parse_args()
    RESULTS_DIR = args.results_dir
    OUTPUT_DIR = args.output_dir or os.path.join(RESULTS_DIR, "comparisons")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datasets = load_metrics()
    summary = load_summary()

    by_environment = defaultdict(list)
    for dataset in datasets:
        by_environment[dataset["environment"]].append(dataset)

    comparison_paths = {}
    for environment, env_datasets in by_environment.items():
        comparison_paths[environment] = plot_environment_comparison(environment, env_datasets)

    extension_notes = detect_extension_candidates(datasets)
    write_report(summary, comparison_paths, extension_notes)
    print(f"comparison report saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
