# Project Structure Guide
# 项目结构说明

This document explains what each folder and major file records.
本文档说明每个文件夹和主要文件记录了什么内容。

## Root Directory
## 根目录

The root directory contains the shared code and main runner scripts.
根目录包含共享代码和主要运行脚本。

`README.md` gives the overall project overview, run commands, current results, and limitations.
`README.md` 提供项目总体介绍、运行命令、当前结果和局限性说明。

`experiment_runner.py` contains the shared implementation for DQN, DDQN, Dueling DQN, PPO, evaluation, CSV logging, model saving, and plotting.
`experiment_runner.py` 包含 DQN、DDQN、Dueling DQN、PPO、评估、CSV 记录、模型保存和绘图的共享实现。

`run_all_experiments.py` runs all official experiment entry scripts in `experiments/`.
`run_all_experiments.py` 会运行 `experiments/` 中所有正式实验入口脚本。

`build_comparison_report.py` reads result CSV files and generates cross-algorithm comparison plots.
`build_comparison_report.py` 会读取结果 CSV 文件并生成跨算法对比图。

`q_network.py` defines the standard Q-network used by DQN and DDQN.
`q_network.py` 定义 DQN 和 DDQN 使用的标准 Q 网络。

`replay_buffer.py` defines the replay buffer used by value-based methods.
`replay_buffer.py` 定义基于价值的方法使用的经验回放缓冲区。

`random_agent.py` contains a simple random agent baseline or testing utility.
`random_agent.py` 包含简单随机智能体基线或测试工具。

## `experiments/`
## `experiments/` 文件夹

This folder contains the official training entry points.
该文件夹包含正式训练入口。

Each file configures one environment-algorithm pair and then calls the shared runner.
每个文件配置一个环境与算法组合，然后调用共享运行器。

For example, `experiments/cartpole_train_dqn.py` runs DQN on `CartPole-v1`.
例如，`experiments/cartpole_train_dqn.py` 会在 `CartPole-v1` 上运行 DQN。

For example, `experiments/lunarlander_train_ppo.py` runs PPO on `LunarLander-v3`.
例如，`experiments/lunarlander_train_ppo.py` 会在 `LunarLander-v3` 上运行 PPO。

`experiments/_bootstrap.py` adds the project root to Python's import path.
`experiments/_bootstrap.py` 会把项目根目录加入 Python 的导入路径。

This allows experiment scripts to import `experiment_runner.py` after being moved into a subfolder.
这样实验脚本移动到子文件夹后仍然可以导入 `experiment_runner.py`。

## `docs/`
## `docs/` 文件夹

This folder contains project documentation.
该文件夹包含项目文档。

`docs/README_SIMPLE.md` gives a short summary of the project.
`docs/README_SIMPLE.md` 提供项目简要说明。

`docs/REPORT_OUTLINE.md` gives a report structure for the dissertation or project report.
`docs/REPORT_OUTLINE.md` 提供毕业论文或项目报告结构。

`docs/RESULTS_NOTES.md` records the current experimental results and reward shaping notes.
`docs/RESULTS_NOTES.md` 记录当前实验结果和奖励塑形说明。

`docs/KNOWLEDGE_POINTS.md` summarizes the theory and implementation knowledge covered by the project.
`docs/KNOWLEDGE_POINTS.md` 总结项目涉及的理论和实现知识点。

`docs/PROJECT_STRUCTURE.md` explains the organized folder structure.
`docs/PROJECT_STRUCTURE.md` 说明整理后的文件夹结构。

## `results/`
## `results/` 文件夹

This folder contains formal experiment outputs.
该文件夹包含正式实验输出。

`*_metrics.csv` files record per-episode training and evaluation metrics.
`*_metrics.csv` 文件记录逐轮训练和评估指标。

`*_summary.csv` files record final per-seed results plus mean and standard deviation.
`*_summary.csv` 文件记录每个随机种子的最终结果以及均值和标准差。

`*_training_curve.png` files show training reward curves.
`*_training_curve.png` 文件展示训练奖励曲线。

`*_evaluation_curve.png` files show evaluation reward curves.
`*_evaluation_curve.png` 文件展示评估奖励曲线。

`*_seed*.pth` files are saved model weights for each random seed.
`*_seed*.pth` 文件是每个随机种子的模型权重。

`results/comparisons/` contains merged comparison plots and the generated comparison report.
`results/comparisons/` 包含合并对比图和自动生成的对比报告。

## `artifacts/legacy_runs/`
## `artifacts/legacy_runs/` 文件夹

This folder stores older model files and older learning-curve images from earlier runs.
该文件夹保存早期实验产生的模型文件和学习曲线图片。

These files are useful for reference but are not the main formal results.
这些文件可作为参考，但不是主要正式结果。

## `legacy_scripts/`
## `legacy_scripts/` 文件夹

This folder stores older standalone training scripts.
该文件夹保存早期独立训练脚本。

These scripts are kept for reference and development history.
这些脚本用于参考和保留开发历史。

The official runnable experiment entry points are now in `experiments/`.
当前正式可运行的实验入口位于 `experiments/`。

## Recommended Commands
## 推荐命令

Run one experiment from the project root:
从项目根目录运行单个实验：

```powershell
G:\uni_course\comp3931\rl_project_env\Scripts\python.exe .\experiments\cartpole_train_dqn.py --seeds 0 1 2 --eval-every 20 --eval-episodes 5 --output-dir results
```

Run all formal experiments from the project root:
从项目根目录运行所有正式实验：

```powershell
G:\uni_course\comp3931\rl_project_env\Scripts\python.exe .\run_all_experiments.py --seeds 0 1 2 --eval-every 20 --eval-episodes 5 --output-dir results
```

Regenerate comparison figures:
重新生成对比图：

```powershell
G:\uni_course\comp3931\rl_project_env\Scripts\python.exe .\build_comparison_report.py
```
