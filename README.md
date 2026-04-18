# Reinforcement Learning Empirical Investigation

# 强化学习实证研究项目

This project studies several deep reinforcement learning algorithms on classic Gymnasium control tasks.
本项目研究多种深度强化学习算法在经典 Gymnasium 控制任务上的表现。

The codebase is organized for empirical comparison rather than only single-run demos.
代码结构面向实证比较，而不仅仅是单次演示运行。

It supports multiple random seeds, periodic evaluation, CSV logging, comparison figures, and learning-curve plots.
它支持多随机种子实验、周期性评估、CSV 日志记录、对比图生成以及学习曲线绘制。

## Scope

## 项目范围

Environments used in this project are listed below.
本项目使用的环境如下。

- `CartPole-v1`
- `MountainCar-v0`
- `Acrobot-v1`
- `LunarLander-v3`

Algorithms used in this project are listed below.
本项目使用的算法如下。

- `DQN`
- `DDQN`
- `Dueling DQN`
- `PPO`

## Project Structure

## 项目结构

The core files are organized as follows.
核心文件结构如下。

```text
rl/
|- docs/
|- experiments/
|- results/
|- experiment_runner.py
|- run_all_experiments.py
|- build_comparison_report.py
|- q_network.py
|- replay_buffer.py
\- README.md
```

`docs/` contains project documentation, result notes, report outlines, and revision notes.
`docs/` 包含项目文档、结果记录、报告提纲和复习资料。

`experiments/` contains the official training entry scripts for each environment and algorithm.
`experiments/` 包含每个环境和算法对应的正式训练入口脚本。

`results/` contains CSV metrics, summaries, comparison figures, final evaluation records, and demonstration GIFs from formal runs.
`results/` 包含正式实验产生的 CSV 指标、汇总结果、对比图、最终评估记录和演示 GIF。

`experiment_runner.py` contains the shared training, evaluation, logging, and plotting logic.
`experiment_runner.py` 包含共享的训练、评估、日志记录和绘图逻辑。

`run_all_experiments.py` is the batch runner for all configured experiments.
`run_all_experiments.py` 是所有实验的批量运行脚本。

`build_comparison_report.py` creates combined comparison figures and a comparison report.
`build_comparison_report.py` 用于生成合并对比图和对比报告。

## Experimental Design

## 实验设计

The project is designed as an empirical investigation.
本项目被设计为一个实证研究。

Each experiment can run with multiple random seeds.
每个实验都可以用多个随机种子运行。

Each experiment can evaluate the current policy every fixed number of episodes.
每个实验都可以每隔固定轮次评估一次当前策略。

Each experiment can save per-episode metrics and final summaries to CSV files.
每个实验都可以将逐轮指标和最终汇总保存为 CSV 文件。

Default batch settings use seeds `0 1 2`.
默认批量实验设置使用 `0 1 2` 三个随机种子。

Default evaluation frequency is every `20` episodes.
默认评估频率是每 `20` 个 episode 评估一次。

Default evaluation runs `5` episodes each time.
默认每次评估运行 `5` 个 episode。

## MountainCar Reward Shaping

## MountainCar 奖励塑形

`MountainCar-v0` is difficult to learn from sparse rewards alone.
`MountainCar-v0` 仅依赖稀疏奖励时学习难度较大。

This project uses a reward shaping wrapper for MountainCar experiments.
本项目对 MountainCar 实验使用了奖励塑形包装器。

The agent receives extra reward when it reaches a new best position.
当小车到达新的历史最远位置时，智能体会获得额外奖励。

This provides denser learning signals during exploration.
这会在探索阶段提供更密集的学习信号。

Important report note: MountainCar results here are shaped-reward results.
报告中的重要说明：这里的 MountainCar 结果属于带奖励塑形的结果。

Shaped and unshaped MountainCar should be treated as different experimental settings.
带塑形和不带塑形的 MountainCar 应视为不同的实验设置。

## How To Run

## 运行方式

Use the project virtual-environment Python to run experiments.
请使用项目虚拟环境中的 Python 来运行实验。

Run a single experiment with default settings:
使用默认设置运行单个实验：

```bash
G:\uni_course\comp3931\rl_project_env\Scripts\python.exe experiments\cartpole_train_dqn.py
```

Run a single experiment with multiple seeds and periodic evaluation:
使用多个随机种子和周期性评估运行单个实验：

```bash
G:\uni_course\comp3931\rl_project_env\Scripts\python.exe experiments\cartpole_train_dqn.py --seeds 0 1 2 --eval-every 20 --eval-episodes 5 --output-dir results
```

Run the full experiment suite:
运行完整实验集合：

```bash
G:\uni_course\comp3931\rl_project_env\Scripts\python.exe run_all_experiments.py --seeds 0 1 2 --eval-every 20 --eval-episodes 5 --output-dir results
```

## Outputs

## 输出结果

Each experiment writes outputs into the selected results directory.
每个实验都会把输出写入指定的结果目录。

Typical repository outputs include metrics CSV files, summary CSV files, training curves, evaluation curves, comparison plots, and demonstration GIFs.
仓库中的典型输出包括指标 CSV、汇总 CSV、训练曲线、评估曲线、对比图以及演示 GIF。

Training scripts can save per-seed model checkpoints locally, but checkpoint files are excluded from the public repository to keep it lightweight and reproducible from source.
训练脚本可以在本地保存每个随机种子的模型检查点，但公开仓库中排除了检查点文件，以保持仓库轻量，并确保结果可以通过源码复现。

`metrics.csv` records environment, algorithm, seed, episode, training reward, evaluation reward, and epsilon when applicable.
`metrics.csv` 会记录环境、算法、随机种子、轮次、训练奖励、评估奖励，以及在适用时记录 epsilon。

`summary.csv` records final performance for each seed together with mean and standard deviation.
`summary.csv` 会记录每个随机种子的最终表现，以及均值和标准差。

## Recorded Results

## 已记录的实验结果

The following table summarizes the current results from the `results` directory using 3 random seeds.
下表汇总了当前 `results` 目录中的实验结果，使用了 3 个随机种子。

| Environment    | Algorithm   | Mean Final Evaluation Reward |    Std |
| -------------- | ----------- | ---------------------------: | -----: |
| CartPole-v1    | PPO         |                       496.93 |   4.34 |
| CartPole-v1    | Dueling DQN |                       333.27 | 118.07 |
| CartPole-v1    | DQN         |                       312.53 | 133.00 |
| CartPole-v1    | DDQN        |                       300.60 | 125.26 |
| Acrobot-v1     | PPO         |                       -94.60 |  12.55 |
| Acrobot-v1     | DDQN        |                      -350.13 | 183.85 |
| Acrobot-v1     | DQN         |                      -386.13 | 161.03 |
| Acrobot-v1     | Dueling DQN |                      -437.93 |  29.79 |
| LunarLander-v3 | DQN         |                        13.48 |  36.09 |
| LunarLander-v3 | DDQN        |                       -36.92 |  31.79 |
| LunarLander-v3 | Dueling DQN |                       -94.48 |  11.62 |
| LunarLander-v3 | PPO         |                      -246.15 | 168.27 |
| MountainCar-v0 | DQN         |                       -28.76 |  93.76 |
| MountainCar-v0 | DDQN        |                       -96.95 |  89.70 |
| MountainCar-v0 | Dueling DQN |                      -142.08 |  16.03 |
| MountainCar-v0 | PPO         |                      -155.37 |   0.00 |

The best algorithm on `CartPole-v1` is PPO.
`CartPole-v1` 上表现最好的算法是 PPO。

The best algorithm on `Acrobot-v1` is PPO.
`Acrobot-v1` 上表现最好的算法是 PPO。

The best algorithm on `LunarLander-v3` is DQN.
`LunarLander-v3` 上表现最好的算法是 DQN。

The best algorithm on `MountainCar-v0` is DQN under the shaped-reward setting.
`MountainCar-v0` 上表现最好的算法是在奖励塑形设置下的 DQN。

## Custom Reward Function

## 自定义奖励函数

The project uses a custom reward function for `MountainCar-v0`.
本项目在 `MountainCar-v0` 中使用了自定义奖励函数。

The main reason is that the original reward is sparse and learning is slow.
主要原因是原始奖励过于稀疏，导致学习速度较慢。

The shaping rule gives extra reward when the car reaches a new maximum position.
该塑形规则会在小车到达新的历史最远位置时给予额外奖励。

The code is shown below.
相关代码如下所示。

```python
class RewardShapingWrapper(gym.RewardWrapper):
    def __init__(self, env):
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
```

MountainCar results should be reported with their reward setting because this repository includes both shaped and unshaped ablation experiments.
报告 MountainCar 结果时应注明奖励设置，因为本仓库同时包含带奖励塑形和不带奖励塑形的消融实验。

## Current Limitations

## 当前局限性

PPO in this project is a simplified implementation rather than a production-grade benchmark.
本项目中的 PPO 是简化实现，而不是工业级基准实现。

Hyperparameters are mostly shared across environments and are not fully tuned for each task.
超参数大多在不同环境之间共享，并没有针对每个任务进行充分调优。

Some environments still show high variance across random seeds.
某些环境在不同随机种子之间仍然表现出较高方差。

## Recommended Next Steps

## 下一步建议

Run at least 3 to 5 seeds for each algorithm-environment pair.
建议每个算法与环境组合至少运行 3 到 5 个随机种子。

Compare final evaluation reward instead of only training reward.
比较时应优先使用最终评估奖励，而不只是训练奖励。

Use the MountainCar shaped-versus-unshaped ablation results to discuss how reward design changes learning behaviour.
使用 MountainCar 带塑形与不带塑形的消融结果，讨论奖励设计如何改变学习行为。
