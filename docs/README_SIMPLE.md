# Simple Project Summary
# 项目简要说明

This project implements several deep reinforcement learning algorithms on classic Gymnasium tasks.
本项目在经典 Gymnasium 任务上实现了多种深度强化学习算法。

The main algorithms are DQN, DDQN, Dueling DQN, and PPO.
主要算法包括 DQN、DDQN、Dueling DQN 和 PPO。

The main environments are CartPole, MountainCar, Acrobot, and LunarLander.
主要环境包括 CartPole、MountainCar、Acrobot 和 LunarLander。

The project supports multi-seed experiments, evaluation, CSV logging, and result plotting.
本项目支持多随机种子实验、策略评估、CSV 日志记录和结果绘图。

MountainCar uses a custom reward shaping function.
MountainCar 使用了自定义奖励塑形函数。

The batch runner is `run_all_experiments.py`.
批量运行脚本是 `run_all_experiments.py`。

The main shared training logic is in `experiment_runner.py`.
主要共享训练逻辑位于 `experiment_runner.py`。
