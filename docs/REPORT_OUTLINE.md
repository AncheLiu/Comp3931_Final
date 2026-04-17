# Report Outline
# 报告提纲

## 1. Introduction
## 1. 引言

Introduce reinforcement learning and explain why empirical comparison matters.
介绍强化学习，并说明为什么实证比较是重要的。

State that the project compares several DRL algorithms on classic Gymnasium environments.
说明本项目比较了多个深度强化学习算法在经典 Gymnasium 环境中的表现。

Define the main goal as comparing performance, stability, and learning behavior.
将主要目标定义为比较性能、稳定性以及学习行为。

## 2. Objectives
## 2. 研究目标

Implement and run DQN, DDQN, Dueling DQN, and PPO.
实现并运行 DQN、DDQN、Dueling DQN 和 PPO。

Evaluate them on `CartPole-v1`, `MountainCar-v0`, `Acrobot-v1`, and `LunarLander-v3`.
在 `CartPole-v1`、`MountainCar-v0`、`Acrobot-v1` 和 `LunarLander-v3` 上评估这些算法。

Measure learning curves and final evaluation performance across multiple seeds.
衡量多随机种子条件下的学习曲线和最终评估表现。

Study the effect of reward shaping on MountainCar.
研究奖励塑形对 MountainCar 的影响。

## 3. Background
## 3. 背景知识

Briefly explain Markov decision processes, policy, value function, and reward.
简要解释马尔可夫决策过程、策略、价值函数和奖励。

Summarize each algorithm at a high level.
从高层角度概述每种算法。

Explain why DDQN reduces overestimation.
解释为什么 DDQN 能减少高估问题。

Explain why Dueling DQN separates value and advantage.
解释为什么 Dueling DQN 要分离状态价值和动作优势。

Explain why PPO is a policy-gradient method.
解释为什么 PPO 属于策略梯度方法。

## 4. Methodology
## 4. 方法

### 4.1 Environments
### 4.1 环境设置

Describe each environment briefly.
简要描述每个环境。

State action space size and observation dimension.
说明动作空间大小和观测维度。

Explain which environments are sparse-reward or harder to explore.
解释哪些环境具有稀疏奖励特征或更难探索。

### 4.2 Algorithms
### 4.2 算法

Describe DQN, DDQN, Dueling DQN, and PPO one by one.
分别介绍 DQN、DDQN、Dueling DQN 和 PPO。

For each one, include the core idea, network structure, and update rule at a high level.
对每种算法都应说明核心思想、网络结构以及高层次的更新规则。

### 4.3 Experimental Setup
### 4.3 实验设置

Record the number of seeds, number of episodes, evaluation frequency, and evaluation episodes.
记录随机种子数量、训练轮次、评估频率以及每次评估的轮数。

State the hardware and software setup.
说明硬件和软件环境配置。

Record shared hyperparameters.
记录共享超参数设置。

### 4.4 Reward Shaping
### 4.4 奖励塑形

Explain the MountainCar shaping rule.
解释 MountainCar 的奖励塑形规则。

Justify why it was introduced.
说明为什么要引入这一设计。

Clearly separate shaped and unshaped settings in the report.
在报告中清楚区分带塑形和不带塑形的实验设置。

## 5. Results
## 5. 实验结果

### 5.1 Learning Curves
### 5.1 学习曲线

Include training curves and evaluation curves.
展示训练曲线和评估曲线。

Compare convergence speed and stability.
比较收敛速度和稳定性。

### 5.2 Final Performance Table
### 5.2 最终表现表格

Use columns such as environment, algorithm, mean final evaluation reward, and standard deviation.
可使用环境、算法、最终平均评估回报和标准差等列来展示结果。

### 5.3 MountainCar Ablation
### 5.3 MountainCar 消融实验

Compare reward shaping versus no reward shaping.
比较使用奖励塑形和不使用奖励塑形的情况。

Discuss whether shaping improves learning speed or final performance.
讨论奖励塑形是否提升了学习速度或最终表现。

## 6. Discussion
## 6. 讨论

Discuss which algorithm worked best on each environment.
讨论每个环境中哪种算法表现最好。

Discuss which algorithm was most stable across seeds.
讨论哪种算法在不同随机种子下最稳定。

Discuss which environment was hardest and why.
讨论哪个环境最困难以及原因。

Discuss whether value-based methods or PPO behaved more reliably.
讨论基于价值的方法和 PPO 哪一类表现得更可靠。

## 7. Limitations
## 7. 局限性

State that PPO is simplified in this project.
说明本项目中的 PPO 是简化实现。

State that hyperparameter tuning is limited.
说明超参数调优仍然有限。

State that the number of seeds is finite.
说明随机种子数量仍然有限。

State that classic control tasks are smaller than real robotics tasks.
说明经典控制任务规模小于真实机器人任务。

## 8. Conclusion
## 8. 结论

Summarize the main findings.
总结主要发现。

State which methods were strongest overall.
说明整体上哪些方法表现更强。

Reflect on what the experiments show about algorithm-task matching.
总结实验对“算法与任务匹配关系”的启示。

## 9. Future Work
## 9. 未来工作

Possible future work includes more seeds, better tuning, prioritized replay, soft target updates, GAE for PPO, and larger control tasks.
未来工作可以包括更多随机种子、更好的调参、优先经验回放、软更新目标网络、为 PPO 加入 GAE，以及更大规模的控制任务。

## Appendix Ideas
## 附录建议

Possible appendix materials include full hyperparameter tables, run commands, extra plots, and sample CSV outputs.
附录可包含完整超参数表、运行命令、额外图像以及示例 CSV 输出。
