# Knowledge Points For This Project
# 本项目涉及的知识点

This document summarizes the main knowledge areas covered by the project.
本文件总结了该项目涉及的主要知识领域。

It can be used for revision, report writing, and viva preparation.
它可以用于复习、写报告以及答辩准备。

## 1. Reinforcement Learning Fundamentals
## 1. 强化学习基础

An agent interacts with an environment step by step.
智能体以逐步交互的方式与环境进行互动。

At each step, the agent observes a state, chooses an action, receives a reward, and transitions to the next state.
在每一步中，智能体观察状态、选择动作、获得奖励，并转移到下一个状态。

The key concepts are state, action, reward, next state, and return.
关键概念包括状态、动作、奖励、下一状态以及回报。

The return is the cumulative future reward.
回报是未来奖励的累积值。

The discount factor `gamma` controls how much the agent values long-term reward.
折扣因子 `gamma` 决定了智能体对长期奖励的重视程度。

A policy defines how actions are chosen from states.
策略定义了如何根据状态选择动作。

`V(s)` represents state value, and `Q(s, a)` represents action value.
`V(s)` 表示状态价值，`Q(s, a)` 表示动作价值。

This project is based on the Markov decision process framework.
本项目建立在马尔可夫决策过程框架之上。

## 2. Deep Reinforcement Learning Concepts
## 2. 深度强化学习概念

This project uses neural networks to approximate value functions or policies.
本项目使用神经网络来逼近价值函数或策略。

This is necessary because the environments use continuous state spaces.
这是必要的，因为这些环境使用连续状态空间。

Exploration means trying actions to gather information.
探索意味着尝试动作以获取信息。

Exploitation means choosing actions that currently appear best.
利用意味着选择当前看起来最优的动作。

DQN-style methods use epsilon-greedy exploration.
DQN 类方法使用 epsilon-greedy 探索策略。

PPO uses stochastic action sampling from a policy distribution.
PPO 使用从策略分布中进行随机动作采样的方式。

DQN, DDQN, and Dueling DQN are off-policy methods.
DQN、DDQN 和 Dueling DQN 属于离策略方法。

PPO is an on-policy method.
PPO 属于在策略方法。

## 3. Deep Learning Knowledge
## 3. 深度学习知识

The project uses fully connected feed-forward neural networks.
本项目使用全连接前馈神经网络。

You need to understand linear layers, activation functions, and forward propagation.
你需要理解线性层、激活函数以及前向传播。

Value-based methods use mean squared error as the main loss.
基于价值的方法主要使用均方误差作为损失函数。

PPO uses a clipped surrogate objective together with value loss and entropy bonus.
PPO 使用裁剪替代目标，并结合价值损失和熵奖励项。

Parameter updates are done through backpropagation and Adam optimization.
参数更新通过反向传播和 Adam 优化器完成。

The implementation relies on PyTorch tensors, modules, and optimizers.
实现依赖于 PyTorch 的张量、模块和优化器。

## 4. Value-Based Algorithms
## 4. 基于价值的算法

### 4.1 DQN
### 4.1 DQN

DQN uses a neural network to approximate `Q(s, a)`.
DQN 使用神经网络来逼近 `Q(s, a)`。

The action with the highest Q-value is selected during exploitation.
在利用阶段会选择 Q 值最高的动作。

Important components include replay buffer, target network, and epsilon-greedy exploration.
重要组成部分包括经验回放、目标网络以及 epsilon-greedy 探索。

The Bellman target is based on future maximum Q-value estimates.
Bellman 目标基于未来最大 Q 值估计。

### 4.2 DDQN
### 4.2 DDQN

DDQN is designed to reduce overestimation bias in standard DQN.
DDQN 的目标是减少标准 DQN 中的高估偏差。

It uses the online network for action selection and the target network for action evaluation.
它使用在线网络进行动作选择，并使用目标网络进行动作评估。

### 4.3 Dueling DQN
### 4.3 Dueling DQN

Dueling DQN decomposes Q-value into state value and action advantage.
Dueling DQN 将 Q 值分解为状态价值和动作优势。

This architecture helps distinguish whether a state is good independently of a specific action.
这种结构有助于区分一个状态本身是否优良，而不完全依赖某个具体动作。

## 5. Policy Gradient Knowledge
## 5. 策略梯度知识

### 5.1 PPO
### 5.1 PPO

PPO directly optimizes the policy rather than only the value function.
PPO 直接优化策略，而不仅仅是优化价值函数。

Its main idea is to prevent policy updates from changing too aggressively.
它的核心思想是防止策略更新变化过于剧烈。

Important concepts include actor-critic structure, probability ratio, clipping, and entropy regularization.
重要概念包括 actor-critic 结构、概率比、裁剪机制以及熵正则化。

### 5.2 Actor-Critic
### 5.2 Actor-Critic

The actor outputs action probabilities.
Actor 输出动作概率分布。

The critic estimates the value of the current state.
Critic 估计当前状态的价值。

The critic helps reduce variance in policy-gradient updates.
Critic 有助于降低策略梯度更新中的方差。

## 6. Environment Knowledge
## 6. 环境知识

`CartPole-v1` is an easier balancing problem and is useful for debugging algorithms.
`CartPole-v1` 是一个较容易的平衡控制问题，适合用于调试算法。

`MountainCar-v0` highlights sparse rewards and difficult exploration.
`MountainCar-v0` 突出了稀疏奖励和困难探索问题。

`Acrobot-v1` involves underactuated control and long-horizon credit assignment.
`Acrobot-v1` 涉及欠驱动控制以及长时程信用分配问题。

`LunarLander-v3` has more complex dynamics and a larger state space.
`LunarLander-v3` 具有更复杂的动力学和更大的状态空间。

## 7. Reward Shaping
## 7. 奖励塑形

Reward shaping is one of the most important custom design choices in this project.
奖励塑形是本项目中最重要的自定义设计之一。

It is used in `MountainCar-v0`.
它被用于 `MountainCar-v0`。

The goal is to turn sparse reward into denser learning feedback.
它的目标是把稀疏奖励转化为更密集的学习反馈。

Reward shaping may improve learning speed, but it also changes the experimental setting.
奖励塑形可能提升学习速度，但它也改变了实验设置。

Therefore shaped and unshaped results must be reported carefully.
因此带塑形和不带塑形的结果必须谨慎区分并单独说明。

## 8. Experiment Design Knowledge
## 8. 实验设计知识

This project includes multiple random seeds, evaluation protocols, mean values, and standard deviations.
本项目包含多随机种子、评估流程、均值计算以及标准差分析。

Random seeds matter because different seeds can produce different training trajectories.
随机种子之所以重要，是因为不同种子会带来不同的训练轨迹。

Evaluation reward is more suitable than training reward for final comparison.
对于最终比较而言，评估奖励比训练奖励更合适。

Convergence speed and stability are both important metrics.
收敛速度和稳定性都是重要指标。

## 9. Result Logging And Visualization
## 9. 结果记录与可视化

The project saves metrics CSV files, summary CSV files, model checkpoints, and plots.
本项目会保存指标 CSV、汇总 CSV、模型检查点以及图像结果。

Matplotlib is used to generate training curves and evaluation curves.
Matplotlib 被用来生成训练曲线和评估曲线。

Comparison plots are generated to make cross-algorithm analysis easier.
对比图用于让跨算法分析更加方便。

## 10. Software Engineering Knowledge
## 10. 软件工程知识

The project uses modular experiment runner design.
本项目使用了模块化实验运行器设计。

It separates algorithm logic from experiment configuration.
它将算法逻辑与实验配置分离开来。

It uses reusable configuration objects and structured output directories.
它使用可复用的配置对象和结构化的输出目录。

Batch execution and comparison reporting are also included.
项目中还包含批量执行和对比报告生成功能。

## 11. Mathematics Used
## 11. 所涉及的数学知识

The project uses probability, linear algebra, and optimization.
本项目涉及概率论、线性代数和优化方法。

Probability is needed for stochastic policies and expected return.
概率论用于理解随机策略和期望回报。

Linear algebra is needed for vectors, matrices, and neural-network computation.
线性代数用于理解向量、矩阵以及神经网络计算。

Optimization is needed for gradient-based learning and parameter updates.
优化方法用于理解基于梯度的学习和参数更新。

## 12. Report And Viva Knowledge
## 12. 报告与答辩知识点

You should be able to explain why multiple algorithms are compared.
你需要能够解释为什么要比较多个算法。

You should be able to explain why multiple environments are needed.
你需要能够解释为什么要使用多个环境。

You should be able to explain why PPO can be better on some tasks and DQN can still be better on others.
你需要能够解释为什么 PPO 在某些任务上更好，而 DQN 在另一些任务上依然可能更优。

You should be able to explain why standard deviation matters.
你需要能够解释为什么标准差很重要。

You should be able to explain why reward shaping must be disclosed clearly.
你需要能够解释为什么奖励塑形必须明确披露。

## 13. Short Summary
## 13. 简短总结

This project combines reinforcement learning, deep learning, experiment design, visualization, and engineering practice.
本项目综合了强化学习、深度学习、实验设计、结果可视化以及工程实践。

## 14. Suggested Revision Order
## 14. 建议复习顺序

Revise RL basics first.
先复习强化学习基础。

Then revise DQN, DDQN, Dueling DQN, and PPO.
然后复习 DQN、DDQN、Dueling DQN 和 PPO。

Then revise reward shaping, multi-seed evaluation, and result interpretation.
最后复习奖励塑形、多随机种子评估以及结果解释。
