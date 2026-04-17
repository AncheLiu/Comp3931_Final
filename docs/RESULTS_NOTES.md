# Results Notes
# 结果记录说明

This file records the current experimental outcomes and the custom reward design used in the project.
本文件记录当前实验结果以及项目中使用的自定义奖励设计。

## Current Results Summary
## 当前结果摘要

The experiments were run with 3 random seeds and periodic evaluation.
这些实验使用了 3 个随机种子并进行了周期性评估。

| Environment | Best Algorithm | Mean Final Evaluation Reward | Std |
| --- | --- | ---: | ---: |
| CartPole-v1 | PPO | 496.93 | 4.34 |
| Acrobot-v1 | PPO | -94.60 | 12.55 |
| LunarLander-v3 | DQN | 13.48 | 36.09 |
| MountainCar-v0 | DQN | -28.76 | 93.76 |

PPO is the strongest method on CartPole and Acrobot in the current runs.
在当前实验中，PPO 在 CartPole 和 Acrobot 上表现最好。

DQN is the strongest method on LunarLander and MountainCar in the current runs.
在当前实验中，DQN 在 LunarLander 和 MountainCar 上表现最好。

## Full Comparison Table
## 完整对比表

The following table records the full comparison results.
下表记录了完整的对比结果。

| Environment | Algorithm | Mean Final Evaluation Reward | Std |
| --- | --- | ---: | ---: |
| CartPole-v1 | PPO | 496.93 | 4.34 |
| CartPole-v1 | Dueling DQN | 333.27 | 118.07 |
| CartPole-v1 | DQN | 312.53 | 133.00 |
| CartPole-v1 | DDQN | 300.60 | 125.26 |
| Acrobot-v1 | PPO | -94.60 | 12.55 |
| Acrobot-v1 | DDQN | -350.13 | 183.85 |
| Acrobot-v1 | DQN | -386.13 | 161.03 |
| Acrobot-v1 | Dueling DQN | -437.93 | 29.79 |
| LunarLander-v3 | DQN | 13.48 | 36.09 |
| LunarLander-v3 | DDQN | -36.92 | 31.79 |
| LunarLander-v3 | Dueling DQN | -94.48 | 11.62 |
| LunarLander-v3 | PPO | -246.15 | 168.27 |
| MountainCar-v0 | DQN | -28.76 | 93.76 |
| MountainCar-v0 | DDQN | -96.95 | 89.70 |
| MountainCar-v0 | Dueling DQN | -142.08 | 16.03 |
| MountainCar-v0 | PPO | -155.37 | 0.00 |

## Custom Reward Function Used In MountainCar
## MountainCar 中使用的自定义奖励函数

The `MountainCar-v0` experiments use a custom reward shaping wrapper.
`MountainCar-v0` 实验使用了自定义奖励塑形包装器。

The purpose is to reduce the sparse-reward problem and provide intermediate progress feedback.
其目的是减轻稀疏奖励问题，并为学习过程提供中间进展反馈。

The rule is simple: if the car reaches a new maximum position, extra reward is added.
规则很简单：如果小车达到新的历史最远位置，就增加额外奖励。

The code is shown below.
代码如下所示。

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

MountainCar results should be explicitly labeled as shaped-reward results.
MountainCar 的结果应明确标注为带奖励塑形的结果。

If a stricter empirical comparison is needed, an additional no-shaping baseline should be added.
如果需要更严格的实证比较，应补充一个不使用奖励塑形的基线实验。
