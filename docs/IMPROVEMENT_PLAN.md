# Improvement Plan for Higher Marking Bands
# 面向高分档的项目改进计划

This document turns the project from a basic algorithm comparison into a stronger empirical investigation.
本文档将项目从基础算法比较升级为更完整的实证研究。

The marking guidance rewards systematic background research, justified methodology, substantial implementation, thorough validation, quantified evaluation, clear presentation, and critical self-appraisal.
评分标准重视系统性背景研究、有依据的方法设计、有复杂度的实现、充分验证、量化评估、清晰展示以及批判性自我反思。

Therefore, the next stage should not only chase higher reward scores.
因此，下一阶段不应该只追求更高的奖励分数。

It should show why each improvement was selected, how it changes the algorithm, and whether the evidence supports the design choice.
它应该展示每个改进为什么被选择、它如何改变算法，以及实验数据是否支持这个设计选择。

## Research Focus
## 研究重点

The revised project aim is to compare baseline deep reinforcement learning algorithms with targeted improvement mechanisms on classic control tasks.
修订后的项目目标是在经典控制任务上比较基础深度强化学习算法与有针对性的改进机制。

The central question is whether common stability and sample-efficiency improvements lead to measurable gains across different environments.
核心问题是常见的稳定性与样本效率改进是否能在不同环境中带来可量化收益。

The strongest report narrative is `baseline vs improved`, supported by ablation-style comparisons.
最有说服力的报告叙事是 `baseline vs improved`，并用类似消融实验的对比支撑。

## Implemented Improvements
## 已实现的改进

Prioritized Experience Replay has been added for DQN, DDQN, and Dueling DQN variants.
已经为 DQN、DDQN 和 Dueling DQN 变体加入 Prioritized Experience Replay。

PER samples transitions with larger TD error more often, which can improve sample efficiency.
PER 会更频繁地采样 TD 误差较大的 transition，从而可能提升样本效率。

Huber loss and gradient clipping have been added to improve value-learning stability.
已经加入 Huber loss 和梯度裁剪，用于提升 value-learning 的稳定性。

Improved PPO now supports Generalized Advantage Estimation, advantage normalization, minibatch updates, and gradient clipping.
改进版 PPO 现在支持 Generalized Advantage Estimation、advantage normalization、minibatch update 和梯度裁剪。

PPO evaluation still records both greedy and stochastic policy performance.
PPO 评估仍然同时记录 greedy 和 stochastic 两种策略表现。

## New Experiment Scripts
## 新增实验脚本

The improved value-based experiments are focused on MountainCar and LunarLander because these tasks showed weaker or unstable baseline performance.
改进版 value-based 实验重点放在 MountainCar 和 LunarLander，因为这些任务中的 baseline 表现较弱或不稳定。

The new value-based scripts are `mountaincar_train_dqn_per.py`, `mountaincar_train_ddqn_per.py`, `mountaincar_train_duelingdqn_per.py`, `lunarlander_train_dqn_per.py`, `lunarlander_train_ddqn_per.py`, and `lunarlander_train_duelingdqn_per.py`.
新增的 value-based 脚本包括 `mountaincar_train_dqn_per.py`、`mountaincar_train_ddqn_per.py`、`mountaincar_train_duelingdqn_per.py`、`lunarlander_train_dqn_per.py`、`lunarlander_train_ddqn_per.py` 和 `lunarlander_train_duelingdqn_per.py`。

The new PPO scripts are `mountaincar_train_ppo_improved.py` and `lunarlander_train_ppo_improved.py`.
新增的 PPO 脚本是 `mountaincar_train_ppo_improved.py` 和 `lunarlander_train_ppo_improved.py`。

The batch runner now supports `--suite baseline`, `--suite improved`, and `--suite all`.
批量运行脚本现在支持 `--suite baseline`、`--suite improved` 和 `--suite all`。

## Recommended Run Commands
## 推荐运行命令

Run only the improved experiments first to avoid overwriting the original baseline results.
建议先只运行改进实验，避免覆盖原始 baseline 结果。

```powershell
G:\uni_course\comp3931\rl_project_env\Scripts\python.exe .\run_all_experiments.py --suite improved --seeds 0 1 2 --eval-every 20 --eval-episodes 5 --output-dir results_improved
```

Regenerate the comparison report after improved experiments finish.
改进实验完成后重新生成对比报告。

```powershell
G:\uni_course\comp3931\rl_project_env\Scripts\python.exe .\build_comparison_report.py --results-dir results_improved
```

If the report should compare old and new results together, copy selected improved CSV files into `results` or run the improved suite with `--output-dir results`.
如果报告需要把旧结果和新结果放在一起比较，可以把选定的 improved CSV 复制到 `results`，或者直接用 `--output-dir results` 运行 improved suite。

## Marking Alignment
## 评分标准对应关系

For background research, the report should explain DQN, DDQN, Dueling DQN, PPO, PER, GAE, and reward shaping with academic references.
对于背景研究，报告应结合学术引用解释 DQN、DDQN、Dueling DQN、PPO、PER、GAE 和 reward shaping。

For methodology, the report should justify why PER is used for off-policy value methods and why GAE/minibatch PPO is used for policy-gradient stability.
对于方法设计，报告应说明为什么 PER 适合 off-policy value 方法，以及为什么 GAE/minibatch PPO 有助于 policy-gradient 稳定性。

For implementation and validation, the report should describe the reusable runner, multi-seed setup, model checkpoints, CSV logging, and syntax or smoke tests.
对于实现与验证，报告应描述可复用 runner、多随机种子设置、模型检查点、CSV 记录以及语法或 smoke tests。

For results and discussion, the report should compare final evaluation reward, mean and standard deviation, learning curves, sample efficiency, and failure cases.
对于结果与讨论，报告应比较最终评估奖励、均值和标准差、学习曲线、样本效率以及失败案例。

For presentation, the report should use concise tables, mean-plus-standard-deviation plots, and clearly labeled baseline/improved figures.
对于展示质量，报告应使用简洁表格、均值加标准差图，以及清楚标注 baseline/improved 的图像。

For self-appraisal, the report should discuss limitations such as compute budget, sensitivity to hyperparameters, environment-specific reward shaping, and reproducibility risks.
对于自我评价，报告应讨论计算资源限制、超参数敏感性、特定环境奖励塑形以及可复现性风险等局限。

## High-Impact Next Steps
## 高影响力下一步

Run the improved suite with three seeds and save it in `results_improved`.
使用三个随机种子运行 improved suite，并保存到 `results_improved`。

Compare baseline and improved results in a table that reports mean, standard deviation, and percentage or absolute improvement.
用表格比较 baseline 与 improved 结果，并报告均值、标准差以及百分比或绝对提升。

Add one MountainCar reward-shaping ablation if time allows, comparing shaped and unshaped settings.
如果时间允许，加入一个 MountainCar reward shaping 消融实验，对比 shaped 与 unshaped 设置。

Add a short failure analysis for PPO on MountainCar and LunarLander if improved PPO still underperforms.
如果改进版 PPO 在 MountainCar 或 LunarLander 上仍表现较弱，应加入简短失败分析。

## Added Ablation Experiments
## 已加入的消融实验

The project now includes unshaped MountainCar scripts for DQN, DDQN, DQN-PER, and DDQN-PER.
项目现在已经加入 MountainCar 的 unshaped 对照脚本，覆盖 DQN、DDQN、DQN-PER 和 DDQN-PER。

These scripts are designed to test whether reward shaping improves learning compared with the original sparse reward setting.
这些脚本用于测试 reward shaping 相比原始 sparse reward 设置是否确实改善学习。

Run the ablation suite with:
使用以下命令运行消融实验套件：

```powershell
G:\uni_course\comp3931\rl_project_env\Scripts\python.exe .\run_all_experiments.py --suite ablation --seeds 0 1 2 --eval-every 50 --eval-episodes 3 --output-dir results
```

Generate the improvement analysis report with:
使用以下命令生成改进分析报告：

```powershell
G:\uni_course\comp3931\rl_project_env\Scripts\python.exe .\build_improvement_analysis.py --results-dir results
```

## Final Experimental Additions
## 最终实验补充

The project also includes three final experimental additions: final evaluation, agent demonstrations, and PER alpha sensitivity.
项目还加入了三个最终实验补充：最终评估、agent 演示以及 PER alpha 敏感性分析。

Final evaluation uses more evaluation episodes for the most important trained models.
最终评估会对最重要的已训练模型使用更多评估回合。

```powershell
G:\uni_course\comp3931\rl_project_env\Scripts\python.exe .\final_evaluate_models.py --results-dir results --eval-episodes 20 --seeds 0 1 2 --device cpu
```

Agent demonstrations save GIFs for selected baseline, improved, and failure-case agents.
Agent 演示会为选定的 baseline、improved 和 failure-case 智能体保存 GIF。

```powershell
G:\uni_course\comp3931\rl_project_env\Scripts\python.exe .\record_agent_demos.py --results-dir results --output-dir results\demos --device cpu
```

PER alpha sensitivity compares `alpha=0.4`, `alpha=0.6`, and `alpha=0.8` for Dueling DQN-PER on LunarLander.
PER alpha 敏感性分析会比较 LunarLander 上 Dueling DQN-PER 的 `alpha=0.4`、`alpha=0.6` 和 `alpha=0.8`。

```powershell
G:\uni_course\comp3931\rl_project_env\Scripts\python.exe .\run_all_experiments.py --suite sensitivity --seeds 0 1 2 --eval-every 50 --eval-episodes 3 --output-dir results
```
