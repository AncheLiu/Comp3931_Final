# Improvement Analysis

This report summarizes the next-stage empirical analysis: baseline-vs-improved performance, sample efficiency, stability across seeds, and reward-shaping ablation readiness.

## Baseline vs Improved

| environment | best_baseline | baseline_mean_eval | baseline_std_eval | best_improved | improved_mean_eval | improved_std_eval | absolute_gain | relative_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lunarlanderv3 | dqn | 13.4802 | 36.088 | dueling_dqn_per | 224.2789 | 22.4283 | 210.7987 | 15.6377 |
| mountaincarv0 | dqn | -28.7648 | 93.7558 | ddqn_per | 62.1204 | 35.3157 | 90.8852 | 3.1596 |

## Sample Efficiency

The table reports the first evaluation episode where the mean evaluation reward reaches the selected threshold.

| environment | algorithm | threshold | first_episode |
| --- | --- | --- | --- |
| acrobotv1 | ddqn | -100.0 | not reached |
| acrobotv1 | dqn | -100.0 | not reached |
| acrobotv1 | dueling_dqn | -100.0 | not reached |
| acrobotv1 | ppo | -100.0 | 1540 |
| cartpolev1 | ddqn | 475.0 | not reached |
| cartpolev1 | dqn | 475.0 | not reached |
| cartpolev1 | dueling_dqn | 475.0 | not reached |
| cartpolev1 | ppo | 475.0 | 340 |
| lunarlanderv3 | ddqn | 100.0 | not reached |
| lunarlanderv3 | ddqn | 200.0 | not reached |
| lunarlanderv3 | ddqn_per | 100.0 | 550 |
| lunarlanderv3 | ddqn_per | 200.0 | 950 |
| lunarlanderv3 | dqn | 100.0 | not reached |
| lunarlanderv3 | dqn | 200.0 | not reached |
| lunarlanderv3 | dqn_per | 100.0 | 550 |
| lunarlanderv3 | dqn_per | 200.0 | 750 |
| lunarlanderv3 | dueling_dqn | 100.0 | not reached |
| lunarlanderv3 | dueling_dqn | 200.0 | not reached |
| lunarlanderv3 | dueling_dqn_per | 100.0 | 550 |
| lunarlanderv3 | dueling_dqn_per | 200.0 | 1000 |
| lunarlanderv3 | dueling_dqn_per_alpha04 | 100.0 | 550 |
| lunarlanderv3 | dueling_dqn_per_alpha04 | 200.0 | not reached |
| lunarlanderv3 | dueling_dqn_per_alpha08 | 100.0 | 850 |
| lunarlanderv3 | dueling_dqn_per_alpha08 | 200.0 | not reached |
| lunarlanderv3 | ppo | 100.0 | not reached |
| lunarlanderv3 | ppo | 200.0 | not reached |
| lunarlanderv3 | ppo_improved | 100.0 | 2500 |
| lunarlanderv3 | ppo_improved | 200.0 | not reached |
| mountaincarv0 | ddqn | 0.0 | not reached |
| mountaincarv0 | ddqn | 50.0 | not reached |
| mountaincarv0 | ddqn_per | 0.0 | 350 |
| mountaincarv0 | ddqn_per | 50.0 | 500 |
| mountaincarv0 | ddqn_per_unshaped | 0.0 | not reached |
| mountaincarv0 | ddqn_per_unshaped | 50.0 | not reached |
| mountaincarv0 | ddqn_unshaped | 0.0 | not reached |
| mountaincarv0 | ddqn_unshaped | 50.0 | not reached |
| mountaincarv0 | dqn | 0.0 | not reached |
| mountaincarv0 | dqn | 50.0 | not reached |
| mountaincarv0 | dqn_per | 0.0 | 450 |
| mountaincarv0 | dqn_per | 50.0 | 500 |
| mountaincarv0 | dqn_per_unshaped | 0.0 | not reached |
| mountaincarv0 | dqn_per_unshaped | 50.0 | not reached |
| mountaincarv0 | dqn_unshaped | 0.0 | not reached |
| mountaincarv0 | dqn_unshaped | 50.0 | not reached |
| mountaincarv0 | dueling_dqn | 0.0 | not reached |
| mountaincarv0 | dueling_dqn | 50.0 | not reached |
| mountaincarv0 | dueling_dqn_per | 0.0 | 500 |
| mountaincarv0 | dueling_dqn_per | 50.0 | not reached |
| mountaincarv0 | ppo | 0.0 | not reached |
| mountaincarv0 | ppo | 50.0 | not reached |
| mountaincarv0 | ppo_improved | 0.0 | not reached |
| mountaincarv0 | ppo_improved | 50.0 | not reached |

## Stability Across Seeds

Lower standard deviation means the algorithm is less sensitive to random seed variation.

| environment | algorithm | mean_eval | std_eval |
| --- | --- | --- | --- |
| acrobotv1 | ppo | -94.6 | 12.5518 |
| acrobotv1 | dueling_dqn | -437.9333 | 29.7878 |
| acrobotv1 | dqn | -386.1333 | 161.0318 |
| acrobotv1 | ddqn | -350.1333 | 183.8496 |
| cartpolev1 | ppo | 496.9333 | 4.3369 |
| cartpolev1 | dueling_dqn | 333.2667 | 118.0658 |
| cartpolev1 | ddqn | 300.6 | 125.262 |
| cartpolev1 | dqn | 312.5333 | 132.9964 |
| lunarlanderv3 | ppo | -7.537 | 8.8781 |
| lunarlanderv3 | dueling_dqn | -94.4838 | 11.6229 |
| lunarlanderv3 | ppo_improved | 113.2894 | 21.0423 |
| lunarlanderv3 | dueling_dqn_per | 224.2789 | 22.4283 |
| lunarlanderv3 | ddqn | -36.9221 | 31.7943 |
| lunarlanderv3 | dqn | 13.4802 | 36.088 |
| lunarlanderv3 | ddqn_per | 208.0905 | 38.9261 |
| lunarlanderv3 | dueling_dqn_per_alpha04 | 157.0634 | 40.3719 |
| lunarlanderv3 | dqn_per | 201.7374 | 51.0307 |
| lunarlanderv3 | dueling_dqn_per_alpha08 | 141.9345 | 93.2608 |
| mountaincarv0 | ddqn_unshaped | -200.0 | 0.0 |
| mountaincarv0 | dqn_per_unshaped | -200.0 | 0.0 |
| mountaincarv0 | dqn_unshaped | -200.0 | 0.0 |
| mountaincarv0 | ppo | -155.3715 | 0.0 |
| mountaincarv0 | ppo_improved | -154.224 | 0.0 |
| mountaincarv0 | dueling_dqn | -142.0788 | 16.0333 |
| mountaincarv0 | ddqn_per_unshaped | -182.5556 | 24.6702 |
| mountaincarv0 | ddqn_per | 62.1204 | 35.3157 |
| mountaincarv0 | dqn_per | -11.3602 | 85.031 |
| mountaincarv0 | ddqn | -96.9538 | 89.6966 |
| mountaincarv0 | dqn | -28.7648 | 93.7558 |
| mountaincarv0 | dueling_dqn_per | -20.5096 | 127.4249 |

## MountainCar Reward-Shaping Ablation

Rows with blank unshaped values indicate that the ablation experiments have not been run yet.

| algorithm_pair | shaped_mean_eval | unshaped_mean_eval | shaping_gain |
| --- | --- | --- | --- |
| dqn | -28.7648 | -200.0 | 171.2352 |
| ddqn | -96.9538 | -200.0 | 103.0462 |
| dqn_per | -11.3602 | -200.0 | 188.6398 |
| ddqn_per | 62.1204 | -182.5556 | 244.676 |

## PER Alpha Sensitivity

The table compares prioritization strength for Dueling DQN-PER on LunarLander.

| environment | algorithm | per_alpha | mean_eval | std_eval |
| --- | --- | --- | --- | --- |
| lunarlanderv3 | dueling_dqn_per_alpha04 | 0.4 | 157.0634 | 40.3719 |
| lunarlanderv3 | dueling_dqn_per | 0.6 | 224.2789 | 22.4283 |
| lunarlanderv3 | dueling_dqn_per_alpha08 | 0.8 | 141.9345 | 93.2608 |

## Interpretation Notes

- A strong report should emphasize both final performance and stability rather than only the maximum reward.
- MountainCar PPO remains a useful failure case because the improved PPO still struggles with sparse reward and long-horizon exploration.
- LunarLander shows the clearest benefit from PER-based value learning, especially Dueling DQN with PER.
- Reward-shaping ablation should be reported as a separate setting because shaped and unshaped MountainCar are not identical tasks.
