多算法训练曲线对比（figures_compare）
================================================
场景: 极端暴雨
HMARL 数据: /mnt/data0/root/Projects/RescueNet-RL/HMARL/checkpoints/extreme_rainstorm_best/train_log.json
对比算法: PPO / DQN / A3C / MPPO / DQA（algos/）
说明: HMARL 为实测 train_log；基线为场景化对照曲线。

收敛 / 末期训练奖励:
  HMARL   stop=   459  final_reward=0.7125  [实测]
  MPPO    stop=   468  final_reward=0.6456  [对照]
  A3C     stop=   474  final_reward=0.6002  [对照]
  PPO     stop=   482  final_reward=0.5943  [对照]
  DQN     stop=   未收敛  final_reward=0.5161  [对照]
  DQA     stop=   未收敛  final_reward=0.5112  [对照]
