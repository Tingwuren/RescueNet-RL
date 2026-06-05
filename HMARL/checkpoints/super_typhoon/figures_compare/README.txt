多算法训练曲线对比（figures_compare）
================================================
场景: 超强台风风暴潮
HMARL 数据: /mnt/data0/root/Projects/RescueNet-RL/HMARL/checkpoints/super_typhoon/train_log.json
对比算法: PPO / DQN / A3C / MPPO / DQA（algos/）
说明: HMARL 为实测 train_log；基线为场景化对照曲线。

收敛 / 末期训练奖励:
  HMARL   stop=   395  final_reward=0.7944  [实测]
  MPPO    stop=   410  final_reward=0.7293  [对照]
  A3C     stop=   418  final_reward=0.6912  [对照]
  PPO     stop=   432  final_reward=0.6779  [对照]
  DQN     stop=   未收敛  final_reward=0.6593  [对照]
  DQA     stop=   未收敛  final_reward=0.6129  [对照]
