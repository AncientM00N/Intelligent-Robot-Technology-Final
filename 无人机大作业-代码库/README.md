# FrozenLake 强化学习无人机飞行控制

基于 OpenAI Gym FrozenLake 环境的强化学习项目，实现 Q-Learning 和 DQN/DDQN 算法。

## 项目结构

```
├── config.py              # 配置参数（学习率、回合数等）
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── demo_flight.py         # 飞行演示（含边界检测+风力模拟）
├── requirements.txt       # 依赖包
│
├── environment/
│   └── frozen_lake_wrapper.py   # Gym环境封装，支持4x4/8x8，Stochastic/Deterministic
│
├── agents/
│   ├── base_agent.py            # Agent基类
│   ├── q_learning_agent.py      # Q-Learning (Q-Table)
│   └── dqn_agent.py             # DQN/Double DQN (PyTorch)
│
└── utils/
    ├── replay_buffer.py         # 经验回放缓冲区
    └── visualization.py         # 可视化（Reward曲线、网格动画）
```

## 快速开始

### 1. 安装依赖
```bash
pip install gym numpy torch matplotlib tqdm
```

### 2. 训练
```bash
# 训练所有算法（Q-Learning + DQN + DDQN）并对比
python train.py --algorithm all

# 单独训练
python train.py --algorithm q_learning    # Q-Learning
python train.py --algorithm dqn           # DQN
python train.py --algorithm ddqn          # Double DQN

# 可选参数
python train.py --map_size 8x8            # 使用8x8地图
python train.py --no-slippery             # 确定性模式（无风力）
python train.py --episodes 5000           # 指定回合数
```

### 3. 评估
```bash
python evaluate.py                        # 评估所有模型
python evaluate.py --visualize            # 可视化策略
python evaluate.py --demo                 # 单回合演示
```

### 4. 飞行演示
```bash
python demo_flight.py
# 选择: 1=模型演示, 2=手动控制, 3=随机演示
```

## 核心功能

| 功能 | 说明 |
|------|------|
| 边界检测 | `demo_flight.py` 中维护 `forward/back/left/right_steps` |
| 风力干扰 | Stochastic 模式下 1/3 概率向左/右偏移 |
| 实时显示 | 打印当前网格坐标 `(row, col)` |
| 训练曲线 | 自动保存到 `plots/` 目录 |

## 输出文件

- `models/q_learning.npz` - Q-Learning 模型
- `models/dqn.pth` - DQN 模型
- `models/ddqn.pth` - Double DQN 模型
- `plots/*.png` - 训练曲线图

## 动作定义

| 动作 | 编号 | 方向 |
|------|------|------|
| LEFT | 0 | ← |
| DOWN | 1 | ↓ (前进) |
| RIGHT | 2 | → |
| UP | 3 | ↑ (后退) |

## 实机飞行

### 运行实机控制
```bash
python real_flight.py
```

### 模式选择
| 模式 | 说明 |
|------|------|
| 1. 预设路径 | 按固定动作序列飞行，简单可靠 |
| 2. RL模型决策 | 用训练好的模型实时决策 |
| 3/4. 模拟模式 | 不实际飞行，用于测试代码 |

### 硬件要求
- 4×4 网格地图，每格 35cm
- 每格贴 AprilTag (ID 0-15)
- 无人机通过串口连接 (默认 COM7)

### 预设路径示例
```
默认路径: 下→下→右→下→右→右
对应动作: [1, 1, 2, 1, 2, 2]
```

