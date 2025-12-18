# 🚁 基于强化学习的无人机路径规划系统

## 📝 项目简介

本项目是"智能机器人技术"课程的期末大作业，基于 **OpenAI Gym FrozenLake 环境**，使用强化学习算法实现无人机的智能路径规划与自主导航。项目实现了从仿真训练到实机飞行的完整闭环，支持确定性与随机性环境下的策略学习。

### 🎯 核心特性

- ✅ **多算法实现**：Q-Learning、DQN、Nature DQN、Double DQN (DDQN)
- ✅ **双模式支持**：确定性环境 (Deterministic) 和随机性环境 (Stochastic)
- ✅ **实机演示**：基于串口通信的真实无人机控制
- ✅ **实时可视化**：训练曲线、策略热图、飞行轨迹可视化
- ✅ **边界检测**：仿真环境下的边界保护与碰撞检测
- ✅ **风力干扰**：模拟真实环境中的随机风力影响

---

## 📂 项目结构

```
├── 无人机大作业-代码库/              # 核心代码库
│   ├── agents/                      # 强化学习智能体
│   │   ├── base_agent.py           # 智能体基类
│   │   ├── q_learning_agent.py     # Q-Learning 算法
│   │   └── dqn_agent.py            # DQN/Nature DQN/DDQN 算法
│   ├── environment/                 # 环境封装
│   │   └── frozen_lake_wrapper.py  # FrozenLake 环境包装器
│   ├── utils/                       # 工具模块
│   │   ├── replay_buffer.py        # 经验回放缓冲区
│   │   └── visualization.py        # 可视化工具
│   ├── models/                      # 训练好的模型（.pth 文件）
│   ├── plots/                       # 训练曲线和可视化结果
│   ├── config.py                    # 全局配置参数
│   ├── train.py                     # 训练脚本（所有算法）
│   ├── train_dqn_models.py          # DDQN 专项训练脚本
│   ├── evaluate.py                  # 模型评估与策略可视化
│   ├── demo_flight.py               # 图形化飞行演示
│   ├── SerialThread.py              # 实机串口控制（含可视化）
│   ├── CommandConstructor.py        # 飞行指令构造器
│   └── requirements.txt             # Python 依赖库
│
└── 无人机大作业-文件库/              # 项目文档
    └── Project_Requirements.md      # 项目需求文档
```

---

## 🛠️ 环境配置

### 系统要求

- Python 3.8+
- PyTorch 1.10+
- OpenAI Gym

### 安装依赖

```bash
cd 无人机大作业-代码库
pip install -r requirements.txt
```

**主要依赖库**：

```
gym==0.26.2
torch>=1.10.0
numpy>=1.21.0
matplotlib>=3.4.0
tqdm>=4.62.0
pyserial>=3.5  # 仅实机飞行需要
```

---

## 🚀 快速开始

### 1️⃣ 训练模型

#### 训练所有算法（Q-Learning + DQN + DDQN）

```bash
python train.py
```

#### 训练指定 DDQN 模型

```bash
python train_dqn_models.py
```

**训练参数**（在 `config.py` 中配置）：

- `MAP_SIZE`: 地图尺寸（4 或 8）
- `IS_SLIPPERY`: 是否启用随机风力
- `NUM_EPISODES`: 训练回合数
- `LEARNING_RATE`: 学习率
- `GAMMA`: 折扣因子

**训练输出**：

- 模型文件保存在 `models/` 目录
- 训练曲线保存在 `plots/` 目录

---

### 2️⃣ 评估模型

```bash
python evaluate.py
```

**评估功能**：

- 📊 100 次测试的成功率统计
- 🗺️ 学习到的策略热图
- 📈 平均奖励和步数分析

---

### 3️⃣ 图形化演示

```bash
python demo_flight.py
```

**演示选项**：

1. **确定性模式**：使用 `dqn_deterministic.pth`
2. **随机模式**：使用 `ddqn.pth`（含风力干扰）

**可视化效果**：

- 实时更新的 4x4 网格地图
- 无人机位置（红色圆点）
- 飞行轨迹（蓝色虚线）
- 终端输出决策信息

---

### 4️⃣ 实机飞行

```bash
python SerialThread.py
```

**前置条件**：

- 无人机已连接至指定串口（默认 `COM5`）
- 已训练好对应模型（确定性或随机性）

**飞行模式**：

1. **键盘手动控制**
2. **确定性自主飞行**：基于 `dqn_deterministic.pth`
3. **随机性自主飞行**：基于 `ddqn.pth`（含风力模拟）

**控制逻辑**：

- 起飞 → 模型推理 → 动作执行 → 位置更新 → 到达目标/越界则降落
- 实时可视化窗口显示飞行轨迹

---

## 🧠 算法原理

### 1. Q-Learning（表格法）

- 使用 Q-Table 存储状态-动作价值
- ε-greedy 策略进行探索与利用
- 适用于小规模离散状态空间

### 2. DQN（深度 Q 网络）

- 使用神经网络逼近 Q 函数
- 经验回放（Experience Replay）
- 固定目标网络（Fixed Target Network）

### 3. Nature DQN

- 在 DQN 基础上增加了归一化和更深的网络

### 4. Double DQN (DDQN) ⭐

- 解决 DQN 的过估计问题
- 使用当前网络选择动作，目标网络评估价值
- **本项目最优算法**，成功率达 76%+

---

## 📊 实验结果

### 训练性能（8x8 地图，随机模式）

| 算法       | 训练回合数 | 最终成功率     | 平均步数 |
| ---------- | ---------- | -------------- | -------- |
| Q-Learning | 10,000     | ~65%           | 25-30    |
| DQN        | 5,000      | ~70%           | 20-25    |
| DDQN       | 5,000      | **~76%** | 18-22    |

### 可视化示例

**训练曲线**：

- `plots/ddqn_loss.png`：DDQN 损失函数变化
- `plots/ddqn_rewards.png`：累计奖励变化

**策略热图**：

- 显示每个格子的最优动作方向
- 绿色箭头指向目标

---

## 🎮 实机演示说明

### 硬件配置

- 无人机：支持串口控制的四轴飞行器
- 通信：USB 转串口（COM5，波特率 115200）
- 场景：4x4 物理网格（60cm × 60cm 每格）

### 演示流程

1. **起飞**：无人机从 (0,0) 起飞至稳定高度
2. **决策**：加载训练好的 DDQN 模型
3. **执行**：根据模型输出执行前/后/左/右动作
4. **反馈**：更新当前位置，检查边界和目标
5. **降落**：到达目标或超时后安全降落

### 特殊处理

- **边界保护**：到达边界时拒绝越界动作
- **风力模拟**（随机模式）：33% 概率随机偏移动作方向
- **实时可视化**：弹出窗口显示飞行轨迹

---

## 🎨 可视化窗口说明

### 网格元素

- **S (绿色)**：起点 (Start)
- **F (白色)**：安全区域 (Frozen)
- **H (灰色)**：危险区域 (Hole)
- **G (黄色)**：目标点 (Goal)
- **红色圆点**：无人机当前位置
- **蓝色虚线**：历史飞行轨迹

---

## 🌟 项目亮点

1. **完整的 RL 项目流程**：从环境搭建到模型训练、评估、部署
2. **多算法对比**：实现并对比了 4 种不同的 RL 算法
3. **虚实结合**：从 Gym 仿真到真实无人机控制的完整链路
4. **鲁棒性设计**：边界检测、风力干扰、异常处理
5. **丰富的可视化**：训练曲线、策略热图、实时飞行动画
6. **模块化代码**：清晰的类设计和接口定义

---

## 📚 参考资料

- [OpenAI Gym FrozenLake 环境](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- [Playing Atari with Deep Reinforcement Learning (DQN)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning (Nature DQN)](https://www.nature.com/articles/nature14236)
- [Deep Reinforcement Learning with Double Q-learning (DDQN)](https://arxiv.org/abs/1509.06461)

---

## 👨‍💻 作者

**课程**：智能机器人技术
**项目**：期末大作业 - 基于强化学习的无人机路径规划
**时间**：2025年12月

---

## 📄 许可证

本项目仅用于学习和研究目的。

---

## 🙏 致谢

感谢 OpenAI Gym 提供的仿真环境，以及 PyTorch 社区的技术支持。

---

**⭐ 如果这个项目对你有帮助，请给一个 Star！**
