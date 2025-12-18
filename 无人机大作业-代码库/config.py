"""
全局配置参数
"""

# ==================== 环境配置 ====================
ENV_CONFIG = {
    'map_size': '4x4',          # 地图大小: '4x4' 或 '8x8'
    'is_slippery': True,        # 是否启用随机模式（风力干扰）
    'render_mode': 'human',     # 渲染模式: 'human', 'rgb_array', 'ansi'
}

# ==================== Q-Learning 配置 ====================
QLEARNING_CONFIG = {
    'learning_rate': 0.8,       # 学习率 alpha
    'discount_factor': 0.95,    # 折扣因子 gamma
    'epsilon_start': 1.0,       # 初始探索率
    'epsilon_end': 0.01,        # 最终探索率
    'epsilon_decay': 0.995,     # 探索率衰减
    'num_episodes': 10000,      # 训练回合数
    'max_steps': 100,           # 每回合最大步数
}

# ==================== DQN 配置 ====================
DQN_CONFIG = {
    'learning_rate': 0.001,     # 学习率
    'discount_factor': 0.99,    # 折扣因子 gamma
    'epsilon_start': 1.0,       # 初始探索率
    'epsilon_end': 0.01,        # 最终探索率
    'epsilon_decay': 0.995,     # 探索率衰减
    'batch_size': 64,           # 批量大小
    'buffer_size': 10000,       # 经验回放缓冲区大小
    'target_update_freq': 100,  # 目标网络更新频率
    'num_episodes': 2000,       # 训练回合数
    'max_steps': 100,           # 每回合最大步数
    'hidden_size': 64,          # 隐藏层大小
    'use_double_dqn': True,     # 是否使用 Double DQN
}

# ==================== 演示配置 ====================
DEMO_CONFIG = {
    'grid_distance': 50,        # 网格间距（厘米），用于实机飞行
    'flight_height': 100,       # 飞行高度（厘米）
    'step_delay': 1.0,          # 每步延迟（秒）
    'show_animation': True,     # 是否显示动画
}

# ==================== 可视化配置 ====================
VIS_CONFIG = {
    'plot_interval': 100,       # 每隔多少回合绘制一次
    'save_plots': True,         # 是否保存图表
    'plot_dir': './plots',      # 图表保存目录
}

