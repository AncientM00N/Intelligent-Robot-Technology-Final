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
    'num_episodes': 20000,      # 训练回合数
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
    'num_episodes': 20000,       # 训练回合数
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

# ==================== SARSA 配置 ====================
SARSA_CONFIG = {
    'learning_rate': 0.8,       # 学习率 alpha
    'discount_factor': 0.995,    # 折扣因子 gamma
    'epsilon_start': 1.0,       # 初始探索率
    'epsilon_end': 0.01,        # 最终探索率
    'epsilon_decay': 0.997,     # 探索率衰减
    'num_episodes': 20000,      # 训练回合数（4x4 地图）
    'max_steps': 100,           # 每回合最大步数
}

# ==================== Benchmark 配置 ====================
BENCHMARK_CONFIG = {
    # 地图配置（支持命令行覆盖）
    'map_sizes': ['4x4', '8x8'],
    'modes': ['deterministic', 'stochastic'],
    
    # 训练回合数（针对不同地图大小优化）
    'episodes': {
        'q_learning': {'4x4': 5000, '8x8': 10000},
        'sarsa': {'4x4': 5000, '8x8': 10000},
        'dqn': {'4x4': 5000, '8x8': 10000},
        'ddqn': {'4x4': 5000, '8x8': 10000},
    },
    
    # 结果保存路径模板
    'model_dir_template': 'models/benchmark_{map_size}_{mode}',
    'result_dir_template': 'results/{map_size}_{mode}',
    'plot_dir_template': 'plots/comparison/{map_size}_{mode}',
    
    # 评估参数
    'eval_episodes': 100,       # 评估时的测试回合数
    'eval_max_steps': 100,      # 评估时每回合最大步数
    
    # 可视化参数
    'plot_window_size': 100,    # 移动平均窗口大小
    'success_threshold': 0.7,   # 成功率阈值（用于收敛分析）
    
    # 算法颜色配置（用于绘图）
    'algorithm_colors': {
        'Q-Learning': '#1f77b4',  # 蓝色
        'SARSA': '#ff7f0e',       # 橙色
        'DQN': '#2ca02c',         # 绿色
        'DDQN': '#d62728',        # 红色
    },
}

# ==================== 可视化配置 ====================
VIS_CONFIG = {
    'plot_interval': 100,       # 每隔多少回合绘制一次
    'save_plots': True,         # 是否保存图表
    'plot_dir': './plots',      # 图表保存目录
}

