"""
统一训练脚本
支持 Q-Learning、SARSA、DQN、DDQN 四种算法的训练和对比
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional

from config import (
    QLEARNING_CONFIG, SARSA_CONFIG, DQN_CONFIG, 
    ENV_CONFIG, BENCHMARK_CONFIG
)
from environment.frozen_lake_wrapper import FrozenLakeWrapper
from agents.q_learning_agent import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.dqn_agent import DQNAgent
from utils.logger import TrainingLogger, BenchmarkLogger
from utils.plotter import BenchmarkPlotter


def save_benchmark_config(result_dir: str, map_size: str, mode: str, algorithms: list):
    """
    保存 Benchmark 实验的统一参数配置文件
    
    Args:
        result_dir: 结果保存目录
        map_size: 地图大小
        mode: 环境模式
        algorithms: 算法列表 [(key, name, config), ...]
    """
    from datetime import datetime
    
    config_path = os.path.join(result_dir, '参数配置.txt')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("强化学习算法 Benchmark 实验 - 参数配置汇总\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"地图大小: {map_size}\n")
        f.write(f"环境模式: {mode} ({'随机/有风' if mode == 'stochastic' else '确定性/无风'})\n")
        f.write(f"算法数量: {len(algorithms)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # 为每个算法写入配置
        for idx, (algo_key, algo_name, base_config) in enumerate(algorithms, 1):
            # 获取实际使用的配置
            config = base_config.copy()
            if algo_key in BENCHMARK_CONFIG['episodes']:
                config['num_episodes'] = BENCHMARK_CONFIG['episodes'][algo_key].get(
                    map_size, config.get('num_episodes', 'N/A')
                )
            
            f.write(f"【算法 {idx}/4】 {algo_name}\n")
            f.write("-" * 80 + "\n")
            
            # 算法类型
            if algo_key in ['q_learning', 'sarsa']:
                f.write(f"算法类型: 表格法 (Tabular Method)\n")
                f.write(f"值函数表示: Q-Table\n")
            else:
                f.write(f"算法类型: 深度学习法 (Deep Reinforcement Learning)\n")
                f.write(f"值函数表示: 神经网络\n")
            
            f.write(f"\n核心参数:\n")
            f.write(f"  训练回合数 (num_episodes):     {config.get('num_episodes', 'N/A'):>8}\n")
            f.write(f"  每回合最大步数 (max_steps):     {config.get('max_steps', 'N/A'):>8}\n")
            f.write(f"  学习率 (learning_rate):        {config.get('learning_rate', 'N/A'):>8}\n")
            f.write(f"  折扣因子 (discount_factor):    {config.get('discount_factor', 'N/A'):>8}\n")
            
            f.write(f"\n探索策略 (ε-greedy):\n")
            f.write(f"  初始探索率 (epsilon_start):    {config.get('epsilon_start', 'N/A'):>8}\n")
            f.write(f"  最终探索率 (epsilon_end):      {config.get('epsilon_end', 'N/A'):>8}\n")
            f.write(f"  探索率衰减 (epsilon_decay):    {config.get('epsilon_decay', 'N/A'):>8}\n")
            
            # DQN 特有参数
            if algo_key in ['dqn', 'ddqn']:
                f.write(f"\n深度学习特有参数:\n")
                f.write(f"  批量大小 (batch_size):         {config.get('batch_size', 'N/A'):>8}\n")
                f.write(f"  经验回放缓冲区 (buffer_size):  {config.get('buffer_size', 'N/A'):>8}\n")
                f.write(f"  目标网络更新频率 (target_update_freq): {config.get('target_update_freq', 'N/A'):>4}\n")
                f.write(f"  隐藏层大小 (hidden_size):      {config.get('hidden_size', 'N/A'):>8}\n")
                f.write(f"  Double DQN:                    {config.get('use_double_dqn', 'N/A')}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        # 添加算法对比说明
        f.write("算法对比说明\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【表格法 vs 深度学习法】\n\n")
        
        f.write("表格法 (Q-Learning, SARSA):\n")
        f.write("  优点: 简单直观，理论保证收敛（充分探索下）\n")
        f.write("  缺点: 无泛化能力，每个状态必须单独学习\n")
        f.write("  适用: 小状态空间 (<100 个状态)\n\n")
        
        f.write("深度学习法 (DQN, DDQN):\n")
        f.write("  优点: 强大泛化能力，经验回放，目标网络稳定训练\n")
        f.write("  缺点: 更复杂，超参数敏感\n")
        f.write("  适用: 大状态空间 (>1000 个状态)\n\n")
        
        f.write("【Q-Learning vs SARSA】\n\n")
        
        f.write("Q-Learning (Off-Policy):\n")
        f.write("  更新: 使用 max Q(s',a') - 学习最优策略\n")
        f.write("  特点: 更激进，更适合随机环境\n\n")
        
        f.write("SARSA (On-Policy):\n")
        f.write("  更新: 使用实际执行的 Q(s',a') - 学习执行策略\n")
        f.write("  特点: 更保守，在随机环境中可能表现较差\n\n")
        
        f.write("【DQN vs DDQN】\n\n")
        
        f.write("DQN (Nature DQN):\n")
        f.write("  目标: max Q(s',a'; θ⁻) - 同一网络选择和评估\n")
        f.write("  问题: 容易过估计 Q 值\n\n")
        
        f.write("DDQN (Double DQN):\n")
        f.write("  目标: Q(s', argmax Q(s',a;θ); θ⁻) - 解耦选择和评估\n")
        f.write("  优势: 减少过估计，通常表现更好\n\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"配置文件生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
    
    print(f"参数配置文件已保存: {config_path}")


def train_q_learning(env: FrozenLakeWrapper, 
                     config: dict,
                     logger: Optional[TrainingLogger] = None,
                     verbose: bool = True) -> Tuple[QLearningAgent, list, list]:
    """
    训练 Q-Learning Agent
    
    Args:
        env: FrozenLake 环境
        config: 训练配置
        logger: 训练日志记录器
        verbose: 是否打印训练进度
        
    Returns:
        (agent, episode_rewards, losses)
    """
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=config['learning_rate'],
        discount_factor=config['discount_factor'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay']
    )
    
    num_episodes = config['num_episodes']
    max_steps = config['max_steps']
    
    if verbose:
        print(f"\n{'='*60}")
        print("Q-Learning 训练开始")
        print(f"{'='*60}")
        print(f"状态空间: {env.n_states}, 动作空间: {env.n_actions}")
        print(f"总回合数: {num_episodes}")
        print(f"学习率: {config['learning_rate']}, 折扣因子: {config['discount_factor']}")
        print(f"{'='*60}\n")
    
    # 训练循环
    iterator = tqdm(range(num_episodes), desc="Q-Learning 训练") if verbose else range(num_episodes)
    
    for episode in iterator:
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 更新 Agent
            agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # 记录回合数据
        agent.record_episode(total_reward, step + 1)
        
        # 记录到日志
        if logger:
            logger.log_episode(episode, total_reward, step + 1, agent.epsilon)
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 打印进度
        if verbose and (episode + 1) % 1000 == 0:
            stats = agent.get_training_stats()
            if logger:
                logger.print_progress(
                    episode + 1, num_episodes,
                    stats['avg_reward_100'],
                    stats['success_rate_100'],
                    agent.epsilon
                )
    
    if verbose:
        print(f"\n训练完成!")
        stats = agent.get_training_stats()
        print(f"最终成功率 (100回合): {stats['success_rate_100']:.2%}")
        print(f"最终平均奖励 (100回合): {stats['avg_reward_100']:.3f}")
    
    return agent, agent.episode_rewards, []


def train_sarsa(env: FrozenLakeWrapper,
                config: dict,
                logger: Optional[TrainingLogger] = None,
                verbose: bool = True) -> Tuple[SARSAAgent, list, list]:
    """
    训练 SARSA Agent
    
    Args:
        env: FrozenLake 环境
        config: 训练配置
        logger: 训练日志记录器
        verbose: 是否打印训练进度
        
    Returns:
        (agent, episode_rewards, losses)
    """
    agent = SARSAAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=config['learning_rate'],
        discount_factor=config['discount_factor'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay']
    )
    
    num_episodes = config['num_episodes']
    max_steps = config['max_steps']
    
    if verbose:
        print(f"\n{'='*60}")
        print("SARSA 训练开始")
        print(f"{'='*60}")
        print(f"状态空间: {env.n_states}, 动作空间: {env.n_actions}")
        print(f"总回合数: {num_episodes}")
        print(f"学习率: {config['learning_rate']}, 折扣因子: {config['discount_factor']}")
        print(f"{'='*60}\n")
    
    # 训练循环
    iterator = tqdm(range(num_episodes), desc="SARSA 训练") if verbose else range(num_episodes)
    
    for episode in iterator:
        state, _ = env.reset()
        action = agent.select_action(state, training=True)  # SARSA: 预先选择动作
        total_reward = 0
        
        for step in range(max_steps):
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 选择下一个动作（如果未结束）
            if not done:
                next_action = agent.select_action(next_state, training=True)
            else:
                next_action = 0  # 终止状态，动作无意义
            
            # SARSA 更新：使用实际选择的 next_action
            agent.update(state, action, reward, next_state, next_action, done)
            
            total_reward += reward
            
            if done:
                break
            
            # 只有未结束才传递状态和动作
            state = next_state
            action = next_action
        
        # 记录回合数据
        agent.record_episode(total_reward, step + 1)
        
        # 记录到日志
        if logger:
            logger.log_episode(episode, total_reward, step + 1, agent.epsilon)
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 打印进度
        if verbose and (episode + 1) % 1000 == 0:
            stats = agent.get_training_stats()
            if logger:
                logger.print_progress(
                    episode + 1, num_episodes,
                    stats['avg_reward_100'],
                    stats['success_rate_100'],
                    agent.epsilon
                )
    
    if verbose:
        print(f"\n训练完成!")
        stats = agent.get_training_stats()
        print(f"最终成功率 (100回合): {stats['success_rate_100']:.2%}")
        print(f"最终平均奖励 (100回合): {stats['avg_reward_100']:.3f}")
    
    return agent, agent.episode_rewards, []


def train_dqn(env: FrozenLakeWrapper,
              config: dict,
              use_double_dqn: bool = False,
              logger: Optional[TrainingLogger] = None,
              verbose: bool = True) -> Tuple[DQNAgent, list, list]:
    """
    训练 DQN/DDQN Agent
    
    Args:
        env: FrozenLake 环境
        config: 训练配置
        use_double_dqn: 是否使用 Double DQN
        logger: 训练日志记录器
        verbose: 是否打印训练进度
        
    Returns:
        (agent, episode_rewards, losses)
    """
    agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=config['learning_rate'],
        discount_factor=config['discount_factor'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay'],
        batch_size=config['batch_size'],
        buffer_size=config['buffer_size'],
        target_update_freq=config['target_update_freq'],
        hidden_size=config['hidden_size'],
        use_double_dqn=use_double_dqn
    )
    
    num_episodes = config['num_episodes']
    max_steps = config['max_steps']
    
    algo_name = "Double DQN" if use_double_dqn else "DQN"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"{algo_name} 训练开始")
        print(f"{'='*60}")
        print(f"状态空间: {env.n_states}, 动作空间: {env.n_actions}")
        print(f"总回合数: {num_episodes}")
        print(f"批量大小: {config['batch_size']}, 缓冲区: {config['buffer_size']}")
        print(f"目标网络更新频率: {config['target_update_freq']}")
        print(f"{'='*60}\n")
    
    # 训练循环
    iterator = tqdm(range(num_episodes), desc=f"{algo_name} 训练") if verbose else range(num_episodes)
    
    for episode in iterator:
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 更新 Agent
            agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # 记录回合数据
        agent.record_episode(total_reward, step + 1)
        
        # 记录到日志
        if logger:
            avg_loss = np.mean(agent.losses[-100:]) if len(agent.losses) >= 100 else None
            logger.log_episode(episode, total_reward, step + 1, agent.epsilon, loss=avg_loss)
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 打印进度
        if verbose and (episode + 1) % 200 == 0:
            stats = agent.get_training_stats()
            if logger:
                logger.print_progress(
                    episode + 1, num_episodes,
                    stats['avg_reward_100'],
                    stats['success_rate_100'],
                    agent.epsilon,
                    avg_loss=stats.get('avg_loss')
                )
    
    if verbose:
        print(f"\n训练完成!")
        stats = agent.get_training_stats()
        print(f"最终成功率 (100回合): {stats['success_rate_100']:.2%}")
        print(f"最终平均奖励 (100回合): {stats['avg_reward_100']:.3f}")
    
    return agent, agent.episode_rewards, agent.losses


def train_single_algorithm(algorithm: str, 
                           map_size: str, 
                           mode: str,
                           save_dir: str,
                           episodes: Optional[int] = None) -> Tuple:
    """
    训练单个算法
    
    Args:
        algorithm: 算法名称 ('q_learning', 'sarsa', 'dqn', 'ddqn')
        map_size: 地图大小 ('4x4' 或 '8x8')
        mode: 模式 ('deterministic' 或 'stochastic')
        save_dir: 模型保存目录
        episodes: 训练回合数（None则使用配置文件默认值）
        
    Returns:
        (agent, rewards, losses)
    """
    # 创建环境
    is_slippery = (mode == 'stochastic')
    env = FrozenLakeWrapper(map_size=map_size, is_slippery=is_slippery)
    
    print(f"\n环境配置:")
    print(f"  地图大小: {map_size}")
    print(f"  模式: {mode} ({'随机' if is_slippery else '确定性'})")
    env.print_map()
    
    # 获取配置
    if algorithm == 'q_learning':
        config = QLEARNING_CONFIG.copy()
        if episodes:
            config['num_episodes'] = episodes
        elif map_size in BENCHMARK_CONFIG['episodes']['q_learning']:
            config['num_episodes'] = BENCHMARK_CONFIG['episodes']['q_learning'][map_size]
        
        agent, rewards, losses = train_q_learning(env, config, verbose=True)
        save_path = os.path.join(save_dir, 'q_learning.npz')
        agent.save(save_path)
        
    elif algorithm == 'sarsa':
        config = SARSA_CONFIG.copy()
        if episodes:
            config['num_episodes'] = episodes
        elif map_size in BENCHMARK_CONFIG['episodes']['sarsa']:
            config['num_episodes'] = BENCHMARK_CONFIG['episodes']['sarsa'][map_size]
        
        agent, rewards, losses = train_sarsa(env, config, verbose=True)
        save_path = os.path.join(save_dir, 'sarsa.npz')
        agent.save(save_path)
        
    elif algorithm == 'dqn':
        config = DQN_CONFIG.copy()
        if episodes:
            config['num_episodes'] = episodes
        elif map_size in BENCHMARK_CONFIG['episodes']['dqn']:
            config['num_episodes'] = BENCHMARK_CONFIG['episodes']['dqn'][map_size]
        
        agent, rewards, losses = train_dqn(env, config, use_double_dqn=False, verbose=True)
        save_path = os.path.join(save_dir, 'dqn.pth')
        agent.save(save_path)
        
    elif algorithm == 'ddqn':
        config = DQN_CONFIG.copy()
        if episodes:
            config['num_episodes'] = episodes
        elif map_size in BENCHMARK_CONFIG['episodes']['ddqn']:
            config['num_episodes'] = BENCHMARK_CONFIG['episodes']['ddqn'][map_size]
        
        agent, rewards, losses = train_dqn(env, config, use_double_dqn=True, verbose=True)
        save_path = os.path.join(save_dir, 'ddqn.pth')
        agent.save(save_path)
    else:
        raise ValueError(f"未知算法: {algorithm}")
    
    env.close()
    return agent, rewards, losses


def run_benchmark(map_size: str, mode: str):
    """
    运行完整的 Benchmark 实验（训练所有4种算法并生成对比）
    
    Args:
        map_size: 地图大小 ('4x4' 或 '8x8')
        mode: 模式 ('deterministic' 或 'stochastic')
    """
    print("\n" + "="*80)
    print(f"开始 Benchmark 实验: {map_size} - {mode}")
    print("="*80)
    
    # 创建目录
    model_dir = BENCHMARK_CONFIG['model_dir_template'].format(map_size=map_size, mode=mode)
    result_dir = BENCHMARK_CONFIG['result_dir_template'].format(map_size=map_size, mode=mode)
    plot_dir = BENCHMARK_CONFIG['plot_dir_template'].format(map_size=map_size, mode=mode)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # 创建 Benchmark 日志管理器
    benchmark_logger = BenchmarkLogger('results', map_size, mode)
    
    # 创建环境
    is_slippery = (mode == 'stochastic')
    env = FrozenLakeWrapper(map_size=map_size, is_slippery=is_slippery)
    
    print(f"\n环境配置:")
    print(f"  地图大小: {map_size}")
    print(f"  模式: {mode} ({'随机' if is_slippery else '确定性'})")
    env.print_map()
    
    # 训练结果存储
    results = {}
    losses_dict = {}
    
    algorithms = [
        ('q_learning', 'Q-Learning', QLEARNING_CONFIG),
        ('sarsa', 'SARSA', SARSA_CONFIG),
        ('dqn', 'DQN', DQN_CONFIG),
        ('ddqn', 'DDQN', DQN_CONFIG),
    ]
    
    for idx, (algo_key, algo_name, base_config) in enumerate(algorithms, 1):
        print(f"\n{'='*80}")
        print(f"【{idx}/4】训练 {algo_name}")
        print(f"{'='*80}")
        
        # 配置训练回合数
        config = base_config.copy()
        if algo_key in BENCHMARK_CONFIG['episodes']:
            config['num_episodes'] = BENCHMARK_CONFIG['episodes'][algo_key].get(map_size, config['num_episodes'])
        
        # 创建日志记录器
        logger = benchmark_logger.create_logger(algo_name, config)
        
        # 训练
        if algo_key == 'q_learning':
            agent, rewards, losses = train_q_learning(env, config, logger, verbose=True)
            agent.save(os.path.join(model_dir, 'q_learning.npz'))
        elif algo_key == 'sarsa':
            agent, rewards, losses = train_sarsa(env, config, logger, verbose=True)
            agent.save(os.path.join(model_dir, 'sarsa.npz'))
        elif algo_key == 'dqn':
            agent, rewards, losses = train_dqn(env, config, use_double_dqn=False, logger=logger, verbose=True)
            agent.save(os.path.join(model_dir, 'dqn.pth'))
        elif algo_key == 'ddqn':
            agent, rewards, losses = train_dqn(env, config, use_double_dqn=True, logger=logger, verbose=True)
            agent.save(os.path.join(model_dir, 'ddqn.pth'))
        
        results[algo_name] = rewards
        if losses:
            losses_dict[algo_name] = losses
    
    env.close()
    
    # 保存日志
    benchmark_logger.save_all_logs()
    
    # 生成对比图表
    print("\n" + "="*80)
    print("生成对比图表...")
    print("="*80)
    
    plotter = BenchmarkPlotter(BENCHMARK_CONFIG['algorithm_colors'])
    window_size = BENCHMARK_CONFIG['plot_window_size']
    
    # 1. 单图对比
    plotter.plot_training_comparison(
        results,
        os.path.join(plot_dir, 'training_comparison_single.png'),
        window_size=window_size,
        style='single',
        title=f'算法训练对比 ({map_size} - {mode})'
    )
    
    # 2. 网格对比
    plotter.plot_training_comparison(
        results,
        os.path.join(plot_dir, 'training_comparison_grid.png'),
        window_size=window_size,
        style='grid',
        title=f'算法训练对比 ({map_size} - {mode})'
    )
    
    # 3. 成功率对比
    plotter.plot_success_rate_comparison(
        results,
        os.path.join(plot_dir, 'success_rate_comparison.png'),
        window_size=window_size,
        title=f'算法成功率对比 ({map_size} - {mode})'
    )
    
    # 4. 损失对比（仅DQN和DDQN）
    if 'DQN' in losses_dict and 'DDQN' in losses_dict:
        plotter.plot_loss_comparison(
            losses_dict['DQN'],
            losses_dict['DDQN'],
            os.path.join(plot_dir, 'loss_comparison.png'),
            window_size=window_size,
            title=f'DQN vs DDQN 损失对比 ({map_size} - {mode})'
        )
    
    # 5. 收敛表格
    plotter.save_convergence_table(
        results,
        os.path.join(result_dir, 'convergence_table.md'),
        threshold=BENCHMARK_CONFIG['success_threshold'],
        window_size=window_size
    )
    
    # 6. 生成对比报告
    benchmark_logger.generate_comparison_report(
        os.path.join(result_dir, 'comparison_report.md')
    )
    
    # 7. 生成统一的参数配置文件
    save_benchmark_config(result_dir, map_size, mode, [
        ('q_learning', 'Q-Learning', QLEARNING_CONFIG),
        ('sarsa', 'SARSA', SARSA_CONFIG),
        ('dqn', 'DQN', DQN_CONFIG),
        ('ddqn', 'DDQN', DQN_CONFIG),
    ])
    
    print(f"\n{'='*80}")
    print("Benchmark 实验完成!")
    print(f"{'='*80}")
    print(f"模型保存在: {model_dir}")
    print(f"结果保存在: {result_dir}")
    print(f"图表保存在: {plot_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='FrozenLake 强化学习训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练单个算法
  python train.py --algorithm sarsa --map_size 8x8 --mode stochastic
  
  # 运行完整 Benchmark（4种算法对比）
  python train.py --benchmark --map_size 8x8 --mode stochastic
  
  # 训练 DDQN（指定回合数）
  python train.py --algorithm ddqn --map_size 4x4 --episodes 3000
        """
    )
    
    parser.add_argument('--algorithm', type=str, 
                       choices=['q_learning', 'sarsa', 'dqn', 'ddqn'],
                       help='训练算法 (不指定则需要使用 --benchmark)')
    parser.add_argument('--map_size', type=str, default='4x4',
                       choices=['4x4', '8x8'],
                       help='地图大小 (默认: 4x4)')
    parser.add_argument('--mode', type=str, default='stochastic',
                       choices=['deterministic', 'stochastic'],
                       help='环境模式 (默认: stochastic)')
    parser.add_argument('--episodes', type=int,
                       help='训练回合数（覆盖配置文件）')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='模型保存目录 (默认: models)')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行完整 Benchmark 实验（训练所有4种算法）')
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.benchmark and not args.algorithm:
        parser.error("必须指定 --algorithm 或 --benchmark")
    
    if args.benchmark:
        # Benchmark 模式
        run_benchmark(args.map_size, args.mode)
    else:
        # 单算法训练模式
        model_dir = os.path.join(args.save_dir, f'{args.map_size}_{args.mode}')
        os.makedirs(model_dir, exist_ok=True)
        
        train_single_algorithm(
            args.algorithm,
            args.map_size,
            args.mode,
            model_dir,
            args.episodes
        )
        
        print(f"\n训练完成! 模型已保存到: {model_dir}")


if __name__ == '__main__':
    main()
