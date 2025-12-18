"""
训练脚本
支持 Q-Learning 和 DQN 算法的训练
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

from config import QLEARNING_CONFIG, DQN_CONFIG, ENV_CONFIG
from environment.frozen_lake_wrapper import FrozenLakeWrapper
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from utils.visualization import Visualizer


def train_q_learning(env: FrozenLakeWrapper, 
                     config: dict = None,
                     verbose: bool = True) -> QLearningAgent:
    """
    训练 Q-Learning Agent
    
    Args:
        env: FrozenLake 环境
        config: 训练配置
        verbose: 是否打印训练进度
        
    Returns:
        训练好的 Agent
    """
    if config is None:
        config = QLEARNING_CONFIG
    
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
    
    print(f"\n{'='*50}")
    print("Q-Learning 训练开始")
    print(f"{'='*50}")
    print(f"状态空间: {env.n_states}, 动作空间: {env.n_actions}")
    print(f"总回合数: {num_episodes}")
    print(f"学习率: {config['learning_rate']}, 折扣因子: {config['discount_factor']}")
    print(f"{'='*50}\n")
    
    # 训练循环
    iterator = tqdm(range(num_episodes), desc="训练中") if verbose else range(num_episodes)
    
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
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 打印进度
        if verbose and (episode + 1) % 1000 == 0:
            stats = agent.get_training_stats()
            tqdm.write(f"回合 {episode+1}/{num_episodes} | "
                      f"成功率: {stats['success_rate_100']:.2%} | "
                      f"平均奖励: {stats['avg_reward_100']:.3f} | "
                      f"探索率: {agent.epsilon:.3f}")
    
    print(f"\n训练完成!")
    stats = agent.get_training_stats()
    print(f"最终成功率 (100回合): {stats['success_rate_100']:.2%}")
    print(f"最终平均奖励 (100回合): {stats['avg_reward_100']:.3f}")
    
    return agent


def train_dqn(env: FrozenLakeWrapper,
              config: dict = None,
              verbose: bool = True) -> DQNAgent:
    """
    训练 DQN Agent
    
    Args:
        env: FrozenLake 环境
        config: 训练配置
        verbose: 是否打印训练进度
        
    Returns:
        训练好的 Agent
    """
    if config is None:
        config = DQN_CONFIG
    
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
        use_double_dqn=config['use_double_dqn']
    )
    
    num_episodes = config['num_episodes']
    max_steps = config['max_steps']
    
    algo_name = "Double DQN" if config['use_double_dqn'] else "DQN"
    
    print(f"\n{'='*50}")
    print(f"{algo_name} 训练开始")
    print(f"{'='*50}")
    print(f"状态空间: {env.n_states}, 动作空间: {env.n_actions}")
    print(f"总回合数: {num_episodes}")
    print(f"批量大小: {config['batch_size']}, 缓冲区: {config['buffer_size']}")
    print(f"目标网络更新频率: {config['target_update_freq']}")
    print(f"{'='*50}\n")
    
    # 训练循环
    iterator = tqdm(range(num_episodes), desc="训练中") if verbose else range(num_episodes)
    
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
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 打印进度
        if verbose and (episode + 1) % 200 == 0:
            stats = agent.get_training_stats()
            loss_str = f"损失: {stats.get('avg_loss', 0):.4f}" if 'avg_loss' in stats else ""
            tqdm.write(f"回合 {episode+1}/{num_episodes} | "
                      f"成功率: {stats['success_rate_100']:.2%} | "
                      f"平均奖励: {stats['avg_reward_100']:.3f} | "
                      f"探索率: {agent.epsilon:.3f} | {loss_str}")
    
    print(f"\n训练完成!")
    stats = agent.get_training_stats()
    print(f"最终成功率 (100回合): {stats['success_rate_100']:.2%}")
    print(f"最终平均奖励 (100回合): {stats['avg_reward_100']:.3f}")
    
    return agent


def train_and_compare(env: FrozenLakeWrapper, save_dir: str = 'models'):
    """
    训练并比较 Q-Learning 和 DQN
    
    Args:
        env: FrozenLake 环境
        save_dir: 模型保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    vis = Visualizer(save_dir='plots')
    results = {}
    
    # 训练 Q-Learning
    print("\n" + "=" * 60)
    print("【1/3】训练 Q-Learning")
    print("=" * 60)
    
    q_agent = train_q_learning(env)
    q_agent.save(os.path.join(save_dir, 'q_learning.npz'))
    results['Q-Learning'] = q_agent.episode_rewards
    
    # 打印 Q-Table 和策略
    print("\nQ-Learning 学习到的策略:")
    q_agent.print_policy(env.grid_size)
    
    # 训练 DQN (Nature DQN)
    print("\n" + "=" * 60)
    print("【2/3】训练 DQN (Nature DQN)")
    print("=" * 60)
    
    dqn_config = DQN_CONFIG.copy()
    dqn_config['use_double_dqn'] = False
    
    env_dqn = FrozenLakeWrapper(
        map_size=ENV_CONFIG['map_size'],
        is_slippery=ENV_CONFIG['is_slippery']
    )
    dqn_agent = train_dqn(env_dqn, config=dqn_config)
    dqn_agent.save(os.path.join(save_dir, 'dqn.pth'))
    results['DQN'] = dqn_agent.episode_rewards
    env_dqn.close()
    
    # 训练 Double DQN
    print("\n" + "=" * 60)
    print("【3/3】训练 Double DQN (DDQN)")
    print("=" * 60)
    
    ddqn_config = DQN_CONFIG.copy()
    ddqn_config['use_double_dqn'] = True
    
    env_ddqn = FrozenLakeWrapper(
        map_size=ENV_CONFIG['map_size'],
        is_slippery=ENV_CONFIG['is_slippery']
    )
    ddqn_agent = train_dqn(env_ddqn, config=ddqn_config)
    ddqn_agent.save(os.path.join(save_dir, 'ddqn.pth'))
    results['Double DQN'] = ddqn_agent.episode_rewards
    env_ddqn.close()
    
    # 绘制对比图
    print("\n" + "=" * 60)
    print("生成训练对比图表...")
    print("=" * 60)
    
    vis.plot_training_comparison(
        results,
        window_size=100,
        title='Q-Learning vs DQN vs DDQN 训练对比',
        save_name='training_comparison',
        show=True
    )
    
    # 分别绘制各算法的成功率
    for name, rewards in results.items():
        vis.plot_success_rate(
            rewards,
            window_size=100,
            title=f'{name} 成功率曲线',
            save_name=f'{name.lower().replace(" ", "_")}_success_rate',
            show=False
        )
    
    # 绘制 DQN 的损失曲线
    if hasattr(ddqn_agent, 'losses') and ddqn_agent.losses:
        vis.plot_loss(
            ddqn_agent.losses,
            window_size=100,
            title='Double DQN 训练损失',
            save_name='ddqn_loss',
            show=False
        )
    
    print("\n训练完成! 模型已保存到:", save_dir)
    print("图表已保存到: plots/")
    
    return q_agent, dqn_agent, ddqn_agent


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FrozenLake 强化学习训练')
    parser.add_argument('--algorithm', type=str, default='all',
                       choices=['q_learning', 'dqn', 'ddqn', 'all'],
                       help='训练算法: q_learning, dqn, ddqn, 或 all')
    parser.add_argument('--map_size', type=str, default='4x4',
                       choices=['4x4', '8x8'],
                       help='地图大小')
    parser.add_argument('--slippery', action='store_true', default=True,
                       help='启用随机模式 (Stochastic)')
    parser.add_argument('--no-slippery', action='store_false', dest='slippery',
                       help='使用确定性模式 (Deterministic)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='训练回合数')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='模型保存目录')
    
    args = parser.parse_args()
    
    # 更新配置
    ENV_CONFIG['map_size'] = args.map_size
    ENV_CONFIG['is_slippery'] = args.slippery
    
    if args.episodes:
        QLEARNING_CONFIG['num_episodes'] = args.episodes
        DQN_CONFIG['num_episodes'] = args.episodes
    
    # 创建环境
    env = FrozenLakeWrapper(
        map_size=args.map_size,
        is_slippery=args.slippery
    )
    
    print(f"\n环境配置:")
    print(f"  地图大小: {args.map_size}")
    print(f"  模式: {'Stochastic (随机)' if args.slippery else 'Deterministic (确定性)'}")
    env.print_map()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.algorithm == 'all':
        train_and_compare(env, args.save_dir)
    elif args.algorithm == 'q_learning':
        agent = train_q_learning(env)
        agent.save(os.path.join(args.save_dir, 'q_learning.npz'))
        agent.print_policy(env.grid_size)
    elif args.algorithm == 'dqn':
        DQN_CONFIG['use_double_dqn'] = False
        agent = train_dqn(env)
        agent.save(os.path.join(args.save_dir, 'dqn.pth'))
    elif args.algorithm == 'ddqn':
        DQN_CONFIG['use_double_dqn'] = True
        agent = train_dqn(env)
        agent.save(os.path.join(args.save_dir, 'ddqn.pth'))
    
    env.close()


if __name__ == '__main__':
    main()

