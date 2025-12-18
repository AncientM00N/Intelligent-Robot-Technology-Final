"""
评估脚本
用于评估训练好的模型性能
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple, List

from config import ENV_CONFIG
from environment.frozen_lake_wrapper import FrozenLakeWrapper
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from utils.visualization import Visualizer, plot_q_table_heatmap


def evaluate_agent(agent, env: FrozenLakeWrapper, 
                  num_episodes: int = 1000,
                  max_steps: int = 100,
                  verbose: bool = True) -> Tuple[float, float, List[float]]:
    """
    评估 Agent 性能
    
    Args:
        agent: 训练好的 Agent
        env: FrozenLake 环境
        num_episodes: 评估回合数
        max_steps: 每回合最大步数
        verbose: 是否打印进度
        
    Returns:
        (success_rate, avg_reward, rewards): 成功率，平均奖励，奖励列表
    """
    rewards = []
    successes = 0
    
    iterator = tqdm(range(num_episodes), desc="评估中") if verbose else range(num_episodes)
    
    for episode in iterator:
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # 使用贪心策略（不探索）
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        rewards.append(total_reward)
        if total_reward > 0:
            successes += 1
    
    success_rate = successes / num_episodes
    avg_reward = np.mean(rewards)
    
    return success_rate, avg_reward, rewards


def compare_models(env: FrozenLakeWrapper, 
                  model_dir: str = 'models',
                  num_episodes: int = 1000):
    """
    比较多个模型的性能
    
    Args:
        env: FrozenLake 环境
        model_dir: 模型目录
        num_episodes: 评估回合数
    """
    results = {}
    
    print(f"\n{'='*60}")
    print("模型性能评估")
    print(f"{'='*60}")
    print(f"评估回合数: {num_episodes}")
    print(f"环境: {env.grid_size}x{env.grid_size}, "
          f"{'Stochastic' if env.is_slippery else 'Deterministic'}")
    print(f"{'='*60}\n")
    
    # 评估 Q-Learning
    q_learning_path = os.path.join(model_dir, 'q_learning.npz')
    if os.path.exists(q_learning_path):
        print("评估 Q-Learning...")
        q_agent = QLearningAgent(env.n_states, env.n_actions)
        q_agent.load(q_learning_path)
        
        success_rate, avg_reward, rewards = evaluate_agent(q_agent, env, num_episodes)
        results['Q-Learning'] = {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'rewards': rewards
        }
        print(f"  成功率: {success_rate:.2%}")
        print(f"  平均奖励: {avg_reward:.4f}\n")
    
    # 评估 DQN
    dqn_path = os.path.join(model_dir, 'dqn.pth')
    if os.path.exists(dqn_path):
        print("评估 DQN...")
        dqn_agent = DQNAgent(env.n_states, env.n_actions)
        dqn_agent.load(dqn_path)
        
        success_rate, avg_reward, rewards = evaluate_agent(dqn_agent, env, num_episodes)
        results['DQN'] = {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'rewards': rewards
        }
        print(f"  成功率: {success_rate:.2%}")
        print(f"  平均奖励: {avg_reward:.4f}\n")
    
    # 评估 Double DQN
    ddqn_path = os.path.join(model_dir, 'ddqn.pth')
    if os.path.exists(ddqn_path):
        print("评估 Double DQN...")
        ddqn_agent = DQNAgent(env.n_states, env.n_actions)
        ddqn_agent.load(ddqn_path)
        
        success_rate, avg_reward, rewards = evaluate_agent(ddqn_agent, env, num_episodes)
        results['Double DQN'] = {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'rewards': rewards
        }
        print(f"  成功率: {success_rate:.2%}")
        print(f"  平均奖励: {avg_reward:.4f}\n")
    
    # 打印对比结果
    if results:
        print("=" * 60)
        print("评估结果对比")
        print("=" * 60)
        print(f"{'算法':<15} {'成功率':<15} {'平均奖励':<15}")
        print("-" * 45)
        
        for name, data in results.items():
            print(f"{name:<15} {data['success_rate']:.2%}{'':<9} {data['avg_reward']:.4f}")
        
        print("=" * 60)
    
    return results


def visualize_policy(agent, env: FrozenLakeWrapper, save_dir: str = 'plots'):
    """
    可视化 Agent 的策略
    
    Args:
        agent: 训练好的 Agent
        env: FrozenLake 环境
        save_dir: 图表保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取策略
    policy = agent.get_policy()
    
    # 动作符号
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    print("\n=== 学习到的策略 ===")
    print("+" + "---+" * env.grid_size)
    
    for row in range(env.grid_size):
        line = "|"
        for col in range(env.grid_size):
            state = row * env.grid_size + col
            cell_type = env.get_cell_type(state)
            
            if cell_type == 'G':
                symbol = ' G '
            elif cell_type == 'H':
                symbol = ' H '
            else:
                action = policy[state]
                symbol = f' {action_symbols[action]} '
            
            line += symbol + '|'
        print(line)
        print("+" + "---+" * env.grid_size)
    
    # 如果是 Q-Learning，绘制 Q-Table 热力图
    if hasattr(agent, 'q_table'):
        plot_q_table_heatmap(
            agent.q_table,
            grid_size=env.grid_size,
            title='Q-Learning Q-Table 热力图',
            save_path=os.path.join(save_dir, 'q_table_heatmap.png'),
            show=True
        )


def run_single_episode(agent, env: FrozenLakeWrapper, 
                       verbose: bool = True,
                       step_delay: float = 0.5):
    """
    运行单个回合并详细展示过程
    
    Args:
        agent: 训练好的 Agent
        env: FrozenLake 环境
        verbose: 是否打印详细信息
        step_delay: 每步延迟
    """
    import time
    
    state, _ = env.reset()
    
    print("\n" + "=" * 50)
    print("单回合演示")
    print("=" * 50)
    
    path = [env.state_to_coord(state)]
    actions = []
    
    env.print_map(state)
    
    for step in range(100):
        action = agent.select_action(state, training=False)
        actions.append(action)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        row, col = env.state_to_coord(next_state)
        path.append((row, col))
        
        if verbose:
            action_name = ['左', '下', '右', '上'][action]
            print(f"\n步骤 {step + 1}: 动作 = {action_name}")
            print(f"状态: {state} -> {next_state}")
            print(f"位置: {env.state_to_coord(state)} -> ({row}, {col})")
            print(f"奖励: {reward}")
            env.print_map(next_state)
        
        state = next_state
        
        if done:
            if reward > 0:
                print("\n🎉 成功到达目标!")
            else:
                print("\n💀 掉入冰窟窿!")
            break
        
        time.sleep(step_delay)
    
    # 打印路径摘要
    print("\n路径摘要:")
    print(" -> ".join([f"({r},{c})" for r, c in path]))
    print(f"总步数: {len(actions)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FrozenLake 模型评估')
    parser.add_argument('--model', type=str, default=None,
                       help='模型路径（如不指定则评估所有模型）')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型目录')
    parser.add_argument('--map_size', type=str, default='4x4',
                       choices=['4x4', '8x8'],
                       help='地图大小')
    parser.add_argument('--slippery', action='store_true', default=True,
                       help='启用随机模式')
    parser.add_argument('--no-slippery', action='store_false', dest='slippery',
                       help='使用确定性模式')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='评估回合数')
    parser.add_argument('--demo', action='store_true',
                       help='运行单回合演示')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化策略')
    
    args = parser.parse_args()
    
    # 创建环境
    env = FrozenLakeWrapper(
        map_size=args.map_size,
        is_slippery=args.slippery
    )
    
    print(f"\n环境配置:")
    print(f"  地图大小: {args.map_size}")
    print(f"  模式: {'Stochastic (随机)' if args.slippery else 'Deterministic (确定性)'}")
    
    if args.model:
        # 评估指定模型
        if args.model.endswith('.npz'):
            agent = QLearningAgent(env.n_states, env.n_actions)
        else:
            agent = DQNAgent(env.n_states, env.n_actions)
        
        agent.load(args.model)
        
        if args.demo:
            run_single_episode(agent, env)
        elif args.visualize:
            visualize_policy(agent, env)
        else:
            success_rate, avg_reward, _ = evaluate_agent(agent, env, args.episodes)
            print(f"\n评估结果:")
            print(f"  成功率: {success_rate:.2%}")
            print(f"  平均奖励: {avg_reward:.4f}")
    else:
        # 评估所有模型
        results = compare_models(env, args.model_dir, args.episodes)
        
        if args.visualize and os.path.exists(os.path.join(args.model_dir, 'q_learning.npz')):
            q_agent = QLearningAgent(env.n_states, env.n_actions)
            q_agent.load(os.path.join(args.model_dir, 'q_learning.npz'))
            visualize_policy(q_agent, env)
    
    env.close()


if __name__ == '__main__':
    main()

