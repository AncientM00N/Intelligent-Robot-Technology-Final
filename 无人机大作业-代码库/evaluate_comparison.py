"""
模型评估与对比脚本
用于加载训练好的模型，进行测试评估并生成对比报告
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
import json

from config import BENCHMARK_CONFIG
from environment.frozen_lake_wrapper import FrozenLakeWrapper
from agents.q_learning_agent import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.dqn_agent import DQNAgent


def evaluate_agent(agent, 
                   env: FrozenLakeWrapper, 
                   n_episodes: int = 100,
                   max_steps: int = 100,
                   verbose: bool = True) -> Dict:
    """
    评估单个 Agent
    
    Args:
        agent: 要评估的智能体
        env: 环境
        n_episodes: 测试回合数
        max_steps: 每回合最大步数
        verbose: 是否打印进度
        
    Returns:
        评估结果字典
    """
    success_count = 0
    rewards = []
    steps = []
    trajectories = []
    
    iterator = tqdm(range(n_episodes), desc="评估中") if verbose else range(n_episodes)
    
    for _ in iterator:
        state, _ = env.reset()
        total_reward = 0
        trajectory = [state]
        
        for step in range(max_steps):
            # 贪心策略（不探索）
            action = agent.select_action(state, training=False)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            state = next_state
            trajectory.append(state)
            
            if done:
                break
        
        rewards.append(total_reward)
        steps.append(step + 1)
        trajectories.append(trajectory)
        
        if total_reward > 0:
            success_count += 1
    
    return {
        'success_count': success_count,
        'success_rate': success_count / n_episodes,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_steps': np.mean(steps),
        'std_steps': np.std(steps),
        'min_steps': np.min(steps),
        'max_steps': np.max(steps),
        'rewards': rewards,
        'steps': steps,
        'trajectories': trajectories,
    }


def load_agents(model_dir: str, map_size: str, env: FrozenLakeWrapper) -> Dict:
    """
    加载所有算法的模型
    
    Args:
        model_dir: 模型目录
        map_size: 地图大小
        env: 环境实例（用于获取状态和动作空间）
        
    Returns:
        {算法名: agent对象} 字典
    """
    agents = {}
    
    # Q-Learning
    q_learning_path = os.path.join(model_dir, 'q_learning.npz')
    if os.path.exists(q_learning_path):
        print(f"加载 Q-Learning 模型: {q_learning_path}")
        agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
        agent.load(q_learning_path)
        agents['Q-Learning'] = agent
    else:
        print(f"警告: 未找到 Q-Learning 模型: {q_learning_path}")
    
    # SARSA
    sarsa_path = os.path.join(model_dir, 'sarsa.npz')
    if os.path.exists(sarsa_path):
        print(f"加载 SARSA 模型: {sarsa_path}")
        agent = SARSAAgent(n_states=env.n_states, n_actions=env.n_actions)
        agent.load(sarsa_path)
        agents['SARSA'] = agent
    else:
        print(f"警告: 未找到 SARSA 模型: {sarsa_path}")
    
    # DQN
    dqn_path = os.path.join(model_dir, 'dqn.pth')
    if os.path.exists(dqn_path):
        print(f"加载 DQN 模型: {dqn_path}")
        agent = DQNAgent(n_states=env.n_states, n_actions=env.n_actions, use_double_dqn=False)
        agent.load(dqn_path)
        agents['DQN'] = agent
    else:
        print(f"警告: 未找到 DQN 模型: {dqn_path}")
    
    # DDQN
    ddqn_path = os.path.join(model_dir, 'ddqn.pth')
    if os.path.exists(ddqn_path):
        print(f"加载 DDQN 模型: {ddqn_path}")
        agent = DQNAgent(n_states=env.n_states, n_actions=env.n_actions, use_double_dqn=True)
        agent.load(ddqn_path)
        agents['DDQN'] = agent
    else:
        print(f"警告: 未找到 DDQN 模型: {ddqn_path}")
    
    print(f"\n成功加载 {len(agents)} 个模型\n")
    return agents


def generate_comparison_report(results: Dict, 
                               map_size: str, 
                               mode: str,
                               n_episodes: int) -> str:
    """
    生成 Markdown 格式的对比报告
    
    Args:
        results: {算法名: 评估结果} 字典
        map_size: 地图大小
        mode: 环境模式
        n_episodes: 测试回合数
        
    Returns:
        Markdown 格式报告文本
    """
    report_lines = [
        "# 算法评估对比报告\n",
        f"**地图大小**: {map_size}",
        f"**环境模式**: {mode}",
        f"**测试回合数**: {n_episodes}\n",
        "---\n",
        "## 📊 性能对比\n",
        "| 算法 | 成功率 | 平均奖励 | 平均步数 | 最少步数 | 最多步数 |",
        "|------|--------|----------|----------|----------|----------|",
    ]
    
    # 按成功率排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['success_rate'], reverse=True)
    
    for algo_name, result in sorted_results:
        report_lines.append(
            f"| {algo_name} | "
            f"{result['success_rate']:.2%} | "
            f"{result['avg_reward']:.3f} ± {result['std_reward']:.3f} | "
            f"{result['avg_steps']:.1f} ± {result['std_steps']:.1f} | "
            f"{result['min_steps']} | "
            f"{result['max_steps']} |"
        )
    
    report_lines.append("\n---\n")
    report_lines.append("## 🏆 最优算法\n")
    
    # 找出最优算法
    best_success = max(results.items(), key=lambda x: x[1]['success_rate'])
    best_reward = max(results.items(), key=lambda x: x[1]['avg_reward'])
    best_steps = min(results.items(), key=lambda x: x[1]['avg_steps'])
    
    report_lines.append(f"- **最高成功率**: {best_success[0]} ({best_success[1]['success_rate']:.2%})")
    report_lines.append(f"- **最高平均奖励**: {best_reward[0]} ({best_reward[1]['avg_reward']:.3f})")
    report_lines.append(f"- **最少平均步数**: {best_steps[0]} ({best_steps[1]['avg_steps']:.1f} 步)")
    
    report_lines.append("\n---\n")
    report_lines.append("## 📈 详细统计\n")
    
    for algo_name, result in sorted_results:
        report_lines.append(f"\n### {algo_name}\n")
        report_lines.append(f"- 成功次数: {result['success_count']}/{n_episodes}")
        report_lines.append(f"- 成功率: {result['success_rate']:.2%}")
        report_lines.append(f"- 平均奖励: {result['avg_reward']:.3f} (标准差: {result['std_reward']:.3f})")
        report_lines.append(f"- 平均步数: {result['avg_steps']:.1f} (标准差: {result['std_steps']:.1f})")
        report_lines.append(f"- 步数范围: [{result['min_steps']}, {result['max_steps']}]")
    
    return "\n".join(report_lines)


def save_results(results: Dict, 
                 save_path: str):
    """
    保存评估结果为 JSON 格式
    
    Args:
        results: 评估结果字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换 numpy 类型为 Python 原生类型
    json_results = {}
    for algo_name, result in results.items():
        json_results[algo_name] = {
            'success_count': int(result['success_count']),
            'success_rate': float(result['success_rate']),
            'avg_reward': float(result['avg_reward']),
            'std_reward': float(result['std_reward']),
            'avg_steps': float(result['avg_steps']),
            'std_steps': float(result['std_steps']),
            'min_steps': int(result['min_steps']),
            'max_steps': int(result['max_steps']),
            'rewards': [float(r) for r in result['rewards']],
            'steps': [int(s) for s in result['steps']],
        }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"评估结果已保存: {save_path}")


def visualize_sample_trajectories(results: Dict,
                                  env: FrozenLakeWrapper,
                                  n_samples: int = 3):
    """
    可视化样例轨迹
    
    Args:
        results: 评估结果
        env: 环境
        n_samples: 展示样例数
    """
    print("\n" + "="*80)
    print("样例轨迹可视化")
    print("="*80)
    
    for algo_name, result in results.items():
        print(f"\n### {algo_name} ###")
        
        # 找出成功的轨迹
        successful_indices = [i for i, r in enumerate(result['rewards']) if r > 0]
        
        if not successful_indices:
            print("  没有成功的轨迹")
            continue
        
        # 随机选择几个成功轨迹
        sample_indices = np.random.choice(successful_indices, 
                                         min(n_samples, len(successful_indices)), 
                                         replace=False)
        
        for idx in sample_indices:
            trajectory = result['trajectories'][idx]
            steps_taken = result['steps'][idx]
            
            print(f"\n  样例 {idx+1} (步数: {steps_taken}):")
            print(f"  轨迹: ", end="")
            
            for state in trajectory:
                row, col = env.state_to_coord(state)
                print(f"({row},{col})", end=" -> " if state != trajectory[-1] else "")
            
            print(" [到达目标]")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='模型评估与对比脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估 8x8 随机模式的模型
  python evaluate_comparison.py --map_size 8x8 --mode stochastic
  
  # 评估 4x4 确定性模式的模型（200次测试）
  python evaluate_comparison.py --map_size 4x4 --mode deterministic --episodes 200
  
  # 指定模型目录
  python evaluate_comparison.py --map_size 8x8 --model_dir models/my_benchmark
        """
    )
    
    parser.add_argument('--map_size', type=str, default='8x8',
                       choices=['4x4', '8x8'],
                       help='地图大小 (默认: 8x8)')
    parser.add_argument('--mode', type=str, default='stochastic',
                       choices=['deterministic', 'stochastic'],
                       help='环境模式 (默认: stochastic)')
    parser.add_argument('--model_dir', type=str,
                       help='模型目录（默认使用 Benchmark 目录）')
    parser.add_argument('--episodes', type=int,
                       help='测试回合数（默认使用配置文件）')
    parser.add_argument('--max_steps', type=int,
                       help='每回合最大步数（默认使用配置文件）')
    parser.add_argument('--save_dir', type=str,
                       help='结果保存目录（默认使用 Benchmark 目录）')
    parser.add_argument('--show_trajectories', action='store_true',
                       help='显示样例轨迹')
    
    args = parser.parse_args()
    
    # 设置路径
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = BENCHMARK_CONFIG['model_dir_template'].format(
            map_size=args.map_size, mode=args.mode
        )
    
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = BENCHMARK_CONFIG['result_dir_template'].format(
            map_size=args.map_size, mode=args.mode
        )
    
    # 评估参数
    n_episodes = args.episodes or BENCHMARK_CONFIG['eval_episodes']
    max_steps = args.max_steps or BENCHMARK_CONFIG['eval_max_steps']
    
    print("\n" + "="*80)
    print("模型评估与对比")
    print("="*80)
    print(f"地图大小: {args.map_size}")
    print(f"环境模式: {args.mode}")
    print(f"模型目录: {model_dir}")
    print(f"测试回合数: {n_episodes}")
    print(f"最大步数: {max_steps}")
    print("="*80)
    
    # 创建环境
    is_slippery = (args.mode == 'stochastic')
    env = FrozenLakeWrapper(map_size=args.map_size, is_slippery=is_slippery)
    
    print(f"\n环境信息:")
    env.print_map()
    
    # 加载模型
    print(f"\n{'='*80}")
    print("加载模型...")
    print(f"{'='*80}")
    
    agents = load_agents(model_dir, args.map_size, env)
    
    if not agents:
        print("错误: 未能加载任何模型！")
        return
    
    # 评估所有模型
    print(f"\n{'='*80}")
    print(f"开始评估 ({n_episodes} 回合)...")
    print(f"{'='*80}\n")
    
    results = {}
    for algo_name, agent in agents.items():
        print(f"\n--- 评估 {algo_name} ---")
        result = evaluate_agent(agent, env, n_episodes, max_steps, verbose=True)
        results[algo_name] = result
        
        print(f"成功率: {result['success_rate']:.2%}")
        print(f"平均奖励: {result['avg_reward']:.3f}")
        print(f"平均步数: {result['avg_steps']:.1f}")
    
    env.close()
    
    # 生成报告
    print(f"\n{'='*80}")
    print("生成对比报告...")
    print(f"{'='*80}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    report = generate_comparison_report(results, args.map_size, args.mode, n_episodes)
    
    # 保存报告
    report_path = os.path.join(save_dir, 'evaluation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"对比报告已保存: {report_path}")
    
    # 保存 JSON 结果
    json_path = os.path.join(save_dir, 'evaluation_results.json')
    save_results(results, json_path)
    
    # 打印报告
    print("\n" + report)
    
    # 可视化轨迹
    if args.show_trajectories:
        visualize_sample_trajectories(results, env, n_samples=3)
    
    print(f"\n{'='*80}")
    print("评估完成!")
    print(f"{'='*80}")
    print(f"结果保存在: {save_dir}")


if __name__ == '__main__':
    main()
