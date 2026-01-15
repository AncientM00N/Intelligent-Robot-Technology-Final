"""
从已有的训练数据生成对比图表
用于在训练完成后单独生成或重新生成图表
"""

import os
import argparse
import json
from utils.plotter import BenchmarkPlotter
from config import BENCHMARK_CONFIG


def load_training_data(result_dir: str) -> dict:
    """
    从结果目录加载所有算法的训练数据
    
    Args:
        result_dir: 结果目录路径
        
    Returns:
        {算法名: 奖励列表} 字典
    """
    results = {}
    losses = {}
    
    algorithms = {
        'Q-Learning': 'Q-Learning_training_data.json',
        'SARSA': 'SARSA_training_data.json',
        'DQN': 'DQN_training_data.json',
        'DDQN': 'DDQN_training_data.json',
    }
    
    for algo_name, filename in algorithms.items():
        filepath = os.path.join(result_dir, filename)
        if os.path.exists(filepath):
            print(f"加载 {algo_name} 训练数据: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 提取奖励数据
            episodes = data.get('episodes', [])
            rewards = [ep['total_reward'] for ep in episodes]
            results[algo_name] = rewards
            
            # 提取损失数据（仅DQN/DDQN）
            if 'DQN' in algo_name or 'DDQN' in algo_name:
                episode_losses = [ep.get('loss') for ep in episodes if ep.get('loss') is not None]
                if episode_losses:
                    losses[algo_name] = episode_losses
            
            print(f"  - 加载了 {len(rewards)} 回合的数据")
        else:
            print(f"警告: 未找到 {algo_name} 数据文件: {filepath}")
    
    return results, losses


def generate_all_plots(map_size: str, mode: str):
    """
    生成所有对比图表
    
    Args:
        map_size: 地图大小 ('4x4' 或 '8x8')
        mode: 环境模式 ('deterministic' 或 'stochastic')
    """
    # 设置路径
    result_dir = BENCHMARK_CONFIG['result_dir_template'].format(
        map_size=map_size, mode=mode
    )
    plot_dir = BENCHMARK_CONFIG['plot_dir_template'].format(
        map_size=map_size, mode=mode
    )
    
    print("\n" + "="*80)
    print(f"生成 {map_size} - {mode} 的对比图表")
    print("="*80)
    print(f"结果目录: {result_dir}")
    print(f"图表目录: {plot_dir}")
    print("="*80 + "\n")
    
    # 创建图表目录
    os.makedirs(plot_dir, exist_ok=True)
    
    # 加载训练数据
    print("正在加载训练数据...\n")
    results, losses = load_training_data(result_dir)
    
    if not results:
        print("错误: 没有找到任何训练数据！")
        print(f"请确保 {result_dir} 目录下有训练数据文件。")
        return
    
    print(f"\n成功加载 {len(results)} 个算法的数据\n")
    
    # 创建绘图工具
    plotter = BenchmarkPlotter(BENCHMARK_CONFIG['algorithm_colors'])
    window_size = BENCHMARK_CONFIG['plot_window_size']
    
    # 1. 单图训练曲线对比
    print("="*80)
    print("【1/5】生成单图训练曲线对比...")
    print("="*80)
    plotter.plot_training_comparison(
        results,
        os.path.join(plot_dir, 'training_comparison_single.png'),
        window_size=window_size,
        style='single',
        title=f'算法训练对比 ({map_size} - {mode})'
    )
    
    # 2. 网格训练曲线对比
    print("\n" + "="*80)
    print("【2/5】生成网格训练曲线对比...")
    print("="*80)
    plotter.plot_training_comparison(
        results,
        os.path.join(plot_dir, 'training_comparison_grid.png'),
        window_size=window_size,
        style='grid',
        title=f'算法训练对比 ({map_size} - {mode})'
    )
    
    # 3. 成功率对比
    print("\n" + "="*80)
    print("【3/5】生成成功率对比...")
    print("="*80)
    plotter.plot_success_rate_comparison(
        results,
        os.path.join(plot_dir, 'success_rate_comparison.png'),
        window_size=window_size,
        title=f'算法成功率对比 ({map_size} - {mode})'
    )
    
    # 4. 损失对比（仅DQN和DDQN）
    if 'DQN' in losses and 'DDQN' in losses:
        print("\n" + "="*80)
        print("【4/5】生成损失函数对比...")
        print("="*80)
        plotter.plot_loss_comparison(
            losses['DQN'],
            losses['DDQN'],
            os.path.join(plot_dir, 'loss_comparison.png'),
            window_size=window_size,
            title=f'DQN vs DDQN 损失对比 ({map_size} - {mode})'
        )
    else:
        print("\n跳过损失对比（缺少 DQN 或 DDQN 损失数据）")
    
    # 5. 收敛表格
    print("\n" + "="*80)
    print("【5/5】生成收敛分析表格...")
    print("="*80)
    plotter.save_convergence_table(
        results,
        os.path.join(result_dir, 'convergence_table.md'),
        threshold=BENCHMARK_CONFIG['success_threshold'],
        window_size=window_size
    )
    
    print("\n" + "="*80)
    print("[SUCCESS] 所有图表生成完成！")
    print("="*80)
    print(f"\n图表保存在: {plot_dir}")
    print(f"收敛表格: {os.path.join(result_dir, 'convergence_table.md')}")
    print("\n生成的图表文件:")
    print(f"  1. training_comparison_single.png   - 单图对比")
    print(f"  2. training_comparison_grid.png     - 网格对比")
    print(f"  3. success_rate_comparison.png      - 成功率对比")
    if 'DQN' in losses and 'DDQN' in losses:
        print(f"  4. loss_comparison.png              - 损失对比")
    print(f"  5. {os.path.basename(result_dir)}/convergence_table.md - 收敛表格")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='从训练数据生成对比图表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 为 8x8 随机模式生成图表
  python generate_plots.py --map_size 8x8 --mode stochastic
  
  # 为 4x4 确定性模式生成图表
  python generate_plots.py --map_size 4x4 --mode deterministic
  
  # 自定义结果目录
  python generate_plots.py --map_size 8x8 --result_dir results/my_experiment
        """
    )
    
    parser.add_argument('--map_size', type=str, default='8x8',
                       choices=['4x4', '8x8'],
                       help='地图大小 (默认: 8x8)')
    parser.add_argument('--mode', type=str, default='stochastic',
                       choices=['deterministic', 'stochastic'],
                       help='环境模式 (默认: stochastic)')
    parser.add_argument('--result_dir', type=str,
                       help='自定义结果目录（默认使用 Benchmark 目录）')
    parser.add_argument('--plot_dir', type=str,
                       help='自定义图表保存目录（默认使用 Benchmark 目录）')
    
    args = parser.parse_args()
    
    # 如果指定了自定义目录，使用自定义目录
    if args.result_dir or args.plot_dir:
        if args.result_dir:
            BENCHMARK_CONFIG['result_dir_template'] = args.result_dir
        if args.plot_dir:
            BENCHMARK_CONFIG['plot_dir_template'] = args.plot_dir
    
    # 生成图表
    generate_all_plots(args.map_size, args.mode)


if __name__ == '__main__':
    main()
