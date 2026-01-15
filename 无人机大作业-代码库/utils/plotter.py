"""
算法对比可视化工具
用于生成 4 种强化学习算法的训练对比图表
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple
import os
import pandas as pd


class BenchmarkPlotter:
    """
    Benchmark 对比绘图工具
    
    支持功能：
    1. 训练曲线对比（单图/网格）
    2. 成功率曲线对比
    3. 损失函数对比（DQN vs DDQN）
    4. 收敛分析表格
    """
    
    def __init__(self, algorithm_colors: Optional[Dict[str, str]] = None):
        """
        初始化绘图工具
        
        Args:
            algorithm_colors: 算法名称到颜色的映射字典
        """
        self.algorithm_colors = algorithm_colors or {
            'Q-Learning': '#1f77b4',  # 蓝色
            'SARSA': '#ff7f0e',       # 橙色
            'DQN': '#2ca02c',         # 绿色
            'DDQN': '#d62728',        # 红色
        }
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def moving_average(self, data: List[float], window_size: int = 100) -> np.ndarray:
        """
        计算移动平均
        
        Args:
            data: 原始数据
            window_size: 窗口大小
            
        Returns:
            平滑后的数据
        """
        if len(data) < window_size:
            window_size = len(data)
        
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    def plot_training_comparison(self, 
                                 results: Dict[str, List[float]],
                                 save_path: str,
                                 window_size: int = 100,
                                 style: str = 'single',
                                 title: str = '算法训练对比',
                                 show: bool = False):
        """
        绘制训练曲线对比图
        
        Args:
            results: 字典 {算法名: 奖励列表}
            save_path: 保存路径
            window_size: 移动平均窗口大小
            style: 'single' 或 'grid'
            title: 图表标题
            show: 是否显示图表
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if style == 'single':
            self._plot_single_comparison(results, save_path, window_size, title, show)
        elif style == 'grid':
            self._plot_grid_comparison(results, save_path, window_size, title, show)
        else:
            raise ValueError(f"未知的样式: {style}，请使用 'single' 或 'grid'")
    
    def _plot_single_comparison(self, 
                                results: Dict[str, List[float]],
                                save_path: str,
                                window_size: int,
                                title: str,
                                show: bool):
        """单图对比"""
        plt.figure(figsize=(12, 6))
        
        for algo_name, rewards in results.items():
            if len(rewards) < window_size:
                print(f"警告: {algo_name} 数据点不足 {window_size}，跳过")
                continue
            
            smoothed = self.moving_average(rewards, window_size)
            episodes = np.arange(window_size, len(rewards) + 1)
            
            color = self.algorithm_colors.get(algo_name, None)
            plt.plot(episodes, smoothed, label=algo_name, color=color, linewidth=2)
        
        plt.xlabel('训练回合数', fontsize=12)
        plt.ylabel(f'平均奖励 (窗口={window_size})', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"单图对比已保存: {save_path}")
        
        if show:
            plt.show()
        plt.close()
    
    def _plot_grid_comparison(self, 
                             results: Dict[str, List[float]],
                             save_path: str,
                             window_size: int,
                             title: str,
                             show: bool):
        """2x2 网格对比"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        algo_names = list(results.keys())
        for idx, algo_name in enumerate(algo_names):
            if idx >= 4:
                break
            
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            rewards = results[algo_name]
            if len(rewards) < window_size:
                ax.text(0.5, 0.5, f'{algo_name}\n数据不足', 
                       ha='center', va='center', fontsize=12)
                ax.set_title(algo_name)
                continue
            
            smoothed = self.moving_average(rewards, window_size)
            episodes = np.arange(window_size, len(rewards) + 1)
            
            color = self.algorithm_colors.get(algo_name, None)
            ax.plot(episodes, smoothed, color=color, linewidth=2)
            ax.set_xlabel('训练回合数', fontsize=10)
            ax.set_ylabel(f'平均奖励 (窗口={window_size})', fontsize=10)
            ax.set_title(algo_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"网格对比已保存: {save_path}")
        
        if show:
            plt.show()
        plt.close()
    
    def plot_success_rate_comparison(self,
                                     results: Dict[str, List[float]],
                                     save_path: str,
                                     window_size: int = 100,
                                     title: str = '算法成功率对比',
                                     show: bool = False):
        """
        绘制成功率曲线对比
        
        Args:
            results: 字典 {算法名: 奖励列表}
            save_path: 保存路径
            window_size: 移动平均窗口
            title: 图表标题
            show: 是否显示
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        for algo_name, rewards in results.items():
            # 计算成功率（reward > 0 表示成功）
            success_flags = [1 if r > 0 else 0 for r in rewards]
            
            if len(success_flags) < window_size:
                print(f"警告: {algo_name} 数据点不足 {window_size}，跳过")
                continue
            
            smoothed_success = self.moving_average(success_flags, window_size)
            episodes = np.arange(window_size, len(rewards) + 1)
            
            color = self.algorithm_colors.get(algo_name, None)
            plt.plot(episodes, smoothed_success * 100, label=algo_name, 
                    color=color, linewidth=2)
        
        plt.xlabel('训练回合数', fontsize=12)
        plt.ylabel(f'成功率 % (窗口={window_size})', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"成功率对比已保存: {save_path}")
        
        if show:
            plt.show()
        plt.close()
    
    def plot_loss_comparison(self,
                            dqn_losses: List[float],
                            ddqn_losses: List[float],
                            save_path: str,
                            window_size: int = 100,
                            title: str = 'DQN vs DDQN 损失对比',
                            show: bool = False):
        """
        绘制 DQN 和 DDQN 的损失函数对比
        
        Args:
            dqn_losses: DQN 损失列表
            ddqn_losses: DDQN 损失列表
            save_path: 保存路径
            window_size: 移动平均窗口
            title: 图表标题
            show: 是否显示
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        if len(dqn_losses) >= window_size:
            smoothed_dqn = self.moving_average(dqn_losses, window_size)
            steps_dqn = np.arange(window_size, len(dqn_losses) + 1)
            plt.plot(steps_dqn, smoothed_dqn, label='DQN', 
                    color=self.algorithm_colors['DQN'], linewidth=2)
        
        if len(ddqn_losses) >= window_size:
            smoothed_ddqn = self.moving_average(ddqn_losses, window_size)
            steps_ddqn = np.arange(window_size, len(ddqn_losses) + 1)
            plt.plot(steps_ddqn, smoothed_ddqn, label='DDQN', 
                    color=self.algorithm_colors['DDQN'], linewidth=2)
        
        plt.xlabel('训练步数', fontsize=12)
        plt.ylabel(f'平均损失 (窗口={window_size})', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"损失对比已保存: {save_path}")
        
        if show:
            plt.show()
        plt.close()
    
    def generate_convergence_table(self,
                                   results: Dict[str, List[float]],
                                   threshold: float = 0.7,
                                   window_size: int = 100) -> pd.DataFrame:
        """
        生成收敛分析表格
        
        Args:
            results: 字典 {算法名: 奖励列表}
            threshold: 成功率阈值
            window_size: 移动平均窗口
            
        Returns:
            pandas DataFrame 包含收敛统计
        """
        convergence_data = []
        
        for algo_name, rewards in results.items():
            # 计算成功率
            success_flags = [1 if r > 0 else 0 for r in rewards]
            
            if len(success_flags) < window_size:
                convergence_data.append({
                    '算法': algo_name,
                    '收敛回合数': 'N/A',
                    '最终成功率': 'N/A',
                    '最高成功率': 'N/A',
                    '平均奖励': 'N/A',
                })
                continue
            
            # 计算移动平均成功率
            smoothed_success = self.moving_average(success_flags, window_size)
            
            # 找到首次达到阈值的回合
            convergence_episode = None
            for i, success_rate in enumerate(smoothed_success):
                if success_rate >= threshold:
                    convergence_episode = i + window_size
                    break
            
            # 统计信息
            final_success_rate = smoothed_success[-1] if len(smoothed_success) > 0 else 0
            max_success_rate = np.max(smoothed_success) if len(smoothed_success) > 0 else 0
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            
            convergence_data.append({
                '算法': algo_name,
                '收敛回合数': convergence_episode if convergence_episode else f'>{len(rewards)}',
                '最终成功率': f'{final_success_rate:.2%}',
                '最高成功率': f'{max_success_rate:.2%}',
                '平均奖励': f'{avg_reward:.3f}',
            })
        
        df = pd.DataFrame(convergence_data)
        return df
    
    def save_convergence_table(self,
                               results: Dict[str, List[float]],
                               save_path: str,
                               threshold: float = 0.7,
                               window_size: int = 100):
        """
        保存收敛分析表格为 Markdown 格式
        
        Args:
            results: 字典 {算法名: 奖励列表}
            save_path: 保存路径
            threshold: 成功率阈值
            window_size: 移动平均窗口
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df = self.generate_convergence_table(results, threshold, window_size)
        
        # 保存为 Markdown
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"# 算法收敛分析表\n\n")
            f.write(f"**成功率阈值**: {threshold:.0%}\n\n")
            f.write(f"**移动平均窗口**: {window_size}\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n")
        
        print(f"收敛表格已保存: {save_path}")
        return df


# 测试代码
if __name__ == '__main__':
    # 模拟数据
    np.random.seed(42)
    
    results = {
        'Q-Learning': list(np.random.randn(10000) * 0.1 + np.linspace(-0.5, 0.6, 10000)),
        'SARSA': list(np.random.randn(10000) * 0.1 + np.linspace(-0.5, 0.55, 10000)),
        'DQN': list(np.random.randn(5000) * 0.15 + np.linspace(-0.3, 0.7, 5000)),
        'DDQN': list(np.random.randn(5000) * 0.15 + np.linspace(-0.2, 0.8, 5000)),
    }
    
    plotter = BenchmarkPlotter()
    
    # 测试训练曲线
    os.makedirs('test_plots', exist_ok=True)
    plotter.plot_training_comparison(results, 'test_plots/comparison_single.png', style='single')
    plotter.plot_training_comparison(results, 'test_plots/comparison_grid.png', style='grid')
    
    # 测试成功率
    plotter.plot_success_rate_comparison(results, 'test_plots/success_rate.png')
    
    # 测试损失对比
    dqn_losses = list(np.random.randn(5000) * 0.1 + np.linspace(1.0, 0.2, 5000))
    ddqn_losses = list(np.random.randn(5000) * 0.1 + np.linspace(1.0, 0.15, 5000))
    plotter.plot_loss_comparison(dqn_losses, ddqn_losses, 'test_plots/loss_comparison.png')
    
    # 测试收敛表格
    plotter.save_convergence_table(results, 'test_plots/convergence_table.md')
    
    print("测试完成！")
