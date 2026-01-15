"""
可视化工具
用于绘制训练曲线、展示网格动画等
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
import os
import time


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """
    可视化工具类
    """
    
    def __init__(self, save_dir: str = './plots'):
        """
        初始化可视化器
        
        Args:
            save_dir: 图表保存目录
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def plot_rewards(self, rewards: List[float], 
                    window_size: int = 100,
                    title: str = '训练奖励曲线',
                    save_name: Optional[str] = None,
                    show: bool = True):
        """
        绘制奖励曲线
        
        Args:
            rewards: 每回合奖励列表
            window_size: 滑动平均窗口大小
            title: 图表标题
            save_name: 保存文件名（不含扩展名）
            show: 是否显示图表
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        episodes = range(1, len(rewards) + 1)
        
        # 绘制原始奖励
        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='原始奖励')
        
        # 计算滑动平均
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(window_size, len(rewards) + 1), moving_avg, 
                   color='red', linewidth=2, label=f'{window_size}回合滑动平均')
        
        ax.set_xlabel('回合数')
        ax.set_ylabel('奖励')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            filepath = os.path.join(self.save_dir, f'{save_name}.png')
            plt.savefig(filepath, dpi=150)
            print(f"图表已保存: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_training_comparison(self, 
                                 rewards_dict: dict,
                                 window_size: int = 100,
                                 title: str = '算法对比',
                                 save_name: Optional[str] = None,
                                 show: bool = True):
        """
        对比多个算法的训练曲线
        
        Args:
            rewards_dict: {'算法名': rewards_list} 字典
            window_size: 滑动平均窗口大小
            title: 图表标题
            save_name: 保存文件名
            show: 是否显示
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (name, rewards) in enumerate(rewards_dict.items()):
            color = colors[i % len(colors)]
            episodes = range(1, len(rewards) + 1)
            
            # 绘制原始奖励（透明）
            ax.plot(episodes, rewards, alpha=0.2, color=color)
            
            # 绘制滑动平均
            if len(rewards) >= window_size:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ax.plot(range(window_size, len(rewards) + 1), moving_avg,
                       color=color, linewidth=2, label=f'{name}')
        
        ax.set_xlabel('回合数')
        ax.set_ylabel('奖励')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            filepath = os.path.join(self.save_dir, f'{save_name}.png')
            plt.savefig(filepath, dpi=150)
            print(f"图表已保存: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_success_rate(self, rewards: List[float],
                         window_size: int = 100,
                         title: str = '成功率曲线',
                         save_name: Optional[str] = None,
                         show: bool = True):
        """
        绘制成功率曲线（基于奖励>0判断成功）
        
        Args:
            rewards: 奖励列表
            window_size: 计算成功率的窗口大小
            title: 图表标题
            save_name: 保存文件名
            show: 是否显示
        """
        # 计算每个窗口的成功率
        success = [1 if r > 0 else 0 for r in rewards]
        
        if len(success) < window_size:
            print(f"数据不足，需要至少 {window_size} 个回合")
            return
        
        success_rates = []
        for i in range(window_size, len(success) + 1):
            rate = np.mean(success[i-window_size:i])
            success_rates.append(rate)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        episodes = range(window_size, len(success) + 1)
        ax.plot(episodes, success_rates, color='green', linewidth=2)
        ax.fill_between(episodes, success_rates, alpha=0.3, color='green')
        
        ax.set_xlabel('回合数')
        ax.set_ylabel('成功率')
        ax.set_title(f'{title} (窗口={window_size})')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            filepath = os.path.join(self.save_dir, f'{save_name}.png')
            plt.savefig(filepath, dpi=150)
            print(f"图表已保存: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_loss(self, losses: List[float],
                 window_size: int = 100,
                 title: str = 'DQN 损失曲线',
                 save_name: Optional[str] = None,
                 show: bool = True):
        """
        绘制损失曲线（用于 DQN）
        
        Args:
            losses: 损失列表
            window_size: 滑动平均窗口大小
            title: 图表标题
            save_name: 保存文件名
            show: 是否显示
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = range(1, len(losses) + 1)
        ax.plot(steps, losses, alpha=0.3, color='orange', label='原始损失')
        
        if len(losses) >= window_size:
            moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(window_size, len(losses) + 1), moving_avg,
                   color='red', linewidth=2, label=f'{window_size}步滑动平均')
        
        ax.set_xlabel('训练步数')
        ax.set_ylabel('损失')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            filepath = os.path.join(self.save_dir, f'{save_name}.png')
            plt.savefig(filepath, dpi=150)
            print(f"图表已保存: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()


class GridVisualizer:
    """
    网格可视化器
    用于实时显示无人机在网格中的位置
    """
    
    # 地图符号
    CELL_SYMBOLS = {
        'S': '起',
        'F': '·',
        'H': '洞',
        'G': '终'
    }
    
    # 动作符号
    ACTION_SYMBOLS = {
        0: '←',
        1: '↓',
        2: '→',
        3: '↑'
    }
    
    def __init__(self, grid_size: int = 4, desc: Optional[np.ndarray] = None):
        """
        初始化网格可视化器
        
        Args:
            grid_size: 网格大小
            desc: 地图描述数组
        """
        self.grid_size = grid_size
        self.desc = desc
        self.path_history: List[Tuple[int, int]] = []
    
    def set_map(self, desc: np.ndarray):
        """设置地图"""
        self.desc = desc
        self.grid_size = len(desc)
    
    def clear_path(self):
        """清除路径历史"""
        self.path_history.clear()
    
    def add_position(self, row: int, col: int):
        """添加位置到路径历史"""
        self.path_history.append((row, col))
    
    def print_grid(self, current_pos: Tuple[int, int], 
                   show_path: bool = True,
                   action: Optional[int] = None,
                   clear_screen: bool = False):
        """
        打印网格状态
        
        Args:
            current_pos: 当前位置 (row, col)
            show_path: 是否显示历史路径
            action: 当前执行的动作
            clear_screen: 是否清屏
        """
        if clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')
        
        row, col = current_pos
        
        print("\n" + "=" * (self.grid_size * 4 + 3))
        
        if action is not None:
            print(f"  执行动作: {self.ACTION_SYMBOLS.get(action, '?')} ({['左','下','右','上'][action]})")
        
        print(f"  当前位置: ({row}, {col})")
        print("  " + "-" * (self.grid_size * 4 + 1))
        
        for r in range(self.grid_size):
            line = "  |"
            for c in range(self.grid_size):
                if (r, c) == current_pos:
                    # 当前位置用 ★ 标记
                    cell = ' ★ '
                elif show_path and (r, c) in self.path_history[:-1]:
                    # 历史路径用 ○ 标记
                    cell = ' ○ '
                elif self.desc is not None:
                    # 显示地图元素
                    symbol = self.CELL_SYMBOLS.get(self.desc[r, c], '?')
                    cell = f' {symbol} '
                else:
                    cell = ' · '
                line += cell + '|'
            print(line)
            print("  " + "-" * (self.grid_size * 4 + 1))
        
        print("=" * (self.grid_size * 4 + 3))
    
    def print_path_summary(self):
        """打印路径摘要"""
        if not self.path_history:
            print("没有路径记录")
            return
        
        print("\n=== 飞行路径摘要 ===")
        print(f"总步数: {len(self.path_history) - 1}")
        print("路径: ", end='')
        for i, (r, c) in enumerate(self.path_history):
            if i > 0:
                print(" -> ", end='')
            print(f"({r},{c})", end='')
        print()
    
    def animate_path(self, path: List[Tuple[int, int]], 
                    actions: List[int] = None,
                    delay: float = 0.5):
        """
        动画显示路径
        
        Args:
            path: 位置序列 [(row, col), ...]
            actions: 动作序列
            delay: 每帧延迟（秒）
        """
        self.clear_path()
        
        for i, pos in enumerate(path):
            self.add_position(pos[0], pos[1])
            action = actions[i] if actions and i < len(actions) else None
            self.print_grid(pos, show_path=True, action=action, clear_screen=True)
            time.sleep(delay)
        
        self.print_path_summary()


def plot_q_table_heatmap(q_table: np.ndarray, 
                        grid_size: int = 4,
                        title: str = 'Q-Table 热力图',
                        save_path: Optional[str] = None,
                        show: bool = True):
    """
    将 Q-Table 以热力图形式可视化
    
    Args:
        q_table: Q 值表，形状 (n_states, n_actions)
        grid_size: 网格大小
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    action_names = ['左', '下', '右', '上']
    
    for action in range(4):
        ax = axes[action // 2, action % 2]
        
        # 提取该动作的 Q 值并重塑为网格
        q_values = q_table[:, action].reshape(grid_size, grid_size)
        
        im = ax.imshow(q_values, cmap='RdYlGn', aspect='equal')
        ax.set_title(f'动作: {action_names[action]}')
        
        # 添加数值标签
        for i in range(grid_size):
            for j in range(grid_size):
                text = ax.text(j, i, f'{q_values[i, j]:.2f}',
                             ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"热力图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# 测试代码
if __name__ == '__main__':
    # 测试奖励曲线绘制
    vis = Visualizer(save_dir='./test_plots')
    
    # 模拟奖励数据
    rewards = np.random.random(500) * np.linspace(0, 1, 500)
    vis.plot_rewards(rewards, title='测试奖励曲线', show=True)
    
    # 测试网格可视化
    desc = np.array([
        ['S', 'F', 'F', 'F'],
        ['F', 'H', 'F', 'H'],
        ['F', 'F', 'F', 'H'],
        ['H', 'F', 'F', 'G']
    ])
    
    grid_vis = GridVisualizer(grid_size=4, desc=desc)
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (3, 3)]
    grid_vis.animate_path(path, delay=0.3)

