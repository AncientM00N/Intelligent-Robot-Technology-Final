"""
Q-Learning Agent 实现
基于 Q-Table 的表格型强化学习算法
"""

import numpy as np
from typing import Optional
import os

from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Q-Learning Agent
    
    Q-Learning 是一种 off-policy 的时序差分学习算法
    更新公式: Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(self, n_states: int, n_actions: int,
                 learning_rate: float = 0.8,
                 discount_factor: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        初始化 Q-Learning Agent
        
        Args:
            n_states: 状态空间大小
            n_actions: 动作空间大小
            learning_rate: 学习率 α
            discount_factor: 折扣因子 γ
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay: 探索率衰减系数
        """
        super().__init__(n_states, n_actions, learning_rate, 
                        discount_factor, epsilon_start, epsilon_end, epsilon_decay)
        
        # 初始化 Q-Table，全部为 0
        self.q_table = np.zeros((n_states, n_actions))
        
    def select_action(self, state: int, training: bool = True) -> int:
        """
        使用 ε-greedy 策略选择动作
        
        Args:
            state: 当前状态
            training: 是否为训练模式
            
        Returns:
            选择的动作
        """
        # 训练时使用 epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # 贪心选择
        return self.get_best_action(state)
    
    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool):
        """
        使用 Q-Learning 更新规则更新 Q 值
        
        Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        self.training_steps += 1
        
        # 当前 Q 值
        current_q = self.q_table[state, action]
        
        # 计算目标 Q 值
        if done:
            target_q = reward
        else:
            # max_a' Q(s', a')
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # TD 误差
        td_error = target_q - current_q
        
        # 更新 Q 值
        self.q_table[state, action] = current_q + self.learning_rate * td_error
    
    def get_q_values(self, state: int) -> np.ndarray:
        """
        获取指定状态的 Q 值
        
        Args:
            state: 状态
            
        Returns:
            该状态下所有动作的 Q 值
        """
        return self.q_table[state].copy()
    
    def save(self, filepath: str):
        """
        保存 Q-Table 到文件
        
        Args:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        np.savez(filepath,
                 q_table=self.q_table,
                 epsilon=self.epsilon,
                 episode_rewards=np.array(self.episode_rewards),
                 episode_lengths=np.array(self.episode_lengths),
                 training_steps=self.training_steps)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str):
        """
        从文件加载 Q-Table
        
        Args:
            filepath: 模型路径
        """
        data = np.load(filepath, allow_pickle=True)
        self.q_table = data['q_table']
        self.epsilon = float(data['epsilon'])
        self.episode_rewards = data['episode_rewards'].tolist()
        self.episode_lengths = data['episode_lengths'].tolist()
        self.training_steps = int(data['training_steps'])
        print(f"模型已加载: {filepath}")
    
    def print_q_table(self, grid_size: int = 4):
        """
        以网格形式打印 Q-Table
        
        Args:
            grid_size: 网格大小（用于格式化显示）
        """
        action_names = ['左', '下', '右', '上']
        print("\n=== Q-Table ===")
        print(f"{'状态':<6}", end='')
        for name in action_names:
            print(f"{name:>8}", end='')
        print(f"{'最优':>8}")
        print("-" * 50)
        
        for s in range(self.n_states):
            row, col = s // grid_size, s % grid_size
            print(f"({row},{col})", end=' ')
            for a in range(self.n_actions):
                print(f"{self.q_table[s, a]:>8.3f}", end='')
            best_action = np.argmax(self.q_table[s])
            print(f"{action_names[best_action]:>8}")
    
    def print_policy(self, grid_size: int = 4):
        """
        以网格形式打印最优策略
        
        Args:
            grid_size: 网格大小
        """
        action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
        print("\n=== 最优策略 ===")
        print("+" + "---+" * grid_size)
        
        for row in range(grid_size):
            line = "|"
            for col in range(grid_size):
                state = row * grid_size + col
                best_action = np.argmax(self.q_table[state])
                # 检查该状态的 Q 值是否全为 0（未探索）
                if np.all(self.q_table[state] == 0):
                    symbol = ' ? '
                else:
                    symbol = f' {action_symbols[best_action]} '
                line += symbol + '|'
            print(line)
            print("+" + "---+" * grid_size)


# 测试代码
if __name__ == '__main__':
    # 简单测试
    agent = QLearningAgent(n_states=16, n_actions=4)
    
    # 模拟一些更新
    agent.update(0, 1, 0, 4, False)
    agent.update(4, 1, 0, 8, False)
    agent.update(8, 2, 0, 9, False)
    agent.update(9, 1, 1, 13, True)  # 到达目标
    
    print("测试 Q-Learning Agent")
    agent.print_q_table()
    agent.print_policy()
    
    # 测试动作选择
    print(f"\n状态 0 的动作: {agent.select_action(0, training=False)}")
    print(f"当前探索率: {agent.epsilon}")

