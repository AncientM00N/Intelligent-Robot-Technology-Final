"""
Agent 基类
定义所有强化学习 Agent 的通用接口
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Optional
import os


class BaseAgent(ABC):
    """
    强化学习 Agent 基类
    
    所有具体的 Agent 实现（如 Q-Learning, DQN）都应继承此类
    """
    
    def __init__(self, n_states: int, n_actions: int, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        初始化 Agent
        
        Args:
            n_states: 状态空间大小
            n_actions: 动作空间大小
            learning_rate: 学习率
            discount_factor: 折扣因子 gamma
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay: 探索率衰减系数
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.training_steps = 0
        
    @abstractmethod
    def select_action(self, state: int, training: bool = True) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            training: 是否为训练模式（训练时使用 epsilon-greedy）
            
        Returns:
            选择的动作
        """
        pass
    
    @abstractmethod
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool):
        """
        更新 Agent（学习）
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        pass
    
    @abstractmethod
    def get_q_values(self, state: int) -> np.ndarray:
        """
        获取指定状态的 Q 值
        
        Args:
            state: 状态
            
        Returns:
            该状态下所有动作的 Q 值数组
        """
        pass
    
    def get_best_action(self, state: int) -> int:
        """
        获取最优动作（贪心策略）
        
        Args:
            state: 当前状态
            
        Returns:
            最优动作
        """
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self):
        """重置探索率"""
        self.epsilon = self.epsilon_start
    
    def record_episode(self, total_reward: float, episode_length: int):
        """
        记录一个回合的统计信息
        
        Args:
            total_reward: 回合总奖励
            episode_length: 回合步数
        """
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
    
    def get_training_stats(self) -> dict:
        """
        获取训练统计信息
        
        Returns:
            包含各种统计指标的字典
        """
        if len(self.episode_rewards) == 0:
            return {}
        
        recent_rewards = self.episode_rewards[-100:]  # 最近100回合
        return {
            'total_episodes': len(self.episode_rewards),
            'total_steps': self.training_steps,
            'current_epsilon': self.epsilon,
            'avg_reward': np.mean(self.episode_rewards),
            'avg_reward_100': np.mean(recent_rewards),
            'max_reward': np.max(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'success_rate_100': np.mean([1 if r > 0 else 0 for r in recent_rewards])
        }
    
    @abstractmethod
    def save(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型路径
        """
        pass
    
    def get_policy(self) -> np.ndarray:
        """
        获取当前策略（每个状态的最优动作）
        
        Returns:
            策略数组，policy[s] = 状态 s 的最优动作
        """
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            policy[s] = self.get_best_action(s)
        return policy

