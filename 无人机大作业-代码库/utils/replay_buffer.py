"""
经验回放缓冲区
用于 DQN 的 Experience Replay 机制
"""

import numpy as np
from collections import deque
import random
from typing import Tuple, List


class ReplayBuffer:
    """
    经验回放缓冲区
    
    存储 (state, action, reward, next_state, done) 元组
    支持随机采样以打破数据相关性
    """
    
    def __init__(self, capacity: int = 10000):
        """
        初始化缓冲区
        
        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: int, action: int, reward: float, 
             next_state: int, done: bool):
        """
        添加一条经验
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        随机采样一个批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (states, actions, rewards, next_states, dones) 元组
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch], dtype=np.float32)
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """返回当前缓冲区大小"""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """
        检查缓冲区是否有足够的样本进行采样
        
        Args:
            batch_size: 所需的批次大小
            
        Returns:
            True 如果有足够的样本
        """
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    优先级经验回放缓冲区（可选实现）
    
    根据 TD 误差对经验进行优先级采样
    误差越大的经验被采样的概率越高
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        """
        初始化优先级缓冲区
        
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数（0=均匀采样，1=完全优先级）
            beta: 重要性采样权重（用于修正偏差）
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state: int, action: int, reward: float,
             next_state: int, done: bool):
        """添加经验，新经验默认最高优先级"""
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        experience = (state, action, reward, next_state, done)
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """根据优先级采样"""
        if self.size < batch_size:
            raise ValueError("缓冲区样本不足")
        
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 采样
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # 计算重要性采样权重
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # 增加 beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 提取经验
        batch = [self.buffer[i] for i in indices]
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch], dtype=np.float32)
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """更新优先级"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size

