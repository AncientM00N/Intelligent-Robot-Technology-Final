"""
DQN (Deep Q-Network) Agent 实现
包含 Nature DQN 和 Double DQN (DDQN)

使用 PyTorch 实现神经网络
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional
import os

from .base_agent import BaseAgent
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """
    Q 值神经网络
    
    将状态（one-hot 编码）映射到各个动作的 Q 值
    """
    
    def __init__(self, n_states: int, n_actions: int, hidden_size: int = 64):
        """
        初始化网络
        
        Args:
            n_states: 状态空间大小（用于 one-hot 编码）
            n_actions: 动作空间大小
            hidden_size: 隐藏层大小
        """
        super(QNetwork, self).__init__()
        
        self.n_states = n_states
        
        # 网络结构: one-hot -> hidden1 -> hidden2 -> Q values
        self.fc1 = nn.Linear(n_states, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量，形状 (batch_size,) 或 (batch_size, n_states)
            
        Returns:
            Q 值张量，形状 (batch_size, n_actions)
        """
        # 如果输入是状态索引，转换为 one-hot
        if state.dim() == 1 or (state.dim() == 2 and state.shape[1] != self.n_states):
            state = F.one_hot(state.long(), num_classes=self.n_states).float()
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class DQNAgent(BaseAgent):
    """
    DQN Agent
    
    实现了以下功能:
    1. Experience Replay: 使用经验回放缓冲区打破数据相关性
    2. Target Network (Nature DQN): 使用目标网络稳定训练
    3. Double DQN (DDQN): 解耦动作选择和价值评估，解决 Q 值过估计问题
    """
    
    def __init__(self, n_states: int, n_actions: int,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 64,
                 buffer_size: int = 10000,
                 target_update_freq: int = 100,
                 hidden_size: int = 64,
                 use_double_dqn: bool = True,
                 device: Optional[str] = None):
        """
        初始化 DQN Agent
        
        Args:
            n_states: 状态空间大小
            n_actions: 动作空间大小
            learning_rate: 学习率
            discount_factor: 折扣因子 γ
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay: 探索率衰减系数
            batch_size: 训练批次大小
            buffer_size: 经验回放缓冲区大小
            target_update_freq: 目标网络更新频率
            hidden_size: 网络隐藏层大小
            use_double_dqn: 是否使用 Double DQN
            device: 计算设备 ('cuda' 或 'cpu')
        """
        super().__init__(n_states, n_actions, learning_rate,
                        discount_factor, epsilon_start, epsilon_end, epsilon_decay)
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"DQN Agent 使用设备: {self.device}")
        
        # DQN 特定参数
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.hidden_size = hidden_size
        
        # 创建网络
        self.q_network = QNetwork(n_states, n_actions, hidden_size).to(self.device)
        self.target_network = QNetwork(n_states, n_actions, hidden_size).to(self.device)
        
        # 同步目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 目标网络不需要梯度
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 损失记录
        self.losses: list = []
        
    def select_action(self, state: int, training: bool = True) -> int:
        """
        使用 ε-greedy 策略选择动作
        
        Args:
            state: 当前状态
            training: 是否为训练模式
            
        Returns:
            选择的动作
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # 使用网络选择动作
        with torch.no_grad():
            state_tensor = torch.tensor([state], device=self.device)
            q_values = self.q_network(state_tensor)
            return int(q_values.argmax(dim=1).item())
    
    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool):
        """
        存储经验并执行一次学习更新
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        # 存储经验
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.training_steps += 1
        
        # 如果缓冲区样本不足，跳过学习
        if not self.replay_buffer.is_ready(self.batch_size):
            return
        
        # 从缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device)
        
        # 计算当前 Q 值: Q(s, a)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标 Q 值
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: 用主网络选动作，目标网络评估
                # a* = argmax_a Q(s', a; θ)
                next_actions = self.q_network(next_states).argmax(dim=1)
                # Q_target = r + γ * Q(s', a*; θ-)
                next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Nature DQN: 直接用目标网络选最大 Q 值
                # Q_target = r + γ * max_a Q(s', a; θ-)
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            # 目标值: r + γ * Q(s', a*) * (1 - done)
            target_q = rewards + self.discount_factor * next_q * (1 - dones)
        
        # 计算损失 (MSE Loss)
        loss = F.mse_loss(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()
        
        # 记录损失
        self.losses.append(loss.item())
        
        # 更新目标网络
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """硬更新：直接复制参数到目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def soft_update_target_network(self, tau: float = 0.001):
        """
        软更新：缓慢更新目标网络参数
        
        θ_target = τ * θ + (1 - τ) * θ_target
        
        Args:
            tau: 软更新系数
        """
        for target_param, local_param in zip(self.target_network.parameters(),
                                            self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def get_q_values(self, state: int) -> np.ndarray:
        """
        获取指定状态的 Q 值
        
        Args:
            state: 状态
            
        Returns:
            该状态下所有动作的 Q 值数组
        """
        with torch.no_grad():
            state_tensor = torch.tensor([state], device=self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def save(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_steps': self.training_steps,
            'losses': self.losses,
            # 保存配置以便重建
            'config': {
                'n_states': self.n_states,
                'n_actions': self.n_actions,
                'hidden_size': self.hidden_size,
                'use_double_dqn': self.use_double_dqn
            }
        }, filepath)
        print(f"DQN 模型已保存到: {filepath}")
    
    def load(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.training_steps = checkpoint['training_steps']
        self.losses = checkpoint.get('losses', [])
        print(f"DQN 模型已加载: {filepath}")
    
    def get_training_stats(self) -> dict:
        """获取训练统计信息，包含 DQN 特有的信息"""
        stats = super().get_training_stats()
        if len(self.losses) > 0:
            stats['avg_loss'] = np.mean(self.losses[-100:])
            stats['buffer_size'] = len(self.replay_buffer)
        return stats


# 测试代码
if __name__ == '__main__':
    print("测试 DQN Agent")
    
    agent = DQNAgent(n_states=16, n_actions=4, 
                    batch_size=4, buffer_size=100)
    
    # 模拟一些经验
    for _ in range(10):
        state = np.random.randint(0, 16)
        action = np.random.randint(0, 4)
        reward = np.random.random()
        next_state = np.random.randint(0, 16)
        done = np.random.random() > 0.8
        agent.update(state, action, reward, next_state, done)
    
    print(f"训练步数: {agent.training_steps}")
    print(f"缓冲区大小: {len(agent.replay_buffer)}")
    print(f"状态 0 的 Q 值: {agent.get_q_values(0)}")
    print(f"状态 0 的最优动作: {agent.get_best_action(0)}")

