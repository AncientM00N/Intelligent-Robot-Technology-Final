"""
DQN 确定性模型训练脚本

训练一个确定性的 DDQN 模型：
- is_slippery=False (无风，无随机性)
- 适合实机精确控制
- 保存为 dqn_deterministic.pth
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.frozen_lake_wrapper import FrozenLakeWrapper
from agents.dqn_agent import DQNAgent
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_dqn_model(is_slippery=False, model_name='dqn_deterministic'):
    """
    训练 DQN 模型
    
    Args:
        is_slippery: 是否启用打滑（随机性）
        model_name: 模型保存名称
    """
    
    mode_str = "不确定性" if is_slippery else "确定性"
    print("\n" + "=" * 60)
    print(f"训练 {mode_str} DQN 模型（实机飞行）")
    print("=" * 60)
    
    # 创建环境
    env = FrozenLakeWrapper(map_size='4x4', is_slippery=is_slippery)
    
    print(f"\n地图布局:")
    env.print_map()
    print(f"模式: {'随机(有风)' if is_slippery else '确定性(无风)'}")
    
    # 创建 agent
    agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_size=10000,
        target_update_freq=100,
        hidden_size=64,
        use_double_dqn=True
    )
    
    # 训练参数
    num_episodes = 3000 if is_slippery else 2000
    max_steps = 100
    
    # 记录
    success_history = []
    loss_history = []
    
    print(f"\n开始训练 {num_episodes} 回合...")
    
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 更新网络（包含存储经验和学习）
            agent.update(state, action, reward, next_state, done)
            
            # 记录损失
            if agent.losses:
                episode_loss.append(agent.losses[-1])
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # 记录
        success_history.append(1 if total_reward > 0 else 0)
        if episode_loss:
            loss_history.append(np.mean(episode_loss))
        
        # 每 500 回合打印进度
        if (episode + 1) % 500 == 0:
            recent_success = sum(success_history[-500:]) / 5
            recent_loss = np.mean(loss_history[-500:]) if loss_history else 0
            print(f"\n回合 {episode + 1}: 成功率={recent_success:.1f}%, 损失={recent_loss:.4f}")
    
    # 保存模型
    model_path = f'models/{model_name}.pth'
    agent.save(model_path)
    print(f"\n✓ 模型已保存: {model_path}")
    
    # 打印最终统计
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    total_success = sum(success_history)
    print(f"总成功次数: {total_success} / {num_episodes}")
    print(f"总成功率: {total_success / num_episodes * 100:.2f}%")
    
    # 测试策略
    print("\n" + "=" * 60)
    print("测试学到的策略...")
    print("=" * 60)
    
    test_episodes = 100
    test_success = 0
    test_paths = []
    
    for _ in range(test_episodes):
        state, _ = env.reset()
        path = [state]
        
        for step in range(max_steps):
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            path.append(next_state)
            
            if done:
                if reward > 0:
                    test_success += 1
                    test_paths.append(path)
                break
            
            state = next_state
    
    print(f"\n测试成功率: {test_success}/{test_episodes} = {test_success/test_episodes*100:.1f}%")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    # 成功率曲线
    plt.subplot(1, 2, 1)
    window = 100
    success_rate = [np.mean(success_history[max(0, i-window):i+1]) * 100 
                   for i in range(len(success_history))]
    plt.plot(success_rate)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.title(f'{mode_str} DQN - Success Rate')
    plt.grid(True)
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    if loss_history:
        plt.plot(loss_history)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title(f'{mode_str} DQN - Training Loss')
        plt.grid(True)
    
    plt.tight_layout()
    plot_path = f'plots/{model_name}_training.png'
    plt.savefig(plot_path)
    print(f"\n✓ 训练曲线已保存: {plot_path}")
    
    env.close()
    
    return test_success / test_episodes


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("确定性 DDQN 模型训练程序")
    print("=" * 70)
    print("\n训练配置:")
    print("  - 环境: FrozenLake 4x4")
    print("  - 模式: 确定性 (is_slippery=False)")
    print("  - 算法: Double DQN")
    print("  - 回合数: 2000")
    print("  - 用途: 实机精确控制")
    
    confirm = input("\n开始训练? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("已取消训练")
        exit(0)
    
    print("\n" + "=" * 70)
    print("开始训练确定性 DDQN 模型")
    print("=" * 70)
    
    success_rate = train_dqn_model(is_slippery=False, model_name='dqn_deterministic')
    
    # 最终总结
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"测试成功率: {success_rate*100:.1f}%")
    print(f"模型已保存: models/dqn_deterministic.pth")
    
    print("\n现在可以运行实机飞行:")
    print("  python SerialThread.py")
    print("  - 选择模式 1: 确定性 DDQN (精确控制)")
    print("  - 选择模式 2: 不确定性 DDQN (随机干扰)")

