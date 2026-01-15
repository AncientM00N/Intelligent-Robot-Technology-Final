"""
无人机飞行演示脚本

实现关键功能:
1. 边界检测：执行动作前检查是否会撞墙
2. 风力干扰模拟：Stochastic 模式下模拟随机漂移
3. 实时位置显示：打印当前网格坐标
4. 图形界面演示：使用 Gym 渲染动画
"""

import gym
import numpy as np
import random
import time
import os
from typing import Optional, Tuple, List

from environment.frozen_lake_wrapper import FrozenLakeWrapper
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from utils.visualization import GridVisualizer


class FlightDemoController:
    """
    无人机飞行演示控制器
    
    核心功能:
    1. 维护边界步数变量 (forward_steps, right_steps, left_steps, back_steps)
    2. 执行动作前进行边界检测
    3. 模拟 Stochastic 模式的风力干扰
    4. 实时显示位置
    """
    
    # 动作定义（与 Gym FrozenLake 一致）
    ACTION_LEFT = 0
    ACTION_DOWN = 1   # 前进 (在网格中向下)
    ACTION_RIGHT = 2
    ACTION_UP = 3     # 后退 (在网格中向上)
    
    ACTION_NAMES = {0: '左', 1: '前(下)', 2: '右', 3: '后(上)'}
    ACTION_SYMBOLS = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    def __init__(self, grid_size: int = 4, 
                 is_stochastic: bool = True,
                 wind_probability: float = 0.333):
        """
        初始化演示控制器
        
        Args:
            grid_size: 网格大小 (4 或 8)
            is_stochastic: 是否为随机模式（有风力干扰）
            wind_probability: 风力干扰概率（每个偏移方向的概率）
                - 0.333: 强风（默认，1/3概率偏移）
                - 0.2: 中风
                - 0.1: 弱风
                - 0.0: 无风
        """
        self.grid_size = grid_size
        self.is_stochastic = is_stochastic
        self.wind_probability = wind_probability
        
        # 当前位置（行, 列）
        self.current_row = 0
        self.current_col = 0
        
        # 边界步数（根据需求文档定义）
        self.forward_steps = grid_size - 1
        self.back_steps = 0
        self.left_steps = 0
        self.right_steps = grid_size - 1
        
        # 飞行历史
        self.path_history: List[Tuple[int, int]] = []
        self.action_history: List[int] = []
        self.intended_actions: List[int] = []
        
        # 环境地图描述
        self.map_desc: Optional[np.ndarray] = None
        
        # 可视化器
        self.visualizer = GridVisualizer(grid_size)
        
    def reset(self, start_row: int = 0, start_col: int = 0):
        """重置控制器到初始位置"""
        self.current_row = start_row
        self.current_col = start_col
        self._update_boundary_steps()
        
        self.path_history.clear()
        self.action_history.clear()
        self.intended_actions.clear()
        self.path_history.append((start_row, start_col))
        
        if self.map_desc is not None:
            self.visualizer.set_map(self.map_desc)
        self.visualizer.clear_path()
        self.visualizer.add_position(start_row, start_col)
        
    def set_map(self, desc: np.ndarray):
        """设置地图描述"""
        self.map_desc = desc
        self.visualizer.set_map(desc)
    
    def _update_boundary_steps(self):
        """更新边界步数"""
        self.forward_steps = self.grid_size - 1 - self.current_row
        self.back_steps = self.current_row
        self.left_steps = self.current_col
        self.right_steps = self.grid_size - 1 - self.current_col
    
    def get_boundary_steps(self) -> dict:
        """获取当前边界步数"""
        return {
            'forward': self.forward_steps,
            'back': self.back_steps,
            'left': self.left_steps,
            'right': self.right_steps
        }
    
    def is_valid_action(self, action: int) -> bool:
        """检查动作是否有效（不会撞墙）"""
        if action == self.ACTION_LEFT:
            return self.left_steps > 0
        elif action == self.ACTION_DOWN:
            return self.forward_steps > 0
        elif action == self.ACTION_RIGHT:
            return self.right_steps > 0
        elif action == self.ACTION_UP:
            return self.back_steps > 0
        return False
    
    def get_valid_actions(self) -> List[int]:
        """获取所有有效动作"""
        return [a for a in range(4) if self.is_valid_action(a)]
    
    def apply_wind_effect(self, intended_action: int) -> Tuple[int, str]:
        """
        应用风力干扰效果
        
        Returns:
            (actual_action, wind_status): 实际动作和风力状态描述
        """
        if not self.is_stochastic or self.wind_probability <= 0:
            return intended_action, "无风"
        
        rand = random.random()
        
        # 计算概率分布
        # 原动作概率 = 1 - 2*wind_probability
        # 左偏概率 = wind_probability
        # 右偏概率 = wind_probability
        no_wind_prob = 1 - 2 * self.wind_probability
        
        if rand < no_wind_prob:
            return intended_action, "无风"
        elif rand < no_wind_prob + self.wind_probability:
            actual_action = (intended_action - 1) % 4
            return actual_action, "左偏风"
        else:
            actual_action = (intended_action + 1) % 4
            return actual_action, "右偏风"
    
    def execute_action(self, intended_action: int, 
                       apply_wind: bool = True,
                       verbose: bool = True) -> Tuple[bool, str, int]:
        """
        执行动作（包含边界检测和风力干扰）
        
        Returns:
            (success, message, actual_action): 是否成功，消息，实际执行的动作
        """
        self.intended_actions.append(intended_action)
        
        # 应用风力干扰
        if apply_wind and self.is_stochastic:
            actual_action, wind_status = self.apply_wind_effect(intended_action)
        else:
            actual_action = intended_action
            wind_status = "无风"
        
        if verbose:
            print(f"\n--- 步骤 {len(self.action_history) + 1} ---")
            print(f"  当前位置: ({self.current_row}, {self.current_col})")
            print(f"  边界: 前={self.forward_steps}, 后={self.back_steps}, "
                  f"左={self.left_steps}, 右={self.right_steps}")
            print(f"  原意动作: {self.ACTION_SYMBOLS[intended_action]} {self.ACTION_NAMES[intended_action]}")
            
            if wind_status != "无风":
                print(f"  ⚠ 风力干扰! [{wind_status}] -> 实际: {self.ACTION_NAMES[actual_action]}")
        
        # 边界检测
        if not self.is_valid_action(actual_action):
            if verbose:
                print(f"  ✗ 边界阻止! 动作 {self.ACTION_NAMES[actual_action]} 会撞墙")
            self.action_history.append(actual_action)
            return False, "边界阻止", actual_action
        
        # 执行动作
        old_row, old_col = self.current_row, self.current_col
        
        if actual_action == self.ACTION_LEFT:
            self.current_col -= 1
        elif actual_action == self.ACTION_DOWN:
            self.current_row += 1
        elif actual_action == self.ACTION_RIGHT:
            self.current_col += 1
        elif actual_action == self.ACTION_UP:
            self.current_row -= 1
        
        self._update_boundary_steps()
        self.path_history.append((self.current_row, self.current_col))
        self.action_history.append(actual_action)
        self.visualizer.add_position(self.current_row, self.current_col)
        
        if verbose:
            print(f"  ✓ 移动: ({old_row}, {old_col}) -> ({self.current_row}, {self.current_col})")
        
        return True, "成功", actual_action
    
    def select_valid_action(self, agent, state: int, max_retries: int = 10) -> int:
        """选择一个有效动作"""
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            raise RuntimeError("没有有效动作!")
        
        for _ in range(max_retries):
            action = agent.select_action(state, training=False)
            if action in valid_actions:
                return action
        
        return random.choice(valid_actions)
    
    def get_current_state(self) -> int:
        """获取当前状态ID"""
        return self.current_row * self.grid_size + self.current_col
    
    def get_cell_type(self) -> str:
        """获取当前格子类型"""
        if self.map_desc is None:
            return 'F'
        return self.map_desc[self.current_row, self.current_col]
    
    def is_terminal(self) -> bool:
        """检查是否到达终止状态"""
        return self.get_cell_type() in ['G', 'H']
    
    def is_goal(self) -> bool:
        return self.get_cell_type() == 'G'
    
    def is_hole(self) -> bool:
        return self.get_cell_type() == 'H'
    
    def print_current_state(self):
        """打印当前状态"""
        if self.map_desc is not None:
            self.visualizer.print_grid(
                (self.current_row, self.current_col),
                show_path=True
            )
    
    def print_summary(self):
        """打印飞行摘要"""
        print("\n" + "=" * 50)
        print("飞行摘要")
        print("=" * 50)
        print(f"总步数: {len(self.action_history)}")
        print(f"最终位置: ({self.current_row}, {self.current_col})")
        
        result = '到达目标!' if self.is_goal() else '掉入冰窟窿!' if self.is_hole() else '未完成'
        print(f"结果: {result}")
        
        # 统计风力影响
        wind_count = sum(1 for i, a in enumerate(self.intended_actions) 
                        if i < len(self.action_history) and a != self.action_history[i])
        print(f"风力偏移次数: {wind_count}/{len(self.action_history)}")
        
        print("\n路径: ", end='')
        print(" -> ".join([f"({r},{c})" for r, c in self.path_history]))
        print("=" * 50)


def run_graphical_demo(agent, grid_size: int = 4, is_slippery: bool = True,
                       wind_strength: str = 'medium', num_episodes: int = 3,
                       step_delay: float = 0.5):
    """
    图形界面演示（小人动画）
    
    Args:
        agent: 训练好的 Agent
        grid_size: 网格大小
        is_slippery: 环境是否随机
        wind_strength: 风力强度 'strong'/'medium'/'weak'/'none'
        num_episodes: 演示回合数
        step_delay: 每步延迟
    """
    # 风力映射
    wind_map = {
        'strong': 0.333,   # 强风 - 1/3概率偏移
        'medium': 0.2,     # 中风 - 20%概率偏移
        'weak': 0.1,       # 弱风 - 10%概率偏移
        'none': 0.0        # 无风
    }
    wind_prob = wind_map.get(wind_strength, 0.2)
    
    map_name = f'{grid_size}x{grid_size}'
    
    # 创建带图形渲染的环境
    env = gym.make('FrozenLake-v1', 
                   map_name=map_name,
                   is_slippery=is_slippery,
                   render_mode='human')
    
    print(f"\n{'='*50}")
    print("图形界面演示")
    print(f"{'='*50}")
    print(f"地图: {map_name}")
    print(f"环境随机性: {'开启' if is_slippery else '关闭'}")
    print(f"风力强度: {wind_strength} (偏移概率: {wind_prob*100:.0f}%)")
    print(f"演示回合: {num_episodes}")
    print(f"{'='*50}")
    
    success_count = 0
    
    for episode in range(num_episodes):
        print(f"\n>>> 回合 {episode + 1}/{num_episodes}")
        
        state, _ = env.reset()
        env.render()
        time.sleep(step_delay)
        
        total_steps = 0
        wind_shifts = 0
        
        for step in range(100):
            # Agent 选择动作
            intended_action = agent.select_action(state, training=False)
            
            # 模拟风力偏移（在我们自己的逻辑中）
            if is_slippery and wind_prob > 0:
                rand = random.random()
                if rand < 1 - 2 * wind_prob:
                    actual_action = intended_action
                elif rand < 1 - wind_prob:
                    actual_action = (intended_action - 1) % 4
                    wind_shifts += 1
                else:
                    actual_action = (intended_action + 1) % 4
                    wind_shifts += 1
            else:
                actual_action = intended_action
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(actual_action)
            done = terminated or truncated
            
            env.render()
            time.sleep(step_delay)
            
            total_steps += 1
            state = next_state
            
            if done:
                if reward > 0:
                    print(f"    🎉 成功! 步数: {total_steps}, 风偏: {wind_shifts}次")
                    success_count += 1
                else:
                    print(f"    💀 失败! 步数: {total_steps}, 风偏: {wind_shifts}次")
                time.sleep(1)
                break
        
        if not done:
            print(f"    ⏰ 超时!")
    
    print(f"\n{'='*50}")
    print(f"演示完成! 成功率: {success_count}/{num_episodes}")
    print(f"{'='*50}")
    
    env.close()


def run_terminal_demo(agent, env_wrapper: FrozenLakeWrapper,
                      wind_strength: str = 'medium',
                      max_steps: int = 100,
                      step_delay: float = 0.3,
                      verbose: bool = True):
    """
    终端文字演示
    """
    wind_map = {
        'strong': 0.333,
        'medium': 0.2,
        'weak': 0.1,
        'none': 0.0
    }
    wind_prob = wind_map.get(wind_strength, 0.2)
    
    controller = FlightDemoController(
        grid_size=env_wrapper.grid_size,
        is_stochastic=env_wrapper.is_slippery,
        wind_probability=wind_prob
    )
    controller.set_map(env_wrapper.desc)
    
    state, _ = env_wrapper.reset()
    start_row, start_col = env_wrapper.state_to_coord(state)
    controller.reset(start_row, start_col)
    
    print(f"\n{'='*50}")
    print("终端文字演示")
    print(f"{'='*50}")
    print(f"网格: {env_wrapper.grid_size}x{env_wrapper.grid_size}")
    print(f"风力: {wind_strength} (偏移概率: {wind_prob*100:.0f}%)")
    print(f"{'='*50}")
    
    controller.print_current_state()
    time.sleep(step_delay)
    
    for step in range(max_steps):
        current_state = controller.get_current_state()
        intended_action = controller.select_valid_action(agent, current_state)
        
        success, msg, actual = controller.execute_action(
            intended_action, 
            apply_wind=env_wrapper.is_slippery,
            verbose=verbose
        )
        
        controller.print_current_state()
        
        if controller.is_terminal():
            if controller.is_goal():
                print("\n🎉 成功到达目标!")
            else:
                print("\n💀 掉入冰窟窿!")
            break
        
        time.sleep(step_delay)
    
    controller.print_summary()
    return controller


def manual_demo():
    """手动控制演示"""
    print("\n" + "=" * 50)
    print("手动控制模式")
    print("=" * 50)
    
    # 配置
    print("\n地图大小: 1=4x4, 2=8x8")
    size = input("选择 (默认1): ").strip()
    grid_size = 8 if size == '2' else 4
    
    print("\n风力强度: 1=无风, 2=弱风, 3=中风, 4=强风")
    wind = input("选择 (默认3): ").strip()
    wind_map = {'1': 'none', '2': 'weak', '3': 'medium', '4': 'strong'}
    wind_strength = wind_map.get(wind, 'medium')
    
    wind_prob_map = {'none': 0, 'weak': 0.1, 'medium': 0.2, 'strong': 0.333}
    wind_prob = wind_prob_map[wind_strength]
    
    env = FrozenLakeWrapper(map_size=f'{grid_size}x{grid_size}', is_slippery=wind_prob > 0)
    controller = FlightDemoController(grid_size, wind_prob > 0, wind_prob)
    controller.set_map(env.desc)
    
    state, _ = env.reset()
    row, col = env.state_to_coord(state)
    controller.reset(row, col)
    
    print("\n操作: W=前, S=后, A=左, D=右, Q=退出")
    controller.print_current_state()
    
    action_map = {'w': 1, 's': 3, 'a': 0, 'd': 2}
    
    while not controller.is_terminal():
        cmd = input("动作: ").strip().lower()
        if cmd == 'q':
            break
        if cmd not in action_map:
            print("无效! 请用 W/A/S/D")
            continue
        
        controller.execute_action(action_map[cmd], apply_wind=wind_prob > 0)
        controller.print_current_state()
    
    controller.print_summary()
    env.close()


def main():
    """主入口"""
    print("\n" + "=" * 50)
    print("🚁 无人机飞行演示系统")
    print("=" * 50)
    print("\n演示模式:")
    print("  1. 图形界面演示 (小人动画)")
    print("  2. 终端文字演示")
    print("  3. 手动控制")
    print("  4. 快速随机测试")
    
    mode = input("\n选择模式 (1/2/3/4): ").strip()
    
    if mode in ['1', '2']:
        # 选择模型
        print("\n选择模型:")
        print("  1. Q-Learning")
        print("  2. DQN")
        print("  3. Double DQN")
        model_choice = input("选择 (默认1): ").strip()
        
        model_paths = {
            '1': ('models/q_learning.npz', 'q_learning'),
            '2': ('models/dqn.pth', 'dqn'),
            '3': ('models/ddqn.pth', 'dqn')
        }
        
        path, model_type = model_paths.get(model_choice, model_paths['1'])
        
        if not os.path.exists(path):
            print(f"\n❌ 模型文件不存在: {path}")
            print("请先运行 python train.py 训练模型")
            return
        
        # 加载模型
        print(f"\n加载模型: {path}")
        if model_type == 'q_learning':
            agent = QLearningAgent(16, 4)  # 4x4
            agent.load(path)
        else:
            agent = DQNAgent(16, 4)
            agent.load(path)
        
        # 选择风力
        print("\n风力强度:")
        print("  1. 无风 (确定性)")
        print("  2. 弱风 (10%偏移)")
        print("  3. 中风 (20%偏移)")
        print("  4. 强风 (33%偏移)")
        wind_choice = input("选择 (默认3): ").strip()
        wind_map = {'1': 'none', '2': 'weak', '3': 'medium', '4': 'strong'}
        wind_strength = wind_map.get(wind_choice, 'medium')
        
        is_slippery = wind_strength != 'none'
        
        if mode == '1':
            # 图形演示
            print("\n演示回合数:")
            episodes = input("输入数量 (默认3): ").strip()
            num_episodes = int(episodes) if episodes.isdigit() else 3
            
            run_graphical_demo(
                agent, 
                grid_size=4,
                is_slippery=is_slippery,
                wind_strength=wind_strength,
                num_episodes=num_episodes,
                step_delay=0.5
            )
        else:
            # 终端演示
            env = FrozenLakeWrapper(map_size='4x4', is_slippery=is_slippery)
            run_terminal_demo(agent, env, wind_strength=wind_strength)
            env.close()
    
    elif mode == '3':
        manual_demo()
    
    else:
        # 随机测试
        print("\n快速随机测试...")
        agent = QLearningAgent(16, 4)
        agent.epsilon = 1.0
        
        run_graphical_demo(
            agent,
            grid_size=4,
            is_slippery=True,
            wind_strength='weak',
            num_episodes=2,
            step_delay=0.3
        )


if __name__ == '__main__':
    main()
