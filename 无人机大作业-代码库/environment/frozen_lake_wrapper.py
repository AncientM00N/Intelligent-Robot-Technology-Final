"""
FrozenLake 环境封装类
支持 4x4/8x8 地图切换，以及 Deterministic/Stochastic 模式切换
"""

import gym
import numpy as np
from typing import Tuple, Optional, List


class FrozenLakeWrapper:
    """
    FrozenLake 环境封装器
    
    动作空间: 0=左, 1=下, 2=右, 3=上
    状态空间: 0 到 n_states-1 的整数
    
    地图说明:
    - S: 起点 (Start)
    - F: 冰面 (Frozen) - 安全
    - H: 冰窟窿 (Hole) - 掉落则失败
    - G: 目标 (Goal) - 到达则成功
    """
    
    # 动作映射
    ACTION_LEFT = 0
    ACTION_DOWN = 1
    ACTION_RIGHT = 2
    ACTION_UP = 3
    
    ACTION_NAMES = {0: '左', 1: '下', 2: '右', 3: '上'}
    
    def __init__(self, map_size: str = '4x4', is_slippery: bool = True, 
                 render_mode: Optional[str] = None, custom_map: Optional[List[str]] = None):
        """
        初始化 FrozenLake 环境
        
        Args:
            map_size: 地图大小 '4x4' 或 '8x8'
            is_slippery: True=随机模式(有风力干扰), False=确定性模式
            render_mode: 渲染模式 'human', 'rgb_array', 'ansi' 或 None
            custom_map: 自定义地图数组，如 ["SFFF", "FHFH", "FFFH", "HFFG"]
        """
        self.map_size = map_size
        self.is_slippery = is_slippery
        self.render_mode = render_mode
        
        # 确定地图尺寸
        if map_size == '4x4':
            self.grid_size = 4
            map_name = '4x4'
        elif map_size == '8x8':
            self.grid_size = 8
            map_name = '8x8'
        else:
            raise ValueError(f"不支持的地图大小: {map_size}，请使用 '4x4' 或 '8x8'")
        
        # 创建环境
        if custom_map is not None:
            self.env = gym.make('FrozenLake-v1', 
                              desc=custom_map,
                              is_slippery=is_slippery,
                              render_mode=render_mode)
            self.grid_size = len(custom_map)
        else:
            self.env = gym.make('FrozenLake-v1',
                              map_name=map_name,
                              is_slippery=is_slippery,
                              render_mode=render_mode)
        
        # 状态和动作空间
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        
        # 获取地图描述
        self.desc = self.env.unwrapped.desc.astype(str)
        
    def reset(self) -> Tuple[int, dict]:
        """
        重置环境
        
        Returns:
            state: 初始状态
            info: 额外信息
        """
        state, info = self.env.reset()
        return int(state), info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """
        执行一步动作
        
        Args:
            action: 动作 (0=左, 1=下, 2=右, 3=上)
            
        Returns:
            next_state: 下一个状态
            reward: 奖励
            terminated: 是否终止（到达目标或掉入冰窟窿）
            truncated: 是否截断
            info: 额外信息
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return int(next_state), float(reward), terminated, truncated, info
    
    def render(self):
        """渲染环境"""
        return self.env.render()
    
    def close(self):
        """关闭环境"""
        self.env.close()
    
    def state_to_coord(self, state: int) -> Tuple[int, int]:
        """
        将状态ID转换为网格坐标
        
        Args:
            state: 状态ID (0 到 n_states-1)
            
        Returns:
            (row, col): 网格坐标，左上角为 (0, 0)
        """
        row = state // self.grid_size
        col = state % self.grid_size
        return (row, col)
    
    def coord_to_state(self, row: int, col: int) -> int:
        """
        将网格坐标转换为状态ID
        
        Args:
            row: 行号
            col: 列号
            
        Returns:
            state: 状态ID
        """
        return row * self.grid_size + col
    
    def get_next_coord(self, row: int, col: int, action: int) -> Tuple[int, int]:
        """
        计算执行动作后的目标坐标（不考虑边界）
        
        Args:
            row: 当前行
            col: 当前列
            action: 动作
            
        Returns:
            (new_row, new_col): 目标坐标
        """
        if action == self.ACTION_LEFT:
            return (row, col - 1)
        elif action == self.ACTION_DOWN:
            return (row + 1, col)
        elif action == self.ACTION_RIGHT:
            return (row, col + 1)
        elif action == self.ACTION_UP:
            return (row - 1, col)
        return (row, col)
    
    def is_valid_action(self, state: int, action: int) -> bool:
        """
        检查动作是否有效（不会撞墙）
        
        Args:
            state: 当前状态
            action: 要执行的动作
            
        Returns:
            True 如果动作有效，False 如果会撞墙
        """
        row, col = self.state_to_coord(state)
        new_row, new_col = self.get_next_coord(row, col, action)
        
        # 检查是否超出边界
        if new_row < 0 or new_row >= self.grid_size:
            return False
        if new_col < 0 or new_col >= self.grid_size:
            return False
        return True
    
    def get_valid_actions(self, state: int) -> List[int]:
        """
        获取当前状态下所有有效的动作
        
        Args:
            state: 当前状态
            
        Returns:
            有效动作列表
        """
        valid = []
        for action in range(self.n_actions):
            if self.is_valid_action(state, action):
                valid.append(action)
        return valid
    
    def get_boundary_steps(self, state: int) -> dict:
        """
        获取当前位置到各方向边界的步数（用于实机飞行边界检测）
        
        Args:
            state: 当前状态
            
        Returns:
            dict: {'forward': n, 'back': n, 'left': n, 'right': n}
            注意: forward=下, back=上（与无人机坐标系对应）
        """
        row, col = self.state_to_coord(state)
        return {
            'forward': self.grid_size - 1 - row,  # 向下(前)可移动步数
            'back': row,                           # 向上(后)可移动步数
            'left': col,                           # 向左可移动步数
            'right': self.grid_size - 1 - col     # 向右可移动步数
        }
    
    def get_cell_type(self, state: int) -> str:
        """
        获取指定状态对应的格子类型
        
        Args:
            state: 状态ID
            
        Returns:
            'S'=起点, 'F'=冰面, 'H'=冰窟窿, 'G'=目标
        """
        row, col = self.state_to_coord(state)
        return self.desc[row, col]
    
    def is_terminal(self, state: int) -> bool:
        """
        检查是否为终止状态
        
        Args:
            state: 状态ID
            
        Returns:
            True 如果是终止状态（目标或冰窟窿）
        """
        cell_type = self.get_cell_type(state)
        return cell_type in ['H', 'G']
    
    def is_goal(self, state: int) -> bool:
        """检查是否到达目标"""
        return self.get_cell_type(state) == 'G'
    
    def is_hole(self, state: int) -> bool:
        """检查是否掉入冰窟窿"""
        return self.get_cell_type(state) == 'H'
    
    def print_map(self, current_state: Optional[int] = None):
        """
        打印地图，可选高亮当前位置
        
        Args:
            current_state: 当前状态ID，用于高亮显示
        """
        print("\n" + "=" * (self.grid_size * 4 + 1))
        for row in range(self.grid_size):
            line = "|"
            for col in range(self.grid_size):
                state = self.coord_to_state(row, col)
                cell = self.desc[row, col]
                if current_state is not None and state == current_state:
                    line += f" * |"  # 用 * 标记当前位置
                else:
                    line += f" {cell} |"
            print(line)
            print("-" * (self.grid_size * 4 + 1))
    
    def get_action_name(self, action: int) -> str:
        """获取动作的中文名称"""
        return self.ACTION_NAMES.get(action, '未知')


# 测试代码
if __name__ == '__main__':
    # 测试 4x4 确定性环境
    print("=== 测试 4x4 确定性环境 ===")
    env = FrozenLakeWrapper(map_size='4x4', is_slippery=False)
    state, _ = env.reset()
    print(f"初始状态: {state}, 坐标: {env.state_to_coord(state)}")
    env.print_map(state)
    
    # 测试边界检测
    print(f"\n边界步数: {env.get_boundary_steps(state)}")
    print(f"有效动作: {[env.get_action_name(a) for a in env.get_valid_actions(state)]}")
    
    # 执行几步
    actions = [env.ACTION_DOWN, env.ACTION_RIGHT, env.ACTION_DOWN]
    for action in actions:
        if env.is_valid_action(state, action):
            next_state, reward, done, _, _ = env.step(action)
            print(f"\n动作: {env.get_action_name(action)}")
            print(f"状态: {state} -> {next_state}, 坐标: {env.state_to_coord(next_state)}")
            print(f"奖励: {reward}, 终止: {done}")
            env.print_map(next_state)
            state = next_state
            if done:
                break
    
    env.close()

