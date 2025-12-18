import serial
import os
import sys
import time
import numpy as np
from queue import Queue, Empty
from threading import Thread
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ReadDataParser import ReadDataParser, Fh0cBase
from CommandConstructor import CommandConstructor
from QueueSignal import QueueSignal

# 导入 RL agents（现在都在同一目录结构下）
from agents.dqn_agent import DQNAgent
import random


class ThreadLocal:
    """used by thead"""
    latest_cmd: bytearray = None
    q: Queue = None
    s: serial.Serial = None
    t: Thread = None
    exit_queue: Queue = Queue()

    rdp: ReadDataParser = None

    def __init__(self):
        pass

    pass


def task_write(thead_local: ThreadLocal):
    """
    serial port write worker thread, this function do the write task in a independent thread.
    this function must run in a independent thread
    :param thead_local: the thread local data object
    """
    print("task_write")
    while True:
        time.sleep(0.02)
        try:
            if thead_local.exit_queue.get(block=False) is QueueSignal.SHUTDOWN:
                break
        except Empty:
            pass
        try:
            d = thead_local.q.get(block=False, timeout=-1)
            if isinstance(d, tuple):
                print("Tuple:", (d[0], d[1].hex(' ')))
                if d[0] is QueueSignal.CMD and len(d[1]) > 0:
                    thead_local.latest_cmd = d[1]
                    pass
                pass
        except Empty:
            pass
        if thead_local.latest_cmd is not None and len(thead_local.latest_cmd) > 0:
            thead_local.s.write(thead_local.latest_cmd)
        pass
    print("task_write done.")
    pass


def task_read(thead_local: ThreadLocal):
    """
    serial port read worker thread
    """
    print("task_read\n")
    while True:
        time.sleep(0.02)
        try:
            if thead_local.exit_queue.get(block=False) is QueueSignal.SHUTDOWN:
                break
        except Empty:
            pass
        d = thead_local.s.read(65535)
        thead_local.rdp.push(d)
    print("task_read done.")
    pass


class SerialThreadCore:
    """
    the core function of serial control
    """

    s: serial.Serial = None
    port: str = None
    thead_local_write: ThreadLocal = None
    thead_local_read: ThreadLocal = None

    def __init__(self, port: str):
        self.port = port
        self.q_write: Queue = Queue()
        self.q_read: Queue = Queue()
        self.s = serial.Serial(port, baudrate=500000, timeout=0.01)

        self.thead_local_write = ThreadLocal()
        self.thead_local_write.q = self.q_write
        self.thead_local_write.s = self.s
        self.thead_local_write.t = Thread(target=task_write, args=(self.thead_local_write,))

        self.thead_local_read = ThreadLocal()
        self.thead_local_read.q = self.q_read
        self.thead_local_read.s = self.s
        self.thead_local_read.rdp = ReadDataParser(self.thead_local_read.q)
        self.thead_local_read.t = Thread(target=task_read, args=(self.thead_local_read,))

        self.thead_local_write.t.start()
        self.thead_local_read.t.start()

    def shutdown(self):
        self.thead_local_write.exit_queue.put(QueueSignal.SHUTDOWN)
        self.thead_local_read.exit_queue.put(QueueSignal.SHUTDOWN)
        self.thead_local_write.t.join()
        self.thead_local_read.t.join()
        self.s.close()
        pass

    def fh0c_base(self) -> Fh0cBase:
        return self.thead_local_read.rdp.get_fh0c_base()
    pass


class SerialThread(SerialThreadCore):
    """
    this class extends SerialThreadCore, and implements more useful functions
    """

    ss: CommandConstructor = None

    def __init__(self, port: str):
        super().__init__(port)
        self.ss = CommandConstructor(self.thead_local_write.q)
        print("ss", self.ss)
        pass

    def send(self) -> CommandConstructor:
        return self.ss

    pass


# ==================== RL 飞行控制配置 ====================

class Config:
    """配置参数"""
    SERIAL_PORT = "COM7"        # 串口号
    GRID_SIZE = 4               # 4x4 网格
    CELL_SIZE_CM = 50           # 每个格子 50cm
    TAKEOFF_HEIGHT = 80         # 起飞高度 cm
    MOVE_DELAY = 3.0            # 每步移动后等待时间（秒）
    
    # FrozenLake 地图
    FROZEN_LAKE_MAP = [
        'SFFF',   # S=起点
        'FHFH',   # H=冰窟窿
        'FFFH',
        'HFFG'    # G=目标
    ]


# 动作定义
ACTION_NAMES = {0: '左移', 1: '前进', 2: '右移', 3: '后退'}


def get_next_pos(row, col, action):
    """计算执行动作后的位置"""
    if action == 0 and col > 0:         # 左
        return row, col - 1
    elif action == 2 and col < 3:       # 右
        return row, col + 1
    elif action == 1 and row < 3:       # 前进（下）
        return row + 1, col
    elif action == 3 and row > 0:       # 后退（上）
        return row - 1, col
    return row, col  # 边界


def get_cell_type(row, col):
    """获取格子类型"""
    return Config.FROZEN_LAKE_MAP[row][col]


def print_map(cur_row, cur_col):
    """打印地图"""
    print("\n  +" + "---+" * 4)
    for r in range(4):
        line = "  |"
        for c in range(4):
            if r == cur_row and c == cur_col:
                line += " ★ |"
            else:
                line += f" {Config.FROZEN_LAKE_MAP[r][c]} |"
        print(line)
        print("  +" + "---+" * 4)


class GridVisualizer:
    """4x4网格可视化"""
    
    def __init__(self):
        """初始化可视化窗口"""
        plt.ion()  # 交互模式
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title('无人机飞行可视化')
        
        # 颜色映射
        self.colors = {
            'S': '#90EE90',  # 浅绿色 - 起点
            'F': '#FFFFFF',  # 白色 - 安全区域
            'H': '#87CEEB',  # 天蓝色 - 冰窟窿
            'G': '#FFD700'   # 金色 - 目标
        }
        
        self.path_history = []
        self.current_pos = None
        
    def draw_grid(self, current_row, current_col, path=None):
        """
        绘制网格
        
        Args:
            current_row: 当前行
            current_col: 当前列
            path: 历史路径 [(row, col), ...]
        """
        self.ax.clear()
        
        # 绘制网格背景
        for r in range(4):
            for c in range(4):
                cell_type = Config.FROZEN_LAKE_MAP[r][c]
                color = self.colors.get(cell_type, 'white')
                
                # 绘制方格
                rect = patches.Rectangle(
                    (c, 3-r), 1, 1,
                    linewidth=2,
                    edgecolor='black',
                    facecolor=color
                )
                self.ax.add_patch(rect)
                
                # 添加文字标签
                self.ax.text(
                    c + 0.5, 3-r + 0.5,
                    cell_type,
                    ha='center', va='center',
                    fontsize=20, fontweight='bold'
                )
        
        # 绘制历史路径
        if path and len(path) > 1:
            path_rows = [3-p[0] for p in path]
            path_cols = [p[1] + 0.5 for p in path]
            path_rows = [r + 0.5 for r in path_rows]
            
            self.ax.plot(
                path_cols, path_rows,
                'b--', linewidth=2, alpha=0.6,
                label='飞行轨迹'
            )
        
        # 绘制当前位置（无人机）
        drone_y = 3 - current_row + 0.5
        drone_x = current_col + 0.5
        
        # 无人机图标（红色圆圈）
        circle = patches.Circle(
            (drone_x, drone_y), 0.3,
            color='red', zorder=10
        )
        self.ax.add_patch(circle)
        
        # 无人机标记
        self.ax.text(
            drone_x, drone_y, '🚁',
            ha='center', va='center',
            fontsize=30, zorder=11
        )
        
        # 设置坐标轴
        self.ax.set_xlim(0, 4)
        self.ax.set_ylim(0, 4)
        self.ax.set_aspect('equal')
        self.ax.set_xticks(range(5))
        self.ax.set_yticks(range(5))
        self.ax.grid(True, linewidth=2)
        
        # 设置标签
        self.ax.set_xlabel('列 (Col)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('行 (Row)', fontsize=12, fontweight='bold')
        self.ax.set_title(
            f'无人机位置: ({current_row}, {current_col})',
            fontsize=14, fontweight='bold'
        )
        
        # 添加图例
        legend_elements = [
            patches.Patch(facecolor=self.colors['S'], label='起点 (S)'),
            patches.Patch(facecolor=self.colors['F'], label='安全 (F)'),
            patches.Patch(facecolor=self.colors['H'], label='冰窟窿 (H)'),
            patches.Patch(facecolor=self.colors['G'], label='目标 (G)'),
            patches.Patch(facecolor='red', label='无人机 🚁')
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def close(self):
        """关闭窗口"""
        plt.ioff()
        plt.close(self.fig)


def run_rl_flight(st: SerialThread, model_path: str, is_stochastic: bool = False):
    """
    执行 RL 飞行任务
    
    Args:
        st: SerialThread 实例
        model_path: 模型路径
        is_stochastic: 是否启用随机干扰（风力模拟）
    """
    mode_str = "不确定性 (有风)" if is_stochastic else "确定性 (无风)"
    print("\n" + "=" * 60)
    print(f"🚁 无人机 DQN 飞行控制 - {mode_str}")
    print("=" * 60)
    
    # 加载模型
    n_states = Config.GRID_SIZE * Config.GRID_SIZE
    n_actions = 4
    
    agent = DQNAgent(n_states, n_actions)
    
    # 获取模型绝对路径（SerialThread.py 现在在主目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_model_path = os.path.join(script_dir, model_path)
    
    if os.path.exists(abs_model_path):
        agent.load(abs_model_path)
        print(f"✓ 模型已加载: {abs_model_path}")
    else:
        print(f"✗ 模型不存在: {abs_model_path}")
        print("  请先运行 python train.py 训练模型")
        return
    
    # 打印地图
    print("\n地图布局:")
    for row in range(4):
        print("  ", end="")
        for col in range(4):
            print(f"{Config.FROZEN_LAKE_MAP[row][col]:^3}", end=" ")
        print()
    
    if is_stochastic:
        print("\n⚠ 随机干扰模式:")
        print("  - 每次移动有 33% 概率受到风力干扰")
        print("  - 干扰会导致无人机偏移到相邻格子")
        print("  - 这模拟了训练时的随机环境")
    
    # 初始位置
    current_row, current_col = 0, 0
    path = [(0, 0)]
    actions_taken = []
    
    print(f"\n配置:")
    print(f"  起点: (0, 0)")
    print(f"  每格: {Config.CELL_SIZE_CM}cm")
    print(f"  移动等待: {Config.MOVE_DELAY}秒")
    
    print_map(current_row, current_col)
    
    # 创建可视化窗口
    print("\n正在打开可视化窗口...")
    visualizer = GridVisualizer()
    visualizer.draw_grid(current_row, current_col, path)
    
    input("\n按 Enter 开始飞行...")
    
    # 起飞
    print(f"\n>>> 起飞到 {Config.TAKEOFF_HEIGHT}cm...")
    st.send().takeoff(Config.TAKEOFF_HEIGHT)
    time.sleep(5)  # 等待起飞稳定
    print(">>> 起飞完成")
    
    print("\n>>> 开始执行 RL 策略...")
    
    max_steps = 999  # 几乎不限制步数
    
    for step in range(max_steps):
        print(f"\n{'='*50}")
        print(f"步骤 {step + 1}")
        print(f"{'='*50}")
        print(f"当前位置: ({current_row}, {current_col})")
        print(f"格子类型: {get_cell_type(current_row, current_col)}")
        
        # 检查是否到达目标
        if get_cell_type(current_row, current_col) == 'G':
            print("\n🎉 成功到达目标!")
            break
        
        # 检查是否掉入冰窟窿
        if get_cell_type(current_row, current_col) == 'H':
            print("\n💀 掉入冰窟窿!")
            break
        
        # 获取动作
        state = current_row * 4 + current_col
        action = agent.select_action(state, training=False)
        
        print(f"模型决策: {ACTION_NAMES[action]} (action={action})")
        
        # 随机干扰（模拟风力）
        actual_action = action
        if is_stochastic and random.random() < 0.33:
            # 33% 概率受到风力干扰，随机偏移到垂直方向
            possible_deviations = []
            
            # 对于前进/后退，可能左右偏移
            if action in [1, 3]:  # 前进或后退
                possible_deviations = [0, 2]  # 左或右
            # 对于左右移动，可能前后偏移
            elif action in [0, 2]:  # 左或右
                possible_deviations = [1, 3]  # 前或后
            
            if possible_deviations:
                actual_action = random.choice(possible_deviations)
                print(f"  💨 受到风力干扰! 实际执行: {ACTION_NAMES[actual_action]}")
        
        # 计算下一个位置
        next_row, next_col = get_next_pos(current_row, current_col, actual_action)
        
        if (next_row, next_col) == (current_row, current_col):
            print(f"  ⚠ 边界阻挡! 当前 ({current_row},{current_col}), 想要{ACTION_NAMES[action]}但已到边界")
            print(f"    边界限制: 行[0-3], 列[0-3]")
            print("  >>> 跳过此步，等待下一次决策...")
            time.sleep(1)
        else:
            # 执行飞行动作（使用实际动作，可能受风力影响）
            distance = Config.CELL_SIZE_CM
            print(f"  执行: {ACTION_NAMES[actual_action]} ({distance}cm)")
            
            if actual_action == 0:      # 左（反过来执行右）
                st.send().right(distance)
            elif actual_action == 1:    # 前进（下）
                st.send().forward(distance)
            elif actual_action == 2:    # 右（反过来执行左）
                st.send().left(distance)
            elif actual_action == 3:    # 后退（上）
                st.send().back(distance)
            
            # 等待移动完成
            print(f"  等待 {Config.MOVE_DELAY} 秒...")
            time.sleep(Config.MOVE_DELAY)
            
            # 更新位置
            current_row, current_col = next_row, next_col
            path.append((current_row, current_col))
            actions_taken.append(actual_action)
        
        # 显示地图
        print_map(current_row, current_col)
        
        # 更新可视化
        visualizer.draw_grid(current_row, current_col, path)
    
    if step + 1 >= max_steps:
        print(f"\n⏰ 达到最大步数 {max_steps}")
    
    # 降落
    print("\n>>> 降落...")
    st.send().land()
    time.sleep(3)
    print(">>> 已降落")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("飞行总结")
    print("=" * 60)
    print(f"总步数: {len(actions_taken)}")
    print(f"最终位置: ({current_row}, {current_col})")
    
    cell = get_cell_type(current_row, current_col)
    result = '成功!' if cell == 'G' else '失败(冰窟窿)' if cell == 'H' else '未完成'
    print(f"结果: {result}")
    
    print(f"\n路径: {' → '.join([f'({r},{c})' for r, c in path])}")
    print(f"动作: {[ACTION_NAMES[a] for a in actions_taken]}")
    print("=" * 60)
    
    # 保持可视化窗口显示
    print("\n可视化窗口将保持打开，按 Enter 关闭...")
    input()
    visualizer.close()


# ==================== 主程序 ====================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚁 无人机控制系统")
    print("=" * 60)
    
    print("\n选择模式:")
    print("  1. DQN 确定性模型 (无风，精确控制) [需训练]")
    print("  2. DDQN 不确定性模型 (有风，随机干扰) [使用现有 ddqn.pth]")
    print("  3. 键盘手动控制")
    
    mode = input("\n选择 (1/2/3): ").strip()
    
    # 连接无人机
    port = input(f"串口 (默认 {Config.SERIAL_PORT}): ").strip()
    if not port:
        port = Config.SERIAL_PORT
    
    print(f"\n连接 {port}...")
    
    try:
        st = SerialThread(port)
        print(f"✓ 连接成功")
        
        if mode == '1':
            # 确定性模型（需要训练）
            model_path = 'models/dqn_deterministic.pth'
            if not os.path.exists(model_path):
                print(f"\n✗ 模型不存在: {model_path}")
                print("  请先运行: python train_dqn_models.py")
            else:
                run_rl_flight(st, model_path, is_stochastic=False)
            
        elif mode == '2':
            # 不确定性模型（使用现有的 ddqn.pth）
            model_path = 'models/ddqn.pth'
            if not os.path.exists(model_path):
                print(f"\n✗ 模型不存在: {model_path}")
            else:
                run_rl_flight(st, model_path, is_stochastic=True)
        else:
            # 键盘控制模式
            import keyboard
            
            print("\n键盘控制模式")
            print("  T=起飞, L=降落, Q=退出")
            print("  W=前进, S=后退, A=左移, D=右移")
            
            is_flying = False
            
            while True:
                time.sleep(0.02)
                
                if keyboard.is_pressed('t') and not is_flying:
                    print(">>> 起飞")
                    st.send().takeoff(Config.TAKEOFF_HEIGHT)
                    is_flying = True
                    time.sleep(1)
                
                elif keyboard.is_pressed('l') and is_flying:
                    print(">>> 降落")
                    st.send().land()
                    is_flying = False
                    time.sleep(1)
                
                elif keyboard.is_pressed('q'):
                    if is_flying:
                        st.send().land()
                        time.sleep(2)
                    break
                
                elif is_flying:
                    if keyboard.is_pressed('w'):
                        st.send().forward(5)
                    elif keyboard.is_pressed('s'):
                        st.send().back(5)
                    elif keyboard.is_pressed('a'):
                        st.send().left(5)
                    elif keyboard.is_pressed('d'):
                        st.send().right(5)
        
    except Exception as e:
        print(f"✗ 错误: {e}")
    
    finally:
        if 'st' in locals():
            st.shutdown()
        print(">>> 程序结束")
