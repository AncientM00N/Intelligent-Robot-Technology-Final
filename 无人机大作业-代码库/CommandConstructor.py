from queue import Queue
from struct import pack, unpack, pack_into, unpack_from
from enum import Enum

from QueueSignal import QueueSignal


class CmdType(Enum):
    MULTI_SETTINGS = 1
    SINGLE_SETTINGS = 2
    SINGLE_CONTROL = 3
    READ_SETTINGS = 4
    pass


class CommandConstructorCore:
    """
    this class impl core read functions
    """

    """
    last order was generate
    """
    order_last = 0
    """
    
    """
    q_write: Queue = None

    def _order_count(self):
        """
        the order_count generator, it generates new order number
        """
        self.order_last = (self.order_last + 1) % 127
        return self.order_last
        pass

    def _check_sum(self, header: bytearray, params: bytearray):
        """
        impl checksum algorithm
        """
        return bytearray([sum(header + params) & 0xFF])
        pass

    def sendCommand(self, data: bytearray):
        """
        direct do the command bytearray send task
        """
        self.q_write.put((QueueSignal.CMD, data), block=True)
        pass

    def __init__(self, q_write: Queue):
        self.q_write = q_write
        pass

    def join_cmd(self, type: CmdType, params: bytearray):
        """
        The join_cmd function concatenates the header, params and checksum to a cmd bytearray to let it become a valid serial cmd package.
        """
        header = bytearray(b'\xBB\x1D')  # HEAD=0xBB, LEN=0x1D(29)

        data_body = bytearray(29)
        data_body[0] = 0xF3  # FUN byte

        if len(params) > 13:
            raise ValueError(f"params length {len(params)} exceeds maximum 13 bytes")

        data_body[1:1+len(params)] = params

        # Copy params to cmd2 position
        data_body[14:14+len(params)] = params

        data_body[27] = 0x64  # volume = 100

        data_body[28] = 0x00  # reserved = 0

        checksum = self._check_sum(header, data_body)

        return header + data_body + checksum

    pass


class CommandConstructor(CommandConstructorCore):
    """
    this class extends CommandConstructorCore,
    it impl all functions that direct construct command .
    TODO Implement All Command Methods On This Class
    """

    def __init__(self, q_write: Queue):
        super().__init__(q_write)
        pass

    def led(self, mode, r, g, b):
        """
        控制无人机LED灯光
        :param mode: 灯光模式 (0常亮 1呼吸灯 2七彩变幻)
        :param r: 红色 0-255
        :param g: 绿色 0-255
        :param b: 蓝色 0-255
        """
        if mode < 0 or mode > 2:
            raise ValueError("mode illegal", mode)
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)              # 无人机ID
        pack_into('<B', params, 1, 0x0D)              # CMD: 灯光控制
        pack_into('<B', params, 2, self._order_count())  # 命令编号
        pack_into('<B', params, 3, mode)              # 灯光模式
        pack_into('<B', params, 4, r)                 # 红色
        pack_into('<B', params, 5, g)                 # 绿色
        pack_into('<B', params, 6, b)                 # 蓝色
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("led", cmd.hex(' '))
        self.sendCommand(cmd)

    def takeoff(self, high: int, ):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 0)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<h', params, 3, high)
        pack_into('<B', params, 5, 50)
        pack_into('<B', params, 6, 0)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("take_off", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def land(self, ):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 254)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 0)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("take_off", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def move(self, direction: int, distance: int):
        """
        控制无人机移动
        :param direction: 移动方向 (1前 2后 3左 4右 5上 6下 7↖ 8↗ 9↙ 10↘)
        :param distance: 移动距离，单位：厘米
        """
        if direction < 1 or direction > 10:
            raise ValueError("direction illegal", direction)
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)              # 无人机ID
        pack_into('<B', params, 1, 0x05)              # CMD: 移动
        pack_into('<B', params, 2, self._order_count())  # 命令编号
        pack_into('<B', params, 3, direction)         # 移动方向
        pack_into('<h', params, 4, distance)          # 移动距离 s16
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("move", cmd.hex(' '))
        self.sendCommand(cmd)

    def up(self, distance: int):
        """向上移动"""
        self.move(5, distance)

    def down(self, distance: int):
        """向下移动"""
        self.move(6, distance)

    def forward(self, distance: int):
        """向前移动"""
        self.move(1, distance)

    def back(self, distance: int):
        """向后移动"""
        self.move(2, distance)

    def left(self, distance: int):
        """向左移动"""
        self.move(3, distance)

    def right(self, distance: int):
        """向右移动"""
        self.move(4, distance)

    def flip(self, direction: int, circle: int):
        """
        控制无人机翻滚
        :param direction: 翻滚方向 (1前 2后 3左 4右)
        :param circle: 翻滚圈数 (1或2)
        """
        if direction < 1 or direction > 4:
            raise ValueError("direction illegal", direction)
        if circle != 1 and circle != 2:
            raise ValueError("circle illegal", circle)
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)              # 无人机ID
        pack_into('<B', params, 1, 0x0C)              # CMD: 翻滚
        pack_into('<B', params, 2, self._order_count())  # 命令编号
        pack_into('<B', params, 3, direction)         # 翻滚方向
        pack_into('<B', params, 4, circle)            # 翻滚圈数
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("flip", cmd.hex(' '))
        self.sendCommand(cmd)

    def flip_forward(self, circle: int):
        """向前翻滚"""
        self.flip(1, circle)

    def flip_back(self, circle: int):
        """向后翻滚"""
        self.flip(2, circle)

    def flip_left(self, circle: int):
        """向左翻滚"""
        self.flip(3, circle)

    def flip_right(self, circle: int):
        """向右翻滚"""
        self.flip(4, circle)

    def arrive(self, x: int, y: int, z: int):
        """
        直达指定坐标
        :param x: x坐标，单位：厘米
        :param y: y坐标，单位：厘米
        :param z: z坐标（高度），单位：厘米
        """
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)              # 无人机ID
        pack_into('<B', params, 1, 0x09)              # CMD: 直达
        pack_into('<B', params, 2, self._order_count())  # 命令编号
        pack_into('<h', params, 3, x)                 # x坐标 s16
        pack_into('<h', params, 5, y)                 # y坐标 s16
        pack_into('<h', params, 7, z)                 # z坐标 s16
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("arrive", cmd.hex(' '))
        self.sendCommand(cmd)

    def rotate(self, degree: int):
        """
        控制无人机自转
        :param degree: 自转角度，正数顺时针，负数逆时针，单位：度
        """
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)              # 无人机ID
        pack_into('<B', params, 1, 0x0A)              # CMD: 自转
        pack_into('<B', params, 2, self._order_count())  # 命令编号
        pack_into('<h', params, 3, degree)            # 自转角度 s16
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("rotate", cmd.hex(' '))
        self.sendCommand(cmd)

    def speed(self, speed: int):
        """
        设置飞行速度
        :param speed: 速度，范围：0~200厘米/秒
        """
        if speed < 0 or speed > 200:
            raise ValueError("speed illegal", speed)
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)              # 无人机ID
        pack_into('<B', params, 1, 0x02)              # CMD: 设置飞行速度
        pack_into('<B', params, 2, self._order_count())  # 命令编号
        pack_into('<h', params, 3, speed)             # 速度 s16
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("speed", cmd.hex(' '))
        self.sendCommand(cmd)

    def high(self, high: int):
        """
        设置飞行高度
        :param high: 高度，单位：厘米
        """
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)              # 无人机ID
        pack_into('<B', params, 1, 0x0B)              # CMD: 设置飞行高度
        pack_into('<B', params, 2, self._order_count())  # 命令编号
        pack_into('<h', params, 3, high)              # 高度 s16
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("high", cmd.hex(' '))
        self.sendCommand(cmd)

    def airplane_mode(self, mode: int):
        """
        切换飞行模式
        :param mode: 模式 (1常规 2巡线 3跟随 4单机编队)
        """
        if mode < 1 or mode > 4:
            raise ValueError("mode illegal", mode)
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)              # 无人机ID
        pack_into('<B', params, 1, 0x01)              # CMD: 切换飞行模式
        pack_into('<B', params, 2, self._order_count())  # 命令编号
        pack_into('<B', params, 3, mode)              # 飞行模式
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("airplane_mode", cmd.hex(' '))
        self.sendCommand(cmd)

    def hovering(self):
        """
        悬停
        """
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)              # 无人机ID
        pack_into('<B', params, 1, 0xFE)              # CMD: 悬停(与降落相同)
        pack_into('<B', params, 2, self._order_count())  # 命令编号
        pack_into('<B', params, 3, 4)                 # mode=4 表示悬停
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("hovering", cmd.hex(' '))
        self.sendCommand(cmd)

    def read_multi_setting(self):
        return self.read_setting(0x02)

    def read_single_setting(self):
        return self.read_setting(0x04)

    def read_hardware_setting(self):
        return self.read_setting(0xA0)

    def read_setting(self, mode: int):
        params = bytearray(1)
        pack_into("!B", params, 0, mode)
        cmd = self.join_cmd(CmdType.READ_SETTINGS, params)
        print("cmd", cmd.hex(' '))
        self.sendCommand(cmd)
        pass
    pass
