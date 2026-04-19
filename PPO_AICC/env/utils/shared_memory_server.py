import time
import struct
import numpy as np
from multiprocessing import shared_memory
from collections import namedtuple

# ---------------------------------------------------------
# 1. 保留原有的数据结构，确保兼容性
# ---------------------------------------------------------
# [重要] 这里必须和 feature_history.py 里的定义以及 C++ 端的写入顺序完全一致
RawFeatures = namedtuple(
    'RawFeatures',
    [
        'rtt_packet_delay',     # float: normalized latency
        'nacks_received',       # uint32
        'cnps_received',        # uint32
        'bytes_sent',           # uint32
        'cur_rate',             # float: normalized rate
        'monitor_interval_width', # uint32
        'packets_sent',         # uint32
        'flow_tag',             # str (decoded from uint32)
        'host',                 # str (decoded from uint32)
    ]
)

# ---------------------------------------------------------
# 2. 定义通信协议常量
# ---------------------------------------------------------
# 状态位定义 (用于同步 Python 和 C++)
STATUS_PYTHON_WAIT = 0   # Python 正在等待 C++ 写入数据
STATUS_CPP_WROTE   = 1   # C++ 已经写完 State，Python 可以读了
STATUS_PYTHON_WROTE = 2  # Python 已经写完 Action，C++ 可以读了
STATUS_FINISHED    = 99  # 仿真结束标志

# 内存布局偏移量 (单位: 字节)
# [ 0-3 ] Status (int32)
# [ 4-7 ] Action (float)
# [ 8-43] State Data (9个数值: 2个float, 7个uint32) -> 36 bytes
OFFSET_STATUS = 0
OFFSET_ACTION = 4
OFFSET_DATA   = 8

# 共享内存总大小 (64字节足够容纳上述数据)
SHM_SLOT_SIZE = 64
MAX_FLOWS = 64  # 必须与 C++ kRlMaxFlows 对齐

class SharedMemoryServer:
    """
    使用共享内存替代 Socket 进行通信。
    机制：基于状态位(Status Flag)的忙等待(Spinlock)。
    """
    def __init__(self, config, shm_key: str):
        self.config = config
        self.shm_key = shm_key
        self._shm = None
        self._buf = None
        self.is_connected = False
        self.active_slot = 0  # 当前正在交互的槽位索引

    def connect(self):
        """
        尝试连接到 C++ 创建的共享内存。
        注意：这里采用“被动”策略，假设 C++ 端负责创建和销毁共享内存，
        或者由外部脚本创建。Python 端只负责 Attach。
        """
        if self.is_connected:
            return

        print(f"[{self.shm_key}] 等待连接共享内存...")
        while self._shm is None:
            try:
                # 尝试打开现有的共享内存块
                self._shm = shared_memory.SharedMemory(name=self.shm_key)
                self._buf = self._shm.buf
                self.is_connected = True
                print(f"[{self.shm_key}] 成功连接到共享内存。")
            except FileNotFoundError:
                # 如果内存还没被创建，就稍等一下继续试
                time.sleep(0.5)

    def reset(self) -> RawFeatures:
        """
        重置连接状态，并等待第一帧数据。
        """
        if not self.is_connected:
            self.connect()

        # 阻塞等待第一帧 State
        return self._wait_for_state()

    def step(self, action: float) -> RawFeatures:
        """
        发送动作，并等待下一个状态。
        """
        offset = self.active_slot * SHM_SLOT_SIZE

        # 1. 写入 Action (float)
        # struct.pack_into format: 'f' for float (4 bytes)
        struct.pack_into('f', self._buf, offset + OFFSET_ACTION, action)

        # 2. 修改状态位：告诉 C++ "Action 写好了，你可以读了"
        self._set_status(STATUS_PYTHON_WROTE, offset)

        # 3. 阻塞等待 C++ 处理并返回新的 State
        return self._wait_for_state()

    def _wait_for_state(self) -> RawFeatures:
        """
        内部函数：忙等待(Busy Wait)，轮询所有槽位直到任一槽位的 Status 变为 STATUS_CPP_WROTE
        """
        while True:
            for i in range(MAX_FLOWS):
                offset = i * SHM_SLOT_SIZE
                status = self._get_status(offset)

                if status == STATUS_CPP_WROTE:
                    # 锁定当前活跃槽位
                    self.active_slot = i
                    # C++ 数据已就绪，开始读取
                    features = self._read_features(offset)
                    # 读完后，把状态改回 "等待中"，防止重复读取
                    self._set_status(STATUS_PYTHON_WAIT, offset)
                    return features

                elif status == STATUS_FINISHED:
                    # C++ 通知仿真结束
                    return None

            # 简单的自旋等待，避免 CPU 100% 占用，加一点点 sleep
            # 如果追求极致性能，可以去掉 sleep，但在调试阶段建议保留
            time.sleep(0.0001) 

    def _read_features(self, offset: int = 0) -> RawFeatures:
        """
        严格还原 server.py 的逻辑：
        C++ 写入 9 个 uint32。
        我们读出来后，再在 Python 里做除法转换。
        """
        # 格式: 9个 I (unsigned int) -> 36 bytes
        # 修正：之前我用了 'fIIIfIIII' 是假设 C++ 改发 float，现在还原为 'I'*9
        data = struct.unpack_from('I' * 9, self._buf, offset + OFFSET_DATA)
        
        # 必须导入原来的常量，或者硬编码
        # from env.utils import BASE_RTT 
        BASE_RTT = 4160 # 假设值，需要修改
        
        return RawFeatures(
            rtt_packet_delay=data[0] / BASE_RTT,        # Int -> Float
            nacks_received=data[1],
            cnps_received=data[2],
            bytes_sent=data[3],
            cur_rate=data[4] * 1. / (1 << 20),          # Fixed-Point Int -> Float
            monitor_interval_width=data[5],
            packets_sent=data[6],
            flow_tag=str(data[7]), 
            host=str(data[8]),     
        )
    def _get_status(self, offset: int = 0):
        return struct.unpack_from('i', self._buf, offset + OFFSET_STATUS)[0]

    def _set_status(self, status, offset: int = 0):
        # 写入 int32 状态位
        struct.pack_into('i', self._buf, offset + OFFSET_STATUS, status)

    def close(self):
        if self._shm:
            try:
                self._shm.close()
                # 只有创建者才应该 unlink，如果是 attach 的，通常只 close
                # self._shm.unlink() 
            except:
                pass
            self.is_connected = False
