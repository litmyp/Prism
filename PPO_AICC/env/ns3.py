import gym
import numpy as np
from typing import Tuple, Dict

from config.config import Config
from .utils.feature_history import FeatureHistory
# 引入我们刚刚确认过的 SharedMemoryServer
from .utils.shared_memory_server import SharedMemoryServer

class ns3(gym.Env):
    """
    A GYM wrapper for the ns3 simulator using Shared Memory.
    
    Changes from original:
    1. Communication: Socket -> Shared Memory
    2. Lifecycle: Python passively waits for C++ (no subprocess.Popen)
    3. Configuration: Logic is agnostic to scenario types (M2O/A2A), handled purely by data.
    """
    def __init__(self, scenario: str, env_index: int, dummy_idx: int, config: Config):
        # 保留 scenario 和 dummy_idx 参数以兼容 env_utils.py 的接口，但内部不再依赖它们解析逻辑
        self.config = config
        self.env_index = env_index
        
        # [核心改动] 使用全局唯一索引生成共享内存 Key
        # 例如: rl_env_0, rl_env_1 ...
        self.shm_key = f"rl_env_{self.env_index}"
        
        print(f"Initializing Env [{self.env_index}] with SHM Key: {self.shm_key}")

        # 初始化共享内存服务 (只建立连接对象，不阻塞)
        self.server = SharedMemoryServer(self.config, self.shm_key)

        # 特征历史记录器 (保持原逻辑)
        self.feature_history = FeatureHistory(self.config, self.env_index)
        
        # 记录上一次的 host/flow 用于 action 对应
        # 注意：由于是异步环境，我们假设 step(action) 是针对上一次观测到的 agent
        self.last_host = None
        self.last_flow_tag = None

        # 定义 Gym 空间
        self.action_space = gym.spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32)
        
        # 计算 Observation 空间维度
        number_of_features = self.feature_history.number_of_features * self.config.env.history_length
        self.observation_space = gym.spaces.Box(np.tile(-np.inf, number_of_features),
                                                np.tile(np.inf, number_of_features),
                                                dtype=np.float32)

    def seed(self, seed: int = None) -> None:
        """
        环境的随机性由 C++ 仿真器控制，这里留空。
        """
        pass

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        重置环境。
        逻辑：清理历史 -> 阻塞等待 C++ 端写入的第一帧 State。
        """
        print(f"[{self.shm_key}] Resetting... Waiting for C++ simulation to start/write initial state...")
        
        # 1. 重置历史记录
        self.feature_history.reset()
        self.last_host = None
        self.last_flow_tag = None

        # 2. [阻塞] 通过共享内存获取第一帧数据
        raw_features = self.server.reset()
        
        # 检查是否获取失败（比如 C++ 端直接结束了）
        if raw_features is None:
            print(f"[{self.shm_key}] Error: Failed to receive initial state (Simulation ended?).")
            # 返回全0状态防止崩溃，虽然理论上应该抛出异常或结束
            return np.zeros(self.observation_space.shape), {}

        # 3. 更新历史并处理观测
        self.feature_history.update_history(raw_features)
        
        # 记录当前的 ID，以便下一次 Step 时更新 Action
        self.last_host = raw_features.host
        self.last_flow_tag = raw_features.flow_tag

        state, info, _ = self.feature_history.process_observation(raw_features.host, raw_features.flow_tag)

        # 构造 Info 字典 (兼容 BaseAgent 的 split 逻辑)
        # Agent key format: "Scenario_EnvNum/Host/QP" -> 我们这里简化为 "EnvNum/Host/QP"
        # 只要保证 unique 即可
        agent_key = f"Env_{self.env_index}/{raw_features.host}/{raw_features.flow_tag}"

        info.update(dict(
            agent_key=agent_key, 
            reward=0, 
            env_num=self.env_index, 
            host=raw_features.host, 
            qp=raw_features.flow_tag
        ))

        print(f"[{self.shm_key}] Connection established. Received initial state from Host={raw_features.host}, Flow={raw_features.flow_tag}")
        return state, info

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        与环境交互一步。
        逻辑：写入 Action -> 阻塞等待下一帧 State -> 计算 Reward
        """
        # 1. 更新历史记录中的 Action (针对上一个 State 的 Action)
        if self.last_host is not None:
            self.feature_history.update_action(self.last_host, self.last_flow_tag, action)

        # 2. [阻塞] 发送动作并等待下一个状态
        raw_features = self.server.step(action)

        #打印
        if raw_features is not None:
            print(f"recv raw_features: {raw_features}")
            print(f"action sent: {action}")

        # 3. 检查仿真是否结束
        if raw_features is None:
            print(f"[{self.shm_key}] Simulation finished (received None from server).")
            # 按照 Gym 协议，返回 done=True
            return np.zeros(self.observation_space.shape), 0, True, {}

        # 4. 处理接收到的新数据
        self.feature_history.update_history(raw_features)
        
        state, info, _ = self.feature_history.process_observation(raw_features.host, raw_features.flow_tag)

        # 5. 计算奖励
        reward = self._calculate_reward(action, info)

        # 6. 更新 ID 记录
        self.last_host = raw_features.host
        self.last_flow_tag = raw_features.flow_tag
        self.previous_cur_rate = raw_features.cur_rate # 兼容某些可能的内部逻辑

        # 构造 Info
        agent_key = f"Env_{self.env_index}/{raw_features.host}/{raw_features.flow_tag}"
        info.update(dict(
            agent_key=agent_key, 
            reward=reward, 
            host=raw_features.host, 
            qp=raw_features.flow_tag, 
            env_num=self.env_index
        ))

        return state, reward, False, info

    def _calculate_reward(self, action: float, info: Dict) -> float:
        """
        计算奖励函数。保持原逻辑不变。
        """
        if self.config.env.reward == 'general':
            # 注意：feature_history 已经在 info 里放入了 nack_ratio, rtt_inflation 等
            reward = (action - 1) - info['rtt_inflation'] * 0.1 - info['cnp_ratio'] - int(info['nack_ratio'] > 0) * 1000
        elif self.config.env.reward == 'distance':
            # 这种奖励依赖于场景解析(optimal_rate)，我们这里可能需要给一个默认值
            # 或者假设 config 里已经有了 target rate
            # 这里简单处理，如果无法解析场景，就给个通用计算
            reward = - (info['bandwidth'] * 1. / 12.5 - 0.1) ** 2 
        elif self.config.env.reward == 'constrained':
            if info['cnp_ratio'] > 2:
                reward = -1
            else:
                reward = action - 1. - info['rtt_inflation'] * 0.1
        else:
            # 尝试直接从 info 获取 (adpg_reward 等)
            reward = info.get(self.config.env.reward, 0)
            
        return reward

    def render(self, mode: str = 'human') -> None:
        pass

    def close(self) -> None:
        if self.server:
            self.server.close()