// lty added
#ifndef RL_INTERFACE_H
#define RL_INTERFACE_H

#include <stdint.h>
#include <array>
#include <cstddef>
#include "ns3/ptr.h"
#include "ns3/object.h"

namespace ns3 {

class RdmaQueuePair;

// 共享内存基本配置（与 PPO_AICC 的布局对齐）
static constexpr uint16_t kRlMaxFlows = 64;
static constexpr size_t kRlSlotBytes = 64; // 每个槽位固定 64B
static const char kRlShmName[] = "/rl_env_0";

// 状态位定义（与 Python 端一致）
static constexpr int32_t kStatusPythonWait = 0;
static constexpr int32_t kStatusCppWrote   = 1;
static constexpr int32_t kStatusPythonWrote = 2;
static constexpr int32_t kStatusFinished   = 99;

#pragma pack(push, 1)
// 9 个特征字段（全部 uint32，顺序需与 PPO_AICC 保持一致）
struct RlRawFeatures {
    uint32_t rtt_ns;
    uint32_t nack_count;
    uint32_t cnp_count;
    uint32_t bytes_sent;
    uint32_t rate_fp20;
    uint32_t interval_ns;
    uint32_t packets_sent;
    uint32_t flow_id;
    uint32_t host_id;
};

// 单个槽位：状态 + 动作 + 特征，共 64B
struct RlShmSlot {
    volatile int32_t status;
    float action;
    RlRawFeatures features;
    uint8_t reserved[kRlSlotBytes - sizeof(int32_t) - sizeof(float) - sizeof(RlRawFeatures)];
};
#pragma pack(pop)

static_assert(sizeof(RlRawFeatures) == 36, "RlRawFeatures must be 36 bytes");
static_assert(sizeof(RlShmSlot) == kRlSlotBytes, "RlShmSlot must be 64 bytes");

struct RlShmLayout {
    RlShmSlot slots[kRlMaxFlows];
};

// 内部采样结构，填充到共享内存特征字段
struct RlSample {
    uint32_t rtt_ns;
    uint32_t nack_count;
    uint32_t cnp_count;
    uint32_t bytes_sent;
    uint32_t rate_fp20;
    uint32_t interval_ns;
    uint32_t packets_sent;
    uint32_t flow_id;
    uint32_t host_id;
};

class RLInterface {
public:
    static RLInterface* Get(); // 单例获取
    
    RLInterface();
    ~RLInterface();

    // 初始化共享内存
    void Init();

    // 注册 QP 到共享内存槽位
    // 返回值：分配到的 slot index，如果满了返回 -1
    int RegisterQp(Ptr<RdmaQueuePair> qp);

    // 注销 QP，释放槽位
    void UnregisterQp(int slot_index);

    // 写入一条采样数据
    // 返回是否成功写入
    bool PublishSample(int slot_index, const RlSample& sample);

    // lty added: 等待指定 slot 的外部动作写回，阻塞直到 status == kStatusPythonWrote
    bool WaitForResult(int slot_index, uint64_t timeout_us = 0);

    // lty added: 读取外部写回的动作值（不阻塞）
    bool GetResultValue(int slot_index, double& out_value);

    // lty added: 处理完一次MI后清空槽位，避免数据滞留
    void ClearSlot(int slot_index);

private:
    void EnsureInit();

    int m_shm_fd;
    RlShmLayout* m_shm_ptr;
    
    // 维护 slot_index -> QP 的映射，方便快速访问
    std::array<Ptr<RdmaQueuePair>, kRlMaxFlows> m_active_qps;
};

} // namespace ns3

#endif
