// lty added
#include "rl-interface.h"
#include "rdma-queue-pair.h"
#include "ns3/log.h"
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <iostream>
#include <limits>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("RLInterface");

static RLInterface* g_instance = nullptr;

RLInterface* RLInterface::Get() {
    if (g_instance == nullptr) {
        g_instance = new RLInterface();
    }
    return g_instance;
}

RLInterface::RLInterface() : m_shm_fd(-1), m_shm_ptr(nullptr) {
    m_active_qps.fill(nullptr);
}

RLInterface::~RLInterface() {
    if (m_shm_ptr != nullptr && m_shm_ptr != MAP_FAILED) {
        munmap(m_shm_ptr, sizeof(RlShmLayout));
    }
    if (m_shm_fd != -1) {
        close(m_shm_fd);
        shm_unlink(kRlShmName); // 删除共享内存文件
    }
}

void RLInterface::Init() {
    // 1. 创建共享内存对象
    m_shm_fd = shm_open(kRlShmName, O_CREAT | O_RDWR, 0666);
    if (m_shm_fd == -1) {
        NS_FATAL_ERROR("Failed to shm_open: " << strerror(errno));
    }

    // 2. 设置文件大小
    if (ftruncate(m_shm_fd, sizeof(RlShmLayout)) == -1) {
        NS_FATAL_ERROR("Failed to ftruncate shm: " << strerror(errno));
    }

    // 3. 映射到内存
    m_shm_ptr = (RlShmLayout*)mmap(0, sizeof(RlShmLayout), PROT_READ | PROT_WRITE, MAP_SHARED, m_shm_fd, 0);
    if (m_shm_ptr == MAP_FAILED) {
        NS_FATAL_ERROR("Failed to mmap shm: " << strerror(errno));
    }

    // 4. 初始化内存区域
    memset(m_shm_ptr, 0, sizeof(RlShmLayout));
    for (int i = 0; i < kRlMaxFlows; ++i) {
        m_shm_ptr->slots[i].status = kStatusPythonWait;
    }
    
    NS_LOG_INFO("RLInterface Initialized. Shm Size: " << sizeof(RlShmLayout));
    std::cout << "RLInterface: Shared Memory created at " << kRlShmName << std::endl;
}

void RLInterface::EnsureInit() {
    if (!m_shm_ptr) {
        Init();
    }
}

int RLInterface::RegisterQp(Ptr<RdmaQueuePair> qp) {
    EnsureInit();

    // 寻找空闲槽位
    for (int i = 0; i < kRlMaxFlows; ++i) {
        if (m_active_qps[i] == nullptr) {
            m_active_qps[i] = qp;

            // 初始化槽位
            memset(&m_shm_ptr->slots[i], 0, sizeof(RlShmSlot));
            m_shm_ptr->slots[i].status = kStatusPythonWait;

            NS_LOG_INFO("Registered QP Hash: " << qp->GetHash() << " at Slot: " << i);
            return i;
        }
    }
    NS_LOG_WARN("RLInterface: No free slots available for new QP!");
    return -1;
}

void RLInterface::UnregisterQp(int slot_index) {
    if (slot_index < 0 || slot_index >= kRlMaxFlows) return;
    
    if (m_active_qps[slot_index] != nullptr) {
        m_active_qps[slot_index] = nullptr;
        if (m_shm_ptr) {
            memset(&m_shm_ptr->slots[slot_index], 0, sizeof(RlShmSlot));
            m_shm_ptr->slots[slot_index].status = kStatusPythonWait;
        }
        NS_LOG_INFO("Unregistered QP at Slot: " << slot_index);
    }
}

bool RLInterface::PublishSample(int slot_index, const RlSample& sample) {
    if (!m_shm_ptr || slot_index < 0 || slot_index >= kRlMaxFlows) {
        return false;
    }
    RlShmSlot& slot = m_shm_ptr->slots[slot_index];

    // 写入特征（均为 uint32），不在这里做单位缩放以保持结构对齐
    slot.features.rtt_ns = sample.rtt_ns;
    slot.features.nack_count = sample.nack_count;
    slot.features.cnp_count = sample.cnp_count;
    slot.features.bytes_sent = sample.bytes_sent;
    slot.features.rate_fp20 = sample.rate_fp20;
    slot.features.interval_ns = sample.interval_ns;
    slot.features.packets_sent = sample.packets_sent;
    slot.features.flow_id = sample.flow_id;
    slot.features.host_id = sample.host_id;

    // 更新状态位，通知 Python 已写好数据
    slot.status = kStatusCppWrote;
    return true;
}

bool RLInterface::WaitForResult(int slot_index, uint64_t timeout_us) {
    // lty added: 阻塞等待外部结果写回，等待 status 变为 kStatusPythonWrote
    if (!m_shm_ptr || slot_index < 0 || slot_index >= kRlMaxFlows) {
        return false;
    }
    RlShmSlot& slot = m_shm_ptr->slots[slot_index];
    const useconds_t sleep_us = 50; // 轻量等待，避免忙等
    uint64_t waited = 0;
    while (slot.status != kStatusPythonWrote) {
        usleep(sleep_us);
        waited += sleep_us;
        if (timeout_us > 0 && waited >= timeout_us) {
            NS_LOG_WARN("RLInterface: wait result timeout. slot=" << slot_index
                         << " waited_us=" << waited);
            break;
        }
    }
    return slot.status == kStatusPythonWrote;
}

bool RLInterface::GetResultValue(int slot_index, double& out_value) {
    // lty added: 读取外部写回的动作值（float -> double），不阻塞
    if (!m_shm_ptr || slot_index < 0 || slot_index >= kRlMaxFlows) {
        return false;
    }
    RlShmSlot& slot = m_shm_ptr->slots[slot_index];
    out_value = static_cast<double>(slot.action);
    return true;
}

void RLInterface::ClearSlot(int slot_index) {
    // lty added: 清空槽位内容（seq/result等），下一次采样重写
    if (!m_shm_ptr || slot_index < 0 || slot_index >= kRlMaxFlows) {
        return;
    }
    memset(&m_shm_ptr->slots[slot_index], 0, sizeof(RlShmSlot));
    m_shm_ptr->slots[slot_index].status = kStatusPythonWait;
}

} // namespace ns3
