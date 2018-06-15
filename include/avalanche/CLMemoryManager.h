#ifndef AVALANCHE_MEMORY_MANAGER_H
#define AVALANCHE_MEMORY_MANAGER_H

#include <memory>

#include "CL_cust/cl2.hpp"

#include "avalanche/CLBufferPool.h"

namespace avalanche {

using DeviceIndex = std::size_t;

struct DeviceInfo {
    std::string name;
    std::string platform;
    DeviceIndex id;
    bool supports_out_of_order_execution;
};

class CLMemoryManager {
    /*
     * The manager should track buffers for all devices.
     */
public:
    CLMemoryManager() :_device_counter{0} {}
    ~CLMemoryManager();
    void init_for_all_gpus();
    void init_for_all(cl_device_type device_type);
    BufferPoolRef buffer_pool(DeviceIndex idx);
    std::size_t num_devices() const;
    const DeviceInfo& device_info(DeviceIndex idx) const;
    const std::vector<DeviceInfo>& list_devices() const;
    static std::shared_ptr<CLMemoryManager> get_default();

private:
    std::vector<BufferPoolRef> _buffer_pools;
    std::size_t _device_counter;
    std::vector<DeviceInfo> _device_info;
};

using MemoryManagerRef = std::shared_ptr<CLMemoryManager>;

} //namespace

#endif //AVALANCHE_MEMORY_MANAGER_H
