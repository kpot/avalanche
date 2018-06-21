#include <algorithm>
#include <iostream>

#include "avalanche/CLMemoryManager.h"
#include "avalanche/opencl_utils.h"

namespace avalanche {


void CLMemoryManager::init_for_all_gpus() {
    init_for_all(CL_DEVICE_TYPE_GPU);
}

void CLMemoryManager::init_for_all(const cl_device_type device_type) {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    for (const auto &platform: all_platforms) {
        std::string platform_name;
        platform.getInfo(CL_PLATFORM_NAME, &platform_name);
        std::vector<cl::Device> devices;
        platform.getDevices(device_type, &devices);
        if (!devices.empty()) {
            cl::Context context(devices);
            for (const auto &device: devices) {
                std::string device_name;
                device.getInfo(CL_DEVICE_NAME, &device_name);
                bool is_new_device = (
                    std::find_if(_buffer_pools.begin(), _buffer_pools.end(),
                                 [&device](const BufferPoolRef &pool){
                                     return pool->is_linked_with_device(device);
                                 })
                    == _buffer_pools.end());
                if (is_new_device) {
                    std::cout << "Registered OpenCL device "
                              << device_name << "\n";
                    cl::CommandQueue queue;
                    bool supports_ooo_execution;
                    try {
                        queue = cl::CommandQueue(
                            context, device,
                            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
                        supports_ooo_execution = true;
                    } catch (cl::Error &e) {
                        if (e.err() == CL_INVALID_VALUE ||
                                e.err() == CL_INVALID_QUEUE_PROPERTIES) {
                            std::cerr
                                << "The platform (" << platform_name
                                << ") doesn't support out of order "
                                << "execution. Falling back to default mode.\n";
                            // Apple OpenCL platform may not support
                            // out-of-order execution and not differentiate
                            // between these two error codes.
                            queue = cl::CommandQueue(context, device, 0);
                        }
                        supports_ooo_execution = false;
                    }

//                        (device_name == "Iris" ?
//                         0 : CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE));
                    _buffer_pools.push_back(
                        std::make_shared<CLBufferPool>(
                            this, _device_counter, context, queue));
                    _device_info.push_back(
                        DeviceInfo({device_name, platform_name,
                                    _device_counter, supports_ooo_execution}));
                    ++_device_counter;
                } else {
                    std::cout << "The device '" << device_name
                              << "' has already been registered\n";
                }
            }
        }
    }
}

BufferPoolRef CLMemoryManager::buffer_pool(DeviceIndex idx) {
    return _buffer_pools[idx];
}

std::size_t CLMemoryManager::num_devices() const {
    return _buffer_pools.size();
}

static std::shared_ptr<CLMemoryManager> default_memory_manager;
static std::mutex default_memory_manager_access;

std::shared_ptr<CLMemoryManager> CLMemoryManager::get_default() {
    std::lock_guard<std::mutex> lock(default_memory_manager_access);
    if (!default_memory_manager) {
        default_memory_manager = std::make_shared<CLMemoryManager>();
        default_memory_manager->init_for_all_gpus();
    }
    return default_memory_manager;
}

CLMemoryManager::~CLMemoryManager() {
    std::cout << "CLMemory manager has been destroyed\n";
}

const DeviceInfo &CLMemoryManager::device_info(DeviceIndex idx) const {
    return _device_info.at(idx);
}

const std::vector<DeviceInfo> &CLMemoryManager::list_devices() const {
    return _device_info;
}

} // namespace
