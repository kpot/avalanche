#include <cmath>
#include <cfenv>
#include <iostream>

#include "avalanche/CLBufferPool.h"
#include "avalanche/CLBuffer.h"
#include "avalanche/opencl_utils.h"
#include "avalanche/MultiArray.h"
#include "avalanche/CLMemoryManager.h"
#include "avalanche/logging.h"

#ifndef __CL_ENABLE_EXCEPTIONS
#ifndef CL_HPP_ENABLE_EXCEPTIONS
#error "Avalanche relies on __CL_ENABLE_EXCEPTION being defined"
#endif
#endif

namespace avalanche {


CLBufferPool::CLBufferPool(
    CLMemoryManager* memory_manager,
    std::size_t device_index,
    const cl::Context &context,
    const cl::CommandQueue &device_queue)
    :_memory_manager{memory_manager},
     _device_index{device_index},
     _cl_context{context},
     _device_queue{device_queue},
     _cl_device{get_device_from_queue(device_queue)} {

}
CLBufferPool::~CLBufferPool() {
}


bool CLBufferPool::is_linked_with_device(
    const cl::Device &device) const {
    return device() == _cl_device();
}

std::shared_ptr<CLBuffer>
CLBufferPool::reserve_buffer(std::size_t size_in_bytes) {
    if (size_in_bytes == 0 || size_in_bytes > MaxBufferSize) {
        throw std::invalid_argument(
            "The size must be greater than zero and less than MaxBufferSize");
    }
    long bucket_idx = find_mem_slot(size_in_bytes);
    cl::Buffer cl_buffer;
    {
        std::lock_guard<std::mutex> lock(_buckets_mutex[bucket_idx]);
        auto &right_bucket = _buckets[bucket_idx];
        if (right_bucket.empty()) {
            auto block_size = static_cast<size_t>(1 << bucket_idx);
            cl_buffer = cl::Buffer(_cl_context, CL_MEM_READ_WRITE, block_size);
        } else {
            cl_buffer = std::move(right_bucket.back());
            right_bucket.pop_back();
//            std::cout << "Taking a previously used buffer\n";
        }
    }
    auto *buffer_wrapper = new CLBuffer(
        own_reference(), std::move(cl_buffer), size_in_bytes, bucket_idx);
    return std::shared_ptr<CLBuffer>(buffer_wrapper);
}

long CLBufferPool::find_mem_slot(size_t size) {
    int save_round = fegetround();
    fesetround(FE_UPWARD);
    long bucket = lrint(std::log2(size));
    fesetround(save_round);
    return bucket;
}

void CLBufferPool::return_buffer(cl::Buffer &&cl_buffer, long bucket_idx) {
    std::lock_guard<std::mutex> lock(_buckets_mutex[bucket_idx]);
    _buckets[bucket_idx].emplace_back(cl_buffer);
}

std::size_t CLBufferPool::num_available_blocks() const {
    std::size_t result = 0;
    for (auto const &bucket: _buckets) {
        result += bucket.size();
    }
    return result;
}

MultiArrayRef
CLBufferPool::make_array(Shape shape, ArrayType dtype) {
    return MultiArray::make(own_reference(), shape, dtype);
}

std::shared_ptr<CLBufferPool> CLBufferPool::own_reference() {
    return _memory_manager->buffer_pool(_device_index);
}

bool CLBufferPool::queue_support_ooo_execution() const {
    return _memory_manager->device_info(_device_index).supports_out_of_order_execution;
}


} // namespace
