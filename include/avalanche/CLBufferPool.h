#ifndef AVALANCHE_CLBUFFERPOOL_H
#define AVALANCHE_CLBUFFERPOOL_H

#include <memory>
#include <mutex>
#include <array>

#include "CL_cust/cl2.hpp"

#include "avalanche/ArrayType.h"
#include "avalanche/Shape.h"

namespace avalanche {

class CLBuffer;
class MultiArray;
class CLMemoryManager;

/** A pool of OpenCL buffers, strictly associated with a particular device */
// TODO: Check if the pool has been destroyed
class CLBufferPool {
public:
    // Enough for up to 1 TB of buffers
    static constexpr std::uint8_t MaxBuckets = 40;
    static constexpr std::size_t MaxBufferSize = (
        static_cast<std::size_t>(1) << MaxBuckets);
    CLBufferPool(CLMemoryManager* memory_manager,
                 std::size_t device_index,
                 const cl::Context &context,
                 const cl::CommandQueue &device_queue);
    ~CLBufferPool();

    bool is_linked_with_device(const cl::Device &device) const;


    std::shared_ptr<MultiArray> make_array(
        Shape shape, ArrayType dtype=ArrayType::float32);
    std::shared_ptr<CLBuffer> reserve_buffer(std::size_t size_in_bytes);
    template <typename Vector>
    std::shared_ptr<CLBuffer> reserve_buffer_for_vector(Vector v) {
        return reserve_buffer(sizeof(typename Vector::value_type) * v.size());
    }
    static long find_mem_slot(size_t size);
    std::size_t num_available_blocks() const;
    std::size_t num_buckets() const;
    std::size_t device_index() { return _device_index; }
    cl::CommandQueue& cl_queue() { return _device_queue; }
    const cl::CommandQueue& cl_queue() const { return _device_queue; }
    cl::Context& cl_context() { return _cl_context; }
    std::shared_ptr<CLBufferPool> own_reference();
    bool queue_support_ooo_execution() const;

private:
    CLMemoryManager *_memory_manager;
    std::size_t _device_index;
    cl::Context _cl_context;
    cl::CommandQueue _device_queue;
    const cl::Device _cl_device;
    std::array<std::vector<cl::Buffer>, MaxBuckets> _buckets;
    std::array<std::mutex, MaxBuckets> _buckets_mutex;

    friend class CLBuffer;
    void return_buffer(cl::Buffer &&cl_buffer, long bucket_idx);
};


using BufferPoolRef = std::shared_ptr<CLBufferPool>;

} // namespace

#endif //AVALANCHE_CLBUFFERPOOL_H
