#ifndef AVALANCHE_OPENCL_UTILS_H
#define AVALANCHE_OPENCL_UTILS_H


#include "CL_cust/cl2.hpp"

namespace avalanche {

const char *get_opencl_error_string(int error);

cl::Device get_device_from_queue(const cl::CommandQueue &queue);

cl::Context get_context_from_queue(const cl::CommandQueue &queue);

template <typename Vector>
const cl::Buffer create_buffer_for_vector(
        const cl::Context &context,
        const Vector &v,
        cl_mem_flags flags=CL_MEM_READ_WRITE) {
    cl::Buffer new_buffer(
        context, flags, sizeof(typename Vector::value_type) * v.size());
    return new_buffer;
}

template <typename Vector>
const cl::Event write_from_vector(
        const cl::Buffer &buffer,
        const cl::CommandQueue &queue,
        const Vector &v) {
    cl::Event ready_event;
    queue.enqueueWriteBuffer(
        buffer, CL_FALSE, 0,
        sizeof(typename Vector::value_type) * v.size(), v.data(),
        nullptr, &ready_event);
    return ready_event;
}


template <typename Vector>
const cl::Event read_into_vector(
        const cl::Buffer &buffer,
        const cl::CommandQueue &queue,
        Vector &v,
        const std::size_t items_to_read,
        const std::vector<cl::Event> &wait_for_events) {
    std::size_t items_to_fill = items_to_read;
    v.resize(items_to_read);
    cl::Event ready_event;
    queue.enqueueReadBuffer(
        buffer, CL_FALSE, 0,
        sizeof(typename Vector::value_type) * v.size(), v.data(),
        wait_for_events.empty() ? nullptr : &wait_for_events,
        &ready_event);
    return ready_event;
}

inline std::vector<cl::Event> make_event_list(
        std::initializer_list<cl::Event> events) {
    std::vector<cl::Event> result;
    result.reserve(events.size());
    for (auto e: events) {
        if (e.get() != nullptr) {
            result.push_back(e);
        }
    }
    return result;
}

constexpr std::size_t make_divisible_by(std::size_t factor, std::size_t x) {
    return (x / factor) * factor + ((x % factor) == 0 ? 0 : factor);
}



} // namespace

#endif //AVALANCHE_OPENCL_UTILS_H
