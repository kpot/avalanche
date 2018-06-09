#include <iostream>
#include <algorithm>

#include "CL_cust/cl2.hpp"
#include <clblast.h>
#include <avalanche/CLBufferPool.h>

#include "avalanche/CodeCache.h"
#include "avalanche/nodes.h"
#include "avalanche/opencl_utils.h"

constexpr std::size_t LARGE_VECTOR_SIZE = 125;

template <typename T>
class SpeedChecker {
public:
    std::vector<T> data1;
    std::vector<T> data2;
    std::vector<T> data3;
    cl::Buffer buffer1;
    cl::Buffer buffer2;
    cl::Buffer buffer3;
    cl::CommandQueue queue;
    cl::Context context;
    std::size_t vector_size;

    SpeedChecker(std::size_t vector_size, cl::Context &context, cl::CommandQueue &queue)
    :buffer1(context, CL_MEM_READ_WRITE, vector_size * sizeof(T)),
     buffer2(context, CL_MEM_READ_WRITE, vector_size * sizeof(T)),
     buffer3(context, CL_MEM_READ_WRITE, vector_size * sizeof(T)),
     context{context},
     queue{queue},
     vector_size{vector_size}
    {
    }

    void upload_data() {
        data1.resize(vector_size);
        data2.resize(vector_size);
        data3.resize(vector_size + 2);
        for (std::size_t i = 0; i < vector_size; ++i) {
            data1[i] = i;
            data2[i] = i;
            data3[i] = 0;
        }
        std::vector<cl::Event> events(2);
        queue.enqueueWriteBuffer(buffer1, CL_FALSE, 0, vector_size * sizeof(T), data1.data(), nullptr, &events[0]);
        queue.enqueueWriteBuffer(buffer2, CL_FALSE, 0, vector_size * sizeof(T), data2.data(), nullptr, &events[1]);
        cl::Event::waitForEvents(events);
    }

    void check_clblast_axpy() {
        cl_event ll_event = nullptr;
        cl_command_queue ll_queue = queue();
        clblast::Axpy<T>(vector_size, 1.0, buffer1(), 0, 1, buffer2(), 0, 1, &ll_queue, &ll_event);
        cl::Event event(ll_event);
        event.wait();
    }

    void download_data() {
        std::vector<cl::Event> events(3);
        queue.enqueueReadBuffer(buffer1, CL_FALSE, 0, vector_size * sizeof(T), data1.data(), nullptr, &events[0]);
        queue.enqueueReadBuffer(buffer2, CL_FALSE, 0, vector_size * sizeof(T), data2.data(), nullptr, &events[1]);
        queue.enqueueReadBuffer(buffer3, CL_FALSE, 0, vector_size * sizeof(T), data3.data(), nullptr, &events[2]);
        cl::Event::waitForEvents(events);
    }

    void check_custom_axpy() {
        std::vector<cl_ulong> left_size_mask, right_size_mask, result_sub_sizes;
        avalanche::Shape data_shape(
            {static_cast<avalanche::ShapeDim>(vector_size)});
        avalanche::broadcast_size_masks(
            data_shape, data_shape,
            left_size_mask, right_size_mask, result_sub_sizes);
        cl::Buffer left_mask_buffer = avalanche::create_buffer_for_vector(
            context, left_size_mask, CL_MEM_READ_WRITE);
        cl::Buffer right_mask_buffer = avalanche::create_buffer_for_vector(
            context, right_size_mask, CL_MEM_READ_WRITE);
        cl::Buffer result_sizes_buffer = avalanche::create_buffer_for_vector(
            context, result_sub_sizes, CL_MEM_READ_WRITE);
        cl::vector<cl::Event> wait_for_data = {
            avalanche::write_from_vector(
                left_mask_buffer, queue, left_size_mask),
            avalanche::write_from_vector(
                right_mask_buffer, queue, right_size_mask),
            avalanche::write_from_vector(
                result_sizes_buffer, queue, result_sub_sizes)
        };
//        cl::Event answer_is_ready = avalanche::call_broadcasted_kernel(
//            queue,
//            "plus",
//            avalanche::ArrayType::float32,
//            data_shape,
//            buffer1,
//            buffer2,
//            buffer3,
//            left_mask_buffer,
//            right_mask_buffer,
//            result_sizes_buffer,
//            wait_for_data);
//        queue.flush();
//        answer_is_ready.wait();
    }
};

int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Context context(devices);
    std::string device_name;
    devices[0].getInfo(CL_DEVICE_NAME, &device_name);
    cl::CommandQueue queue(
        context, devices[0],
        (device_name == "Iris" ? 0 : CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE));
    SpeedChecker<float> checker(LARGE_VECTOR_SIZE, context, queue);
    checker.upload_data();
    for (int i = 0; i < 10; ++i) {
//        checker.check_clblast_axpy();
        checker.check_custom_axpy();
    }
    checker.download_data();
    std::cout << "Head: ";
    for (std::size_t i = 0; i < std::min<std::size_t>(checker.vector_size, 100); ++i) {
        std::cout << checker.data3[i] << ", ";
    }
    std::cout << "\nTail: ";
    for (long i = std::max<long>(checker.vector_size - 10, 0); i < checker.data3.size(); ++i) {
        std::cout << checker.data3[i] << ", ";
    }
    return 0;
}