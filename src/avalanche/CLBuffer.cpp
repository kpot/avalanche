#include <iostream>
#include <chrono>

#include "avalanche/logging.h"
#include "avalanche/CLBuffer.h"
#include "avalanche/CLBufferPool.h"
#include "avalanche/opencl_utils.h"


namespace avalanche {

void buffer_event_occurred(cl_event event, cl_int status, void *user_data) {
    auto buffer = static_cast<CLBuffer*>(user_data);
    if (event != buffer->_ready_event.get()) {
        auto logger = get_logger();
        logger->error("Buffer {} Received a call back "
                      "from an already cancelled event!", buffer->_label);
        return;
    }
    buffer->clear_dependencies();
    if (status < 0) {
        buffer->_ready_promise.set_exception(
            std::make_exception_ptr(
                std::runtime_error(
                    std::string("OpenCL event reported failure: ") +
                    get_opencl_error_string(status))));
    } else if (status == CL_COMPLETE) {
        buffer->_ready_promise.set_value();
    }
}

CLBuffer::CLBuffer(
    const std::shared_ptr<CLBufferPool> &pool,
    cl::Buffer&& cl_buffer,
    std::size_t size, long bucket)
    :_pool{pool}, _cl_buffer{cl_buffer}, _size{size}, _bucket{bucket},
     _ready_event{nullptr},
     _ready_promise{std::promise<void>()},
     _is_ready{_ready_promise.get_future()} {
}

void CLBuffer::set_completion_event(const cl::Event &event) {
//    std::cout << "Assigned callback" << this << std::endl;
    // the event cannot be replaced until the previous event is done
    wait_until_ready();
    // this releases any previous event, although any previously set
    // callbacks might still be called
    _ready_event = event;
    _ready_promise = std::promise<void>();
    _is_ready = std::shared_future<void>(_ready_promise.get_future());
    if (event.get() != nullptr) {
        _ready_event.setCallback(CL_COMPLETE, buffer_event_occurred, this);
    }
}

CLBuffer::~CLBuffer() {
    if (_pool) {
        // cannot be destroyed until the event is done
        wait_until_ready();
        _dependencies.clear();
        _ready_event = nullptr;
        _pool->return_buffer(std::move(_cl_buffer), _bucket);
        _cl_buffer = nullptr;
        _pool = nullptr;
    }
}

void CLBuffer::wait_until_ready() const {
    if (_ready_event.get() != nullptr) {
        // Intel OpenCL framework can be very reluctant in sending commands
        // to devices, and we should encourage it to do so before we start
        // waiting for any events to happen.
        pool_queue().flush();
        if (_is_ready.valid()) {
            _is_ready.get();  // to make sure the callback has been called too
        }
    }
}

void CLBuffer::add_dependencies(
        const std::vector<CLBufferRef> &dependencies) {
    // Dependencies should not be added if there's an event we're waiting for.
    // Because during the event the list of dependencies will be cleared, which
    // is happening from an another thread, and we don't want any conflicts here
    wait_until_ready();
    std::copy(dependencies.begin(), dependencies.end(),
              std::back_inserter(_dependencies));
}

std::size_t CLBuffer::device_index() const { return _pool->device_index(); }

const cl::CommandQueue& CLBuffer::pool_queue() const { return _pool->cl_queue(); }

const cl::Buffer& CLBuffer::cl_buffer_when_ready() const {
    wait_until_ready();
    return _cl_buffer;
}

/**
 * Be careful with writing the data because the memory buffer must remain
 * valid until the writing is done.
 */
const cl::Event &
CLBuffer::write_data(const void *data, const std::size_t bytes_to_write,
                     const std::size_t dest_offset_in_bytes) {
    if (bytes_to_write > byte_size()) {
        throw std::invalid_argument(
            "Too much data for this buffer");
    }
    cl::Event ready_event;
    pool_queue().enqueueWriteBuffer(
        _cl_buffer, CL_FALSE, dest_offset_in_bytes,
        bytes_to_write, data, nullptr, &ready_event);
    set_completion_event(ready_event);
    return _ready_event;
}

cl::Event CLBuffer::read_data(
        void *buffer,
        std::size_t bytes_to_read,
        std::size_t source_offset_in_bytes,
        const std::vector<cl::Event> *wait_for_events) const {
    if (byte_size() < bytes_to_read) {
        throw std::invalid_argument(
            fmt::format("The buffer is too small ({} bytes) "
                        "for the data requested ({} bytes)",
                        byte_size(), bytes_to_read));
    }
    cl::Event ready_event;
    pool_queue().enqueueReadBuffer(
        _cl_buffer, CL_FALSE, source_offset_in_bytes,
        bytes_to_read, buffer, wait_for_events, &ready_event);
    return ready_event;
}

} // namespace
