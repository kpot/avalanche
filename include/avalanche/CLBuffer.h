#ifndef AVALANCHE_CLBUFFER_H
#define AVALANCHE_CLBUFFER_H

#include <future>

#include "CL_cust/cl2.hpp"

namespace avalanche {

class CLBufferPool;

/**
 * This class is not thread-safe!
 */
class CLBuffer {
public:
    friend class CLBufferPool;
    ~CLBuffer();
    std::size_t device_index() const;
    std::size_t byte_size() const { return _size; }
    std::size_t capacity() const { return 1 << _bucket; }
    const cl::Buffer& cl_buffer_unsafe() const { return _cl_buffer; }
    /**
     * Makes sure that the completion event (if any) has happened its callback
     * function has been called. Does nothing if the buffer has no such event.
     */
    void wait_until_ready() const;
    const cl::Buffer& cl_buffer_when_ready() const;
    std::shared_ptr<CLBufferPool> pool() { return _pool; }
    const cl::CommandQueue& pool_queue() const;
    void set_label(const std::string &label) { _label = label; }
    void set_label(const char *func, int line) {
        _label = std::string(func) + ":" + std::to_string(line);
    }
    void set_label(const std::string &func, int line) {
        _label = func + ":" + std::to_string(line);
    }

    template <typename Vector>
    const cl::Event& write_from_vector(const Vector &v,
                                       std::size_t destination_offset) {
        const std::size_t vector_byte_size = (
            sizeof(typename Vector::value_type) * v.size());
        return write_data(
            v.data(), vector_byte_size,
            sizeof(typename Vector::value_type) * destination_offset);
    }

    const cl::Event& write_data(const void *data, std::size_t bytes_to_write,
                                std::size_t dest_offset_in_bytes);

    /**
     * See `read_into_vector`
     */
    cl::Event read_data(
        void *buffer,
        std::size_t bytes_to_read,
        std::size_t source_offset_in_bytes,
        const std::vector<cl::Event> *wait_for_events = nullptr) const;

    /**
     * Asynchronously reads all data from the buffer into a vector, once
     * all events listed in wait_for_events have happened.
     * Make sure you wait until the copying is done before using the vector!
     * You can do this by calling event.wait() on the event returned by this
     * function.
     * Please note that unlike write_from_vector(...) method, this method
     * doesn't assign the operation's result as a completion event for
     * the buffer.
     *
     * @tparam Vector - the type of the vector
     * @param v - the vector that needs to be filled. It's going to be resized
     *      to fit all the data from the buffer.
     * @param source_offset start reading from the offset in elements
     *    (not bytes!) from the beginning of the buffer
     * @param wait_for_events - the list of events we need to wait before
     *      we can try to read the data.
     * @return A new event, indicating when the copying is done.
     */
    template <typename Vector>
    cl::Event read_into_vector(Vector &v, std::size_t source_offset, std::size_t num_elements,
                                   const std::vector<cl::Event> *wait_for_events) const {
        const auto bytes_to_read = sizeof(typename Vector::value_type) * num_elements;
        v.resize(num_elements);
        return read_data(v.data(), bytes_to_read,
                         source_offset * sizeof(typename Vector::value_type),
                         wait_for_events);
    }

    void set_completion_event(const cl::Event &event);
    cl::Event& completion_event() { return _ready_event; }
    void add_dependencies(
        const std::vector<std::shared_ptr<CLBuffer>> &dependencies);
    void clear_dependencies() { _dependencies.clear(); }

private:
    cl::Event _ready_event;
    const long _bucket;
    std::shared_ptr<CLBufferPool> _pool;
    const std::size_t _size;
    cl::Buffer _cl_buffer;
    std::vector<std::shared_ptr<CLBuffer>> _dependencies;
    // There's no ways to mark the buffer as ready other than by assigning
    // an OpenCL "completion event" to it. When the event happens,
    // this promise will be set. If the event comes with a negative status
    // (an error), it will be thrown as an exception through
    // the same std::promise object.
    std::promise<void> _ready_promise;
    std::shared_future<void> _is_ready;
    std::string _label;

    CLBuffer(const std::shared_ptr<CLBufferPool> &pool,
             cl::Buffer&& _cl_buffer,
             std::size_t size, long bucket);

    friend void buffer_event_occurred(cl_event event, cl_int status, void *user_data);
};

using CLBufferRef = std::shared_ptr<CLBuffer>;

} // namespace

#endif //AVALANCHE_CLBUFFER_H
