#include <string>
#include <sstream>

#include "avalanche/CLMemoryManager.h"
#include "avalanche/MultiArray.h"

namespace avalanche {

std::string MultiArray::to_string() {
    std::ostringstream result;
    result << "MultiArray(device" << _buffer->device_index() << ", "
           << _shape.to_string() << ", " << array_type_name(_dtype) << ")";
    return result.str();
}

void MultiArray::set_completion_event(cl::Event &event) {
    _buffer->set_completion_event(event);
}

void MultiArray::set_completion_event(cl_event event) {
    cl::Event e(event);
    set_completion_event(e);
}

MultiArrayRef MultiArray::ref_copy() {
    auto result = MultiArray::make(_buffer->pool(), _shape, _dtype);
    cl::Event ready_event;
    _buffer->pool()->cl_queue().enqueueCopyBuffer(
        // waits until current buffer is ready
        _buffer->cl_buffer_when_ready(),
        result->_buffer->cl_buffer_unsafe(),
        0, 0,
        _buffer->byte_size(),
        nullptr, &ready_event);
    result->set_completion_event(ready_event);
    return result;
}

void MultiArray::add_dependencies(
        std::initializer_list<CLBufferRef> dependencies) {
    _buffer->add_dependencies(dependencies);
}

void
MultiArray::add_dependencies(const std::vector<MultiArrayRef> &dependencies) {

    std::vector<CLBufferRef> buffers;
    for (auto const &dep: dependencies) {
        buffers.push_back(dep->_buffer);
    }
    _buffer->add_dependencies(buffers);
}


} // namespace
