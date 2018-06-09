#include <iostream>

#include <clblast.h>

#include "avalanche/MultiArray.h"
#include "avalanche/terminal_nodes.h"
#include "avalanche/macroses.h"
#include "avalanche/casting.h"

namespace avalanche {

template <typename T>
void fill_array_with_value(cl::CommandQueue &queue,
                           MultiArrayRef &array,
                           float value) {
    std::cout << "Filling buffer " << array->buffer_unsafe().get() << std::endl;
    cl::Event result_event;
    T casted_value = to_array_type<T>(value);
    queue.enqueueFillBuffer(
        array->cl_buffer_unsafe(), casted_value, 0,
        array->buffer_unsafe()->byte_size(),
        nullptr, &result_event);
    // Surprisingly, Apple OpenCL doesn't (always?) want to allow assignment
    // of custom callbacks to events returned from clEnqueueFillBuffer.
    // So it's better to just make it synchronous. Which is fine since we
    // use this function only for constants which are going to be cached anyway.
    result_event.wait();
    array->set_completion_event(result_event);
}

ARRAY_DTYPE_SWITCH_FUNCTION(fill_array_switch, fill_array_with_value, void,);

const NodeRef Constant::fill(Shape shape, ArrayType dtype, float value) {
    Initializer initializer = [shape, dtype, value](Context &context) {
        auto result = context.device_pool()->make_array(shape, dtype);
        auto queue = result->buffer_unsafe()->pool()->cl_queue();
        fill_array_switch(dtype, queue, result, value);
        return result;
    };
    return std::static_pointer_cast<BaseNode>(
        std::make_shared<Constant>(std::string("Fill") + std::to_string(value),
                                   initializer, shape, dtype));
}

MultiArrayRef Constant::eval(Context &context, ExecutionCache &cache) const {
    MultiArrayRef cached_value;
    if (!context.get(id, cached_value)) {
        cached_value = _initializer(context);
        cached_value->set_label(to_string());
        context.init(id, cached_value);
        std::cout << "Constant " << this << " is now initialized" << std::endl;
    }
    return cached_value;
}

#undef FILL_BUFFER

} // namespace
