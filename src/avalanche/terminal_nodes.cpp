#include <cstdint>
#include <cstring>
// FIXME: cleanup
#include <iostream>

#include <clblast.h>
#include <fmt/format.h>

#include "avalanche/MultiArray.h"
#include "avalanche/terminal_nodes.h"
#include "avalanche/macroses.h"
#include "avalanche/casting.h"

namespace avalanche {

template <typename T>
void fill_array_with_value(cl::CommandQueue &queue,
                           MultiArrayRef &array,
                           float value) {
    // FIXME: cleanup
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
    Initializer initializer = [shape, dtype, value](Context &context, ExecutionCache &cache) {
        auto result = context.device_pool()->make_array(shape, dtype);
        auto queue = result->buffer_unsafe()->pool()->cl_queue();
        fill_array_switch(dtype, queue, result, value);
        return result;
    };
    return std::static_pointer_cast<BaseNode>(
        std::make_shared<Constant>(
            (std::string("Fill ") + shape.to_string() +
                " with " + std::to_string(value)),
            initializer, shape, dtype));
}

template <typename T>
void check_compatibility(const BaseNode *node, T other_thing) {
    if (other_thing->dtype() != node->dtype()) {
        throw std::invalid_argument(
            fmt::format("Initializer for node {} has "
                        "an incompatible data type {}",
                        node->repr(), array_type_name(other_thing->dtype())));
    }
    if (other_thing->shape() != node->shape()) {
        throw std::invalid_argument(
            fmt::format(
                "Initializer for node {} has an incompatible shape {}",
                node->repr(), other_thing->shape().to_string()));
    }
}

MultiArrayRef Constant::eval(Context &context, ExecutionCache &cache) const {
    MultiArrayRef cached_value;
    if (!context.get(id, cached_value)) {
        cached_value = _initializer(context, cache);
        check_compatibility(this, cached_value);
        cached_value->set_label(to_string());
        context.init(id, cached_value);
        // FIXME: cleanup
        std::cout << "Constant " << this << " is now initialized" << std::endl;
    }
    return cached_value;
}

const NodeRef
Constant::tensor(const std::string &name,
                 const void *data,
                 std::size_t num_bytes,
                 ArrayType dtype,
                 const Shape &shape) {
    std::vector<std::uint8_t> copy_of_data(num_bytes);
    std::memcpy(copy_of_data.data(), data, num_bytes);
    return std::make_shared<Constant>(
        name,
        [shape, copy_of_data, dtype](Context &context,
                                     ExecutionCache &cache) -> MultiArrayRef {
            auto result = context.device_pool()->make_array(shape, dtype);
            // write_from_vector will update the result's completion event
            result->buffer_unsafe()->write_from_vector(copy_of_data);
            return result;
        },
        shape,
        dtype);
}

MultiArrayRef Variable::eval(Context &context, ExecutionCache &cache) const {
    MultiArrayRef cached_value;
    if (!context.get(id, cached_value)) {
        if (_initializer) {
            cached_value = _initializer(context, cache);
            check_compatibility(this, cached_value);
            cached_value->set_label(to_string());
            context.init(id, cached_value);
            // FIXME: cleanup
            std::cout << "Variable " << this << " is now initialized"
                      << std::endl;
        } else {
            throw std::runtime_error(
                "Cannot find an initial value for variable " + name);
        }
    }
    return cached_value;
}

NodeRef
Variable::make_from_node(const std::string &name,
                         const NodeRef &initialize_from_node) {
    if (!initialize_from_node) {
        throw std::invalid_argument("Cannot initialize from a nullptr node");
    }
    Variable *new_node_ptr = new Variable(
        name, node_initializer(initialize_from_node),
        initialize_from_node->shape(), initialize_from_node->dtype());
    return std::static_pointer_cast<BaseNode>(
        std::shared_ptr<Variable>(new_node_ptr));
}

NodeRef
Variable::make(const std::string &name, const std::vector<ShapeDim> &shape_dims,
               ArrayType dtype, Initializer initializer) {
    Variable *raw_ptr = new Variable(
        name, std::move(initializer), Shape(shape_dims), dtype);
    return std::static_pointer_cast<BaseNode>(
        std::shared_ptr<Variable>(raw_ptr));
}

Initializer node_initializer(const NodeRef &node) {
    if (node) {
        Initializer initializer = [node](Context &context,
                                         ExecutionCache &cache) {
            // the values used for initialization are not cached, just like in
            // TF where initialization is a separate step. Doing it otherwise
            // (with caching) would have greatly complicated everything.
            return node->eval(context, cache);
        };
        return initializer;
    }
    return nullptr;
}
} // namespace
