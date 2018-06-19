#include <cstdint>
#include <cstring>

#include <clblast.h>

#include "avalanche/base_ops_nodes.h"
#include "avalanche/MultiArray.h"
#include "avalanche/macroses.h"
#include "avalanche/casting.h"
#include "avalanche/logging.h"

#include "avalanche/terminal_nodes.h"

namespace avalanche {

template <typename T>
void fill_array_with_value(cl::CommandQueue &queue,
                           MultiArrayRef &array,
                           float value) {
    get_logger()->debug(
        "Filling buffer {}",
        reinterpret_cast<void*>(array->buffer_unsafe().get()));
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
    if (!shape.is_complete()) {
        throw std::invalid_argument("The shape must be fully defined");
    }
    Initializer initializer{
        [shape, dtype, value](Context &context, ExecutionCache &cache) {
            auto result = context.device_pool()->make_array(shape, dtype);
            auto queue = result->buffer_unsafe()->pool()->cl_queue();
            fill_array_switch(dtype, queue, result, value);
            return result;
        },
        dtype
    };
    return std::static_pointer_cast<BaseNode>(
        std::make_shared<Constant>(
            (std::string("Fill ") + shape.to_string() +
                " with " + std::to_string(value)),
            initializer, shape, dtype, NodeRefList()));
}

template <typename T>
void check_compatibility(const BaseNode *node, T other_thing) {
    if (other_thing->dtype() != node->dtype()) {
        throw std::invalid_argument(
            fmt::format("Initializer for node {} has "
                        "an incompatible data type {}",
                        node->repr(), array_type_name(other_thing->dtype())));
    }
    if (!other_thing->shape().agrees_with(node->shape())) {
        throw std::invalid_argument(
            fmt::format(
                "Initializer for node {} has an incompatible shape {}",
                node->repr(), other_thing->shape().to_string()));
    }
}

MultiArrayRef Constant::eval(Context &context, ExecutionCache &cache) const {
    MultiArrayRef cached_value;
    if (!context.get(id, cached_value)) {
        cached_value = _initializer.code(context, cache);
        check_compatibility(this, cached_value);
        cached_value->set_label(to_string());
        if (_dependencies.empty()) {
            // TODO: now we can safely cache the constant only if it doesn't depend from anything else
            context.init(id, cached_value);
        }
        get_logger()->debug(
            "Constant {} is now initialized", reinterpret_cast<const void*>(this));
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
    Initializer initializer {
        [shape, copy_of_data, dtype](Context &context,
                                     ExecutionCache &cache) -> MultiArrayRef {
            auto result = context.device_pool()->make_array(shape, dtype);
            // write_from_vector will update the result's completion event
            result->buffer_unsafe()->write_from_vector(copy_of_data, 0);
            return result;
        },
        dtype
    };
    return std::make_shared<Constant>(
        name,
        initializer,
        shape,
        dtype, NodeRefList());
}

const NodeRef Constant::fill_shape(const avalanche::NodeRef &shape_node,
                                   ArrayType dtype,
                                   float value) {
    if (shape_node->dtype() != ShapeOf::dtype()) {
        throw std::invalid_argument(
            fmt::format("Given node has data type {} while {} is required",
                        array_type_name(shape_node->dtype()),
                        array_type_name(ShapeOf::dtype())));
    }
    if (shape_node->shape().rank() > 1) {
        throw std::invalid_argument(
            fmt::format(
                "Shape node must output not more than 1-D vector. "
                "Currently it's {}-D",
                shape_node->shape().rank()));
    }
    Initializer initializer{
        [shape_node, value, dtype](Context &context, ExecutionCache &cache) {
            auto shape_array = shape_node->eval(context, cache);
            std::vector<ShapeDim> shape_dims;
            shape_array->fetch_data_into(shape_dims);

            auto result = context.device_pool()->make_array(shape_dims, dtype);
            auto queue = result->buffer_unsafe()->pool()->cl_queue();
            fill_array_switch(dtype, queue, result, value);
            return result;
        },
        dtype
    };
    // We don't know the shape at this stage, but we know it's rank at least
    std::vector<ShapeDim> proto_dims(
        static_cast<std::size_t>(
            shape_node->shape().rank() == 1 ? shape_node->shape().dim(0) : 0));
    std::fill(proto_dims.begin(), proto_dims.end(), UnknownDim);
    NodeRefList dependencies({F<NoBackProp>(shape_node)});
    return std::static_pointer_cast<BaseNode>(
        std::make_shared<Constant>(
            fmt::format("Fill shape {} with {}", shape_node->repr(), value),
            initializer, proto_dims, dtype, dependencies));
}

const NodeRef Constant::fill_like(const NodeRef &other_node, float value) {
    return fill_like_with_type(other_node, other_node->dtype(), value);
}

const NodeRef
Constant::fill_like_with_type(const NodeRef &other_node, ArrayType dtype,
                              float value) {
    Initializer initializer {
        [value, other_node, dtype](Context &context, ExecutionCache &cache) {
            auto real_value = other_node->eval(context, cache);
            auto result = context.device_pool()->make_array(
                real_value->shape(), dtype);
            auto queue = result->buffer_unsafe()->pool()->cl_queue();
            fill_array_switch(dtype, queue, result, value);
            return result;
        },
        dtype,
    };
    return std::static_pointer_cast<BaseNode>(
        std::make_shared<Constant>(
            fmt::format("Fill shape like {} with {}",
                        other_node->repr(), value),
            initializer, other_node->shape(),
            dtype,
            NodeRefList({F<NoBackProp>(other_node)})));
}

const NodeRef
Constant::zeros_like_with_type(const NodeRef &other_node, ArrayType dtype) {
    return fill_like_with_type(other_node, dtype, 0);
}

MultiArrayRef Variable::eval(Context &context, ExecutionCache &cache) const {
    MultiArrayRef cached_value;
    if (!context.get(id, cached_value)) {
        if (_initializer) {
            cached_value = _initializer.code(context, cache);
            check_compatibility(this, cached_value);
            cached_value->set_label(to_string());
            context.init(id, cached_value);
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
    if (initializer) {
       if (dtype != initializer.dtype)  {
           throw std::invalid_argument(
               fmt::format(
                   "Given initializer's type ({}) is incompatible "
                   "with the variable's type {}",
                   array_type_name(initializer.dtype), array_type_name(dtype)));
       }
    }
    Variable *raw_ptr = new Variable(
        name, std::move(initializer), Shape(shape_dims), dtype);
    return std::static_pointer_cast<BaseNode>(
        std::shared_ptr<Variable>(raw_ptr));
}

Initializer node_initializer(const NodeRef &node) {
    if (node) {
        Initializer initializer {
            [node](Context &context,
                   ExecutionCache &cache) {
                // the values used for initialization are not cached, just like in
                // TF where initialization is a separate step. Doing it otherwise
                // (with caching) would have greatly complicated everything.
                return node->eval(context, cache);
            },
            node->dtype()};
        return initializer;
    }
    return Initializer{};
}
} // namespace
