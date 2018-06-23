#include <fmt/format.h>

#include "avalanche/BaseNode.h"
#include "avalanche/Context.h"

namespace avalanche {

void Context::init(const NodeRef &node,
                   const std::shared_ptr<MultiArray> &array) {
    if (array->dtype() != node->dtype()) {
        throw std::invalid_argument(
            "Given MultiArray instance has different "
            "data type thant that of the node.");
    }
    check_data_shape_compatibility(array->shape(), node->shape());
    init(node->id, array);
}

void Context::init(const NodeId node_id,
                   const MultiArrayRef &array) {
    check_multi_array_compatibility(array);
    operator[](node_id) = array;
}

void Context::check_multi_array_compatibility(
        const MultiArrayRef &array) const {
    if (array->buffer_unsafe()->pool() != _buffer_pool) {
        throw std::invalid_argument(
            "MultiArray and the Context cannot be linked to different"
            " devices or contexts");
    }
}

MultiArrayRef
Context::init(const NodeRef &node, const void *data, std::size_t num_bytes,
              ArrayType array_type, const Shape &shape) {
    check_data_shape_compatibility(shape, node->shape());
    auto array = device_pool()->make_array(shape, array_type);
    auto writing_is_done = (
        array->buffer_unsafe()->write_data(data, num_bytes, 0));
    init(node, array);
    writing_is_done.wait();
    return array;
}

void Context::check_data_shape_compatibility(const Shape &data_shape,
                                             const Shape &node_shape) const {
    if (!data_shape.agrees_with(node_shape)) {
        throw std::invalid_argument(
            fmt::format("The shape of the data {} is incompatible "
                        "with the shape of the node {}",
                        data_shape.to_string(),
                        node_shape.to_string()));
    }
}

MultiArrayRef Context::eval(const NodeRef &node) const {
    MultiArrayRef result;
    if (!get(node->id, result)) {
        throw std::invalid_argument(
            fmt::format("Given graph node ({}) has not been initialized yet",
                        node->repr()));
    }
    return result;
}

bool Context::get(NodeId node_id, MultiArrayRef &result) const {
    auto cached = find(node_id);
    if (cached != this->end()) {
        result = cached->second;
        return true;
    }
    result = nullptr;
    return false;
}


} // namespace
