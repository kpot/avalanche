#include "avalanche/BaseNode.h"
#include "avalanche/Context.h"

namespace avalanche {

void Context::init(const NodeRef &node,
                   const std::shared_ptr<MultiArray> &array) {
    if (array->dtype() != node->dtype()) {
        throw std::invalid_argument(
            "Given MultiArray instance has different "
            "data type that of the node.");
    }
    if (array->shape() != node->shape()) {
        throw std::invalid_argument(
            "Given MultiArray instance differs in shape from the node");
    }
    init(node->id, array);
}

void Context::init(const NodeId node_id,
                   const MultiArrayRef &array) {
    check_multi_array_compatibility(array);
    operator[](node_id) = array;
}

void Context::check_multi_array_compatibility(const MultiArrayRef &array) {
    if (array->buffer_unsafe()->pool() != _buffer_pool) {
        throw std::invalid_argument(
            "MultiArray and the Context cannot be linked to different"
            " devices or contexts");
    }
}

MultiArrayRef
Context::init(const NodeRef &node, const void *data, std::size_t num_bytes,
              ArrayType array_type, const Shape &shape) {
    auto array = device_pool()->make_array(shape, array_type);
    auto writing_is_done = (
        array->buffer_unsafe()->write_data(data, num_bytes));
    init(node, array);
    writing_is_done.wait();
    return array;
}


} // namespace
