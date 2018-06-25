#ifndef AVALANCHE_CONTEXT_H
#define AVALANCHE_CONTEXT_H

#include <map>
#include <memory>

#include "avalanche/MultiArray.h"
#include "avalanche/BaseNode.h"

namespace avalanche {

class Context;

using ContextRef = std::shared_ptr<Context>;


/**
 * Context it's an environment consisting of multiple initialized GPU arrays,
 * each attached to a computational graph node. This allows to keep the data
 * (such as variables and constants) around during and between the computations.
 * Without the Context, those variables and constants don't really exist.
 *
 * This makes the class the closest analog to TensorFlow's sessions,
 * although `Context` doesn't run the computation itself (see `Executor`).
 * And it doesn't store various temporary data during the computation
 * (this is done by `ExecutionCache`).
 *
 * Each Context is strictly associated with only one GPU.
 */
class Context : private std::map<NodeId, MultiArrayRef> {
public:

    void init(const NodeRef &node, const MultiArrayRef& array);
    void init(const NodeId node_id, const MultiArrayRef& array);

    template <typename T>
    MultiArrayRef init(const NodeRef &node,
                       const std::vector<T> &data) {
        return init(node, data, node->shape());
    }

    template <typename T>
    MultiArrayRef init(const NodeRef &node,
                       const std::vector<T> &data,
                       const Shape &shape) {
        // Note: having shape as an argument is necessary since the node
        // may not have a fully defined shape assigned
        check_data_shape_compatibility(shape, node->shape());
        auto array = device_pool()->make_array(
            shape, dtype_of_static_type<T>);
        auto writing_is_done = array->buffer_unsafe()->write_from_vector(data, 0);
        init(node, array);
        writing_is_done.wait();
        return array;
    }

    template <typename T>
    MultiArrayRef init(const NodeRef &node,
                       const std::vector<T> &data,
                       const std::initializer_list<ShapeDim> shape) {
        return init(node, data, Shape(shape));
    }

    MultiArrayRef init(const NodeRef &node, const void *data,
                       std::size_t num_bytes, ArrayType array_type,
                       const Shape &shape);

    bool get(NodeId node_id, MultiArrayRef &result) const;

    // Works like get() but throws exception if the node wasn't found
    MultiArrayRef eval(const NodeRef &node) const;

    BufferPoolRef device_pool() { return _buffer_pool; };

    static ContextRef make(BufferPoolRef buffer_pool) {
        return std::shared_ptr<Context>(new Context(buffer_pool));
    }

    static ContextRef make_for_device(DeviceIndex device_idx) {
        return std::shared_ptr<Context>(
            new Context(
                CLMemoryManager::get_default()->buffer_pool(device_idx)));
    }

private:
    BufferPoolRef _buffer_pool;

    Context(BufferPoolRef buffer_pool)
        :std::map<NodeId, MultiArrayRef>(),
         _buffer_pool{buffer_pool}{}

    void check_multi_array_compatibility(const MultiArrayRef &array) const;
    void check_data_shape_compatibility(const Shape &data_shape,
                                        const Shape &node_shape) const;
};


} // namespace


#endif //AVALANCHE_CONTEXT_H
