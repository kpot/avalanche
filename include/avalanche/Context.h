#ifndef AVALANCHE_CONTEXT_H
#define AVALANCHE_CONTEXT_H

#include <map>
#include <memory>

#include "avalanche/MultiArray.h"
#include "avalanche/BaseNode.h"

namespace avalanche {

class Context;

using ContextRef = std::shared_ptr<Context>;

class Context : private std::map<NodeId, MultiArrayRef> {
public:

    void init(const NodeRef &node, const MultiArrayRef& array);
    void init(const NodeId node_id, const MultiArrayRef& array);

    // FIXME: Why can't you just take the shape from the node?
    template <typename T>
    MultiArrayRef init(const NodeRef &node,
                       const std::vector<T> &data,
                       const Shape &shape) {
        auto array = device_pool()->make_array(
            shape, dtype_of_static_type<T>);
        auto writing_is_done = array->buffer_unsafe()->write_from_vector(data);
        init(node, array);
        writing_is_done.wait();
        return array;
    }
    // TODO: add init working with low-level data without types, for which we know only the size of the element
    template <typename T>
    MultiArrayRef init(const NodeRef &node,
                       const std::vector<T> &data,
                       const std::initializer_list<ShapeDim> shape) {
        return init(node, data, Shape(shape));
    }

    MultiArrayRef init(const NodeRef &node, const void *data,
                       std::size_t num_bytes, ArrayType array_type,
                       const Shape &shape);

    bool get(NodeId node_id, MultiArrayRef &result) const {
        auto cached = find(node_id);
        if (cached != this->end()) {
            result = cached->second;
            return true;
        }
        result = nullptr;
        return false;
    }

    // Works like get() but throws exception if node wasn't found
    MultiArrayRef eval(const NodeRef &node) const {
        MultiArrayRef result;
        if (!get(node->id, result)) {
            throw std::invalid_argument(
                "Given graph node has not been initialized yet");
        }
        return result;
    }

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

    void check_multi_array_compatibility(const MultiArrayRef &array);
};


} // namespace


#endif //AVALANCHE_CONTEXT_H
