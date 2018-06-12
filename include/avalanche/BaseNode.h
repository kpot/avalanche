//
// Created by Kirill on 29/01/18.
//

#ifndef AVALANCHE_BASENODE_H
#define AVALANCHE_BASENODE_H

#include <memory>
#include <vector>

#include "avalanche/MultiArray.h"

namespace avalanche {

struct BaseNode;
class Context;
class ExecutionCache;

using NodeId = std::size_t;
using NodeRef = std::shared_ptr<BaseNode>;
using NodeRefList = std::vector<NodeRef>;

class BaseNode {

public:
    const NodeId id;

    BaseNode() :id{new_global_id()}, _shape{}, _dtype{ArrayType::float32} {}

    virtual MultiArrayRef eval(Context &context, ExecutionCache &cache) const = 0;

    virtual const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const = 0;

    virtual std::string to_string() const = 0;
    virtual std::string repr() const = 0;

    virtual NodeRefList inputs() const = 0;

    const Shape& shape() const { return _shape; }
    ArrayType dtype() const { return _dtype; }
    std::string format_repr(const std::string &operation,
                            const std::string &name) const;

private:
    Shape _shape;
    ArrayType _dtype;

    static NodeId new_global_id() {
        static NodeId counter = 0;
        static std::mutex mutex;
        std::lock_guard <std::mutex> lock(mutex);
        return counter++;
    }

protected:
    void set_shape(const Shape &shape) { _shape = shape; }
    void set_dtype(const ArrayType dtype) { _dtype = dtype; }
};

}

#endif //AVALANCHE_BASENODE_H
