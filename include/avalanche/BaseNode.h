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

    /**
     * Calculates derivative of a target node with respect to one
     * of the current node's inputs using the chain rule
     * https://en.wikipedia.org/wiki/Chain_rule
     *
     *     dz   dz   dy
     *     -- = -- * --
     *     dx   dy   dx
     *
     * @param wrt_input with respect to which input we need
     *                  the derivative (what is the "x")
     * @param d_target_wrt_this derivative of the target with respect
     *                          to the current node (dz/dy)
     * @param all_inputs all other inputs of the node,
     *                       including "x" (useful in some cases)
     */
    virtual const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const = 0;

    virtual std::string to_string() const = 0;
    virtual std::string repr() const = 0;

    virtual NodeRefList inputs() const = 0;
    /** Determines whether we need to stop back-propagation at the current node
     * or not.
     * If false is returned, this node will not be considered neither
     * as a differentiable input, nor as a consumer of a differentiable node. */
    virtual bool use_in_back_propagation() const { return true; };

    /**
     * Returns the shape of the current computational node.
     * This shape is only a suggestion, or the result of an inference,
     * since it can have "unknown holes" in it.
     * It's only purpose is to be a safeguard for validation and help with
     * debugging by letting humans see what's been inspected.
     * The final fully defined shapes can only be obtained in runtime,
     * from `MultiArray`s.
     */
    const Shape& shape() const { return _shape; }
    ArrayType dtype() const { return _dtype; }
    std::string format_repr(const std::string &operation,
                            const std::string &name,
                            const std::string &extra) const;

    std::string tree_repr();

private:
    Shape _shape;
    ArrayType _dtype;

    static NodeId new_global_id() {
        static NodeId counter = 0;
        static std::mutex mutex;
        std::lock_guard <std::mutex> lock(mutex);
        return counter++;
    }

    void _tree_repr_body(int depth, std::ostringstream &out);

protected:
    void set_shape(const Shape &shape) { _shape = shape; }
    void set_dtype(const ArrayType dtype) { _dtype = dtype; }
};

}

#endif //AVALANCHE_BASENODE_H
