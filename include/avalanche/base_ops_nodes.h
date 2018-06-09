//
// Created by Kirill on 04/02/18.
//

#ifndef AVALANCHE_BASE_OPS_NODES_H
#define AVALANCHE_BASE_OPS_NODES_H

#include "avalanche/BaseNode.h"
#include "avalanche/Context.h"
#include "ExecutionCache.h"

namespace avalanche {

template <typename Op>
class UnaryOp : public BaseNode {
public:
    const NodeRef input;
    const Op op;

    template <typename... Args>
    UnaryOp(NodeRef input, Args&&... args)
        :input{input},
         op(input, std::forward<Args>(args)...) {
        set_shape(op.shape());
        set_dtype(op.dtype());
    }

    MultiArrayRef eval(Context &context, ExecutionCache &cache) const override {
        // Here you need to take into account the reference count
        MultiArrayRef result;
        if (!cache.get(id, result)) {
            result = op.forward(input->eval(context, cache));
            cache.put(id, result);
        }
        return result;
    }

    std::string to_string() const override {
        std::string output = op.lh_name() + input->to_string() + op.rh_name();
        return output;
    }

    NodeRefList inputs() const override {
        return NodeRefList({input});
    }

    const NodeRef
    apply_chain_rule(const NodeRef &wrt_input, const NodeRef &d_target_wrt_this,
                     const NodeRefList &all_inputs) const override {
        return op.apply_chain_rule(wrt_input, d_target_wrt_this, all_inputs);
    }
};


template <typename Op>
class BinaryOp : public BaseNode {
public:
    const NodeRef left;
    const NodeRef right;
    const Op op;

    template <class... Args>
    BinaryOp(const NodeRef &node1, const NodeRef &node2, Args&&... args)
        : left{node1}, right{node2},
          op{node1, node2, std::forward<Args>(args)...}
    {
        set_shape(op.shape());
        set_dtype(op.dtype());
    }

    MultiArrayRef eval(Context &context, ExecutionCache &cache) const override {
        MultiArrayRef result;
        if (!cache.get(id, result)) {
            result = op.forward(left->eval(context, cache),
                                right->eval(context, cache));
            cache.put(id, result);
        }
        return result;
    }

    std::string to_string() const override {
        std::string output;
        output += "(";
        output += left->to_string() + " " + op.name() + " " + right->to_string();
        output += ")";
        return output;
    }

    NodeRefList inputs() const override {
        return NodeRefList({left, right});
    }

    const NodeRef
    apply_chain_rule(const NodeRef &wrt_input, const NodeRef &d_target_wrt_this,
                     const NodeRefList &all_inputs) const override {
        return op.apply_chain_rule(wrt_input, d_target_wrt_this, all_inputs);
    }
};


template<typename Op>
inline NodeRef F(const NodeRef &node) {
    return std::static_pointer_cast<BaseNode>(
        std::make_shared<UnaryOp<Op>>(node));
}

template<typename Op, class... Args>
inline NodeRef FU(const NodeRef &node, Args &&... args) {
    return std::static_pointer_cast<BaseNode>(
        std::make_shared<UnaryOp<Op>>(node, std::forward<Args>(args)...));
}

template<typename Op, class... Args>
inline NodeRef F(const NodeRef &a, const NodeRef &b, Args&&... args) {
    return std::static_pointer_cast<BaseNode>(
        std::make_shared<BinaryOp<Op>>(a, b, std::forward<Args>(args)...));
}


} // namespace

#endif //AVALANCHE_BASE_OPS_NODES_H
