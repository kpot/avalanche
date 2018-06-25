//
// Simple templates allowing quickly construct many similarly-behaving
// computational nodes from any class without the need to directly inherit
// it from `BaseNode`. This ensures more generalized behaviour and allows
// for some tricks like making some methods static if necessary, building
// complex class hierarchies, building nodes from structs, etc.
//

#ifndef AVALANCHE_BASE_OPS_NODES_H
#define AVALANCHE_BASE_OPS_NODES_H

#include "avalanche/BaseNode.h"
#include "avalanche/Context.h"
#include "avalanche/ExecutionCache.h"

namespace avalanche {

/**
 * SFINAE trick checking if operation has `repr_extra` method
 */
template <typename T>
class has_repr_extra_method
{
    typedef char one;
    typedef long two;

    template <typename C> static one test( typeof(&C::repr_extra) ) ;
    template <typename C> static two test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

/**
 * SFINAE trick calling operation`s `repr_extra` method only if it's present
 * This allows to simplify the descriptions of the operations.
 */
template <typename T>
typename std::enable_if<!has_repr_extra_method<T>::value, std::string>::type
get_op_repr_extra(const T &a) {
    return "";
}

template <typename T>
typename std::enable_if<has_repr_extra_method<T>::value, std::string>::type
get_op_repr_extra(const T &a) {
    return a.repr_extra();
}


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

    std::string repr() const override {
        return format_repr(typeid(op).name(), "", get_op_repr_extra(op));
    }

    NodeRefList inputs() const override {
        return NodeRefList({input});
    }

    const NodeRef
    apply_chain_rule(const NodeRef &wrt_input, const NodeRef &d_target_wrt_this,
                     const NodeRefList &all_inputs) const override {
        return op.apply_chain_rule(wrt_input, d_target_wrt_this, all_inputs);
    }

    bool use_in_back_propagation() const override {
        return op.use_in_back_propagation();
    };
};


template <typename Op>
class OpListsInputs {
    template <typename C> static char test(typeof(&C::inputs)) { return 0; }
    template <typename C> static long test(...) { return 0; }
public:
    enum {value = (sizeof(test<Op>(0) == sizeof(char)))};
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

    std::string repr() const override {
        return format_repr(typeid(op).name(), "", get_op_repr_extra(op));
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
