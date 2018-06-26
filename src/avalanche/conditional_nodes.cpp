#include <fmt/format.h>

#include "avalanche/conditional_nodes.h"
#include "avalanche/Context.h"
#include "avalanche/ExecutionCache.h"
#include "avalanche/math_ops/comparisons.h"
#include "avalanche/math_ops/messages.h"
#include "avalanche/shape_nodes.h"

namespace avalanche {

Cond::Cond(const NodeRef &condition, const NodeRef &true_node,
           const NodeRef &false_node)
    :_cond_node{F<NoBackProp>(condition)},
     _true_node{true_node},
     _false_node{false_node}
{
    if (_cond_node->dtype() != BoolArrayType) {
        throw std::invalid_argument(
            fmt::format("The condition node type must be {}. "
                        "But its {} instead",
                        array_type_name(BoolArrayType),
                        array_type_name(_cond_node->dtype())));
    }
    if (!_cond_node->shape().is_scalar()) {
        throw std::invalid_argument(
            fmt::format("The condition node must be a scalar. "
                        "But its current shape is {} instead",
                        _cond_node->shape().to_string()));
    }
    if (_true_node->dtype() != _false_node->dtype()) {
        throw std::invalid_argument(
            fmt::format("It is inacceptable for the True and the False nodes "
                        "to have different type: {} and {}",
                        array_type_name(_true_node->dtype()),
                        array_type_name(_false_node->dtype())));
    }
    set_dtype(_true_node->dtype());
    if (_true_node->shape().rank() == _false_node->shape().rank()) {
        set_shape(
            ElemWiseBinaryOp::infer_elemwise_shape(_true_node->shape(),
                                                   _false_node->shape()));
    } else {
        set_shape(Shape({UnknownDim}));
    }
}

/**
 * Makes sure all inputs of the given node have been evaluated, without
 * evaluating the node itself, unless it has already been evaluated
 * and cached previously. This it necessary for proper work of the Cond
 * node by two reasons:
 * 1. To match the behaviour of `cond` from TF, which is this (a quote):
 *
 *        Note that the conditional execution applies only to the operations
 *        defined in true_fn and false_fn. Consider the following simple program:
 *
 *        z = tf.multiply(a, b)
 *        result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
 *
 *        If x < y, the tf.add operation will be executed and tf.square
 *        operation will not be executed. Since z is needed for at least
 *        one branch of the cond, the tf.multiply operation is always executed,
 *        unconditionally. Although this behavior is consistent with
 *        the dataflow model of TensorFlow, it has occasionally surprised
 *        some users who expected a lazier semantics.
 *
 *        https://www.tensorflow.org/api_docs/python/tf/cond
 *
 * 2. Such evaluation helps to make sure we don't have any values stored
 *    in cache with counters > 0 waiting to be used during the run.
 *    By evaluating those nodes we imitate usage of them as inputs, thus
 *    making sure that caching works as expected.
 */

void handle_not_evaluated_node(const NodeRef &node, Context &context,
                               ExecutionCache &cache) {
    cache.decrease_counter(node->id);
}

MultiArrayRef Cond::eval(Context &context, ExecutionCache &cache) const {
    MultiArrayRef result;
    if (!cache.get(id, result)) {
        auto cond_value = _cond_node->eval(context, cache);
        std::vector<BoolArrayStaticType> condition;
        cond_value->fetch_data_into(condition);
        if (condition[0]) {
            result = _true_node->eval(context, cache);
            handle_not_evaluated_node(_false_node, context, cache);
        } else {
            result = _false_node->eval(context, cache);
            handle_not_evaluated_node(_true_node, context, cache);
        }
        cache.put(id, result);
    }
    return result;
}

std::string Cond::to_string() const {
    return fmt::format("(if {} then {} else {})",
                       _cond_node->to_string(),
                       _true_node->to_string(),
                       _false_node->to_string());
}

NodeRefList Cond::inputs() const {
    return NodeRefList({_cond_node, _true_node, _false_node});
}

const NodeRef Cond::apply_chain_rule(const NodeRef &wrt_input,
                                     const NodeRef &d_target_wrt_this,
                                     const NodeRefList &all_inputs) const {
    if (wrt_input == all_inputs[0]) {
        return Constant::zeros_like(wrt_input);
    } else if (wrt_input == all_inputs[1]) {
        return Cond::make(all_inputs[0],
                          Constant::ones_like(all_inputs[1]),
                          Constant::zeros_like(all_inputs[1]));
    } else if (wrt_input == all_inputs[2]) {
        return Cond::make(all_inputs[0],
                          Constant::zeros_like(all_inputs[2]),
                          Constant::ones_like(all_inputs[2]));
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}

NodeRef Cond::make(const NodeRef &condition, CondExpression true_fn,
                   CondExpression false_fn) {
    auto *raw_ptr = new Cond(condition, true_fn(), false_fn());
    return std::static_pointer_cast<BaseNode>(std::shared_ptr<Cond>(raw_ptr));
}

NodeRef Cond::make(const NodeRef &condition, const NodeRef &true_node,
                   const NodeRef &false_node) {
    auto *raw_ptr = new Cond(condition, true_node, false_node);
    return std::static_pointer_cast<BaseNode>(std::shared_ptr<Cond>(raw_ptr));
}

NodeRef Cond::make(const NodeRef &condition, const NodeRef &true_node,
                   CondExpression false_fn) {
    auto *raw_ptr = new Cond(condition, true_node, false_fn());
    return std::static_pointer_cast<BaseNode>(std::shared_ptr<Cond>(raw_ptr));
}

NodeRef Cond::make(const NodeRef &condition, CondExpression true_fn,
                   const NodeRef &false_node) {
    auto *raw_ptr = new Cond(condition, true_fn(), false_node);
    return std::static_pointer_cast<BaseNode>(std::shared_ptr<Cond>(raw_ptr));
}

} // namespace
