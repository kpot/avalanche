#include <iostream>

#include "avalanche/opencl_utils.h"
#include "avalanche/CodeCache.h"
#include "avalanche/MultiArray.h"
#include "avalanche/terminal_nodes.h"
#include "avalanche/math_ops/simple_arithemic.h"
#include "avalanche/math_ops/messages.h"
#include "avalanche/math_ops/const_transformation.h"
#include "avalanche/math_ops/reductions.h"

namespace avalanche {


const NodeRef Plus::apply_chain_rule(
    const NodeRef &wrt_input,
    const NodeRef &d_target_wrt_this,
    const NodeRefList &all_inputs) const {
    if (all_inputs[0] == wrt_input) {
        // Here we need to determine the dimensions we remove
        return left_vs_result_shape_diff.empty() ? d_target_wrt_this : FU<ReduceSum>(d_target_wrt_this, left_vs_result_shape_diff);
    } else if (all_inputs[1] == wrt_input) {
        // Here we need to determine the dimensions we remove
        return right_vs_result_shape_diff.empty() ? d_target_wrt_this : FU<ReduceSum>(d_target_wrt_this, right_vs_result_shape_diff);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}


const NodeRef
Minus::apply_chain_rule(const NodeRef &wrt_input,
                        const NodeRef &d_target_wrt_this,
                        const NodeRefList &all_inputs) const {
    if (all_inputs[0] == wrt_input) {
        // with respect to left operand
        return left_vs_result_shape_diff.empty() ? d_target_wrt_this : FU<ReduceSum>(d_target_wrt_this, left_vs_result_shape_diff);
    } else if (all_inputs[1] == wrt_input) {
        // with respect to right_operand
        return FU<Scale>(
            right_vs_result_shape_diff.empty() ? d_target_wrt_this : FU<ReduceSum>(d_target_wrt_this, right_vs_result_shape_diff),
            -1);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}


const NodeRef Multiply::apply_chain_rule(const NodeRef &wrt_input,
                                         const NodeRef &d_target_wrt_this,
                                         const NodeRefList &all_inputs) const {
    if (all_inputs[0] == wrt_input) {
        // with respect to left operand
        auto part_derivative = F<Multiply>(d_target_wrt_this, all_inputs[1]);
        return left_vs_result_shape_diff.empty() ? part_derivative : FU<ReduceSum>(part_derivative, left_vs_result_shape_diff, true);
    } else if (all_inputs[1] == wrt_input) {
        // with respect to right_operand
        auto part_derivative = F<Multiply>(d_target_wrt_this, all_inputs[0]);
        return right_vs_result_shape_diff.empty() ? part_derivative : FU<ReduceSum>(part_derivative, right_vs_result_shape_diff, true);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}

const NodeRef Divide::apply_chain_rule(const NodeRef &wrt_input,
                                       const NodeRef &d_target_wrt_this,
                                       const NodeRefList &all_inputs) const {
    if (all_inputs[0] == wrt_input) {
        // with respect to left operand
        auto part_derivative = F<Divide>(d_target_wrt_this, all_inputs[1]);
        return left_vs_result_shape_diff.empty() ? part_derivative : FU<ReduceSum>(part_derivative, left_vs_result_shape_diff, true);
    } else if (all_inputs[1] == wrt_input) {
        // with respect to right_operand
        auto part_derivative = F<Multiply>(
            d_target_wrt_this,
            F<Divide>(F<Negate>(all_inputs[0]),
                      F<Multiply>(all_inputs[1], all_inputs[1])));
        return right_vs_result_shape_diff.empty() ? part_derivative : FU<ReduceSum>(part_derivative, right_vs_result_shape_diff, true);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}


} // namespace
