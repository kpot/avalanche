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
        return FU<ReduceSum>(d_target_wrt_this, left_vs_result_shape_diff);
    } else if (all_inputs[1] == wrt_input) {
        // Here we need to determine the dimensions we remove
        return FU<ReduceSum>(d_target_wrt_this, right_vs_result_shape_diff);
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
        return FU<ReduceSum>(d_target_wrt_this, left_vs_result_shape_diff);
    } else if (all_inputs[1] == wrt_input) {
        // with respect to right_operand
        return FU<Scale>(
            FU<ReduceSum>(d_target_wrt_this, right_vs_result_shape_diff),
            -1);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}


const NodeRef Multiply::apply_chain_rule(const NodeRef &wrt_input,
                                         const NodeRef &d_target_wrt_this,
                                         const NodeRefList &all_inputs) const {
    NodeRef input_product = nullptr;
    for (auto &input: all_inputs) {
        if (input == wrt_input) continue;
        if (input_product == nullptr) {
            input_product = input;
        } else {
            input_product = F<Multiply>(input_product, input);
        }
    }
    return F<Multiply>(input_product, d_target_wrt_this);
}

const NodeRef Divide::apply_chain_rule(const NodeRef &wrt_input,
                                       const NodeRef &d_target_wrt_this,
                                       const NodeRefList &all_inputs) const {
    if (all_inputs[0] == wrt_input) {
        // with respect to left operand
        return F<Divide>(d_target_wrt_this, all_inputs[1]);
    } else if (all_inputs[1] == wrt_input) {
        // with respect to right_operand
        return F<Multiply>(
            F<Divide>(F<Negate>(all_inputs[0]),
                      F<Multiply>(all_inputs[1], all_inputs[1])),
            d_target_wrt_this);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}


} // namespace
