#include <iostream>

#include "avalanche/opencl_utils.h"
#include "avalanche/CodeCache.h"
#include "avalanche/MultiArray.h"
#include "avalanche/terminal_nodes.h"
#include "avalanche/math_ops/simple_arithemic.h"
#include "avalanche/math_ops/messages.h"
#include "avalanche/math_ops/const_transformation.h"
#include "avalanche/math_ops/reductions.h"
#include "avalanche/shape_nodes.h"

namespace avalanche {


const NodeRef Plus::apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const {
    NodeRef derivative = nullptr;
    if (all_inputs[0] == wrt_input || all_inputs[1] == wrt_input) {
        derivative = F<ReduceSum>(d_target_wrt_this, F<NoBackProp>(wrt_input), true);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
    // TODO: This can be made unnecessary
    if (derivative->shape() != wrt_input->shape()) {
        derivative = ReshapeLike::make(derivative, wrt_input);
    }
    return derivative;
}


const NodeRef
Minus::apply_chain_rule(const NodeRef &wrt_input,
                        const NodeRef &d_target_wrt_this,
                        const NodeRefList &all_inputs) const {
    NodeRef derivative = nullptr;
    if (all_inputs[0] == wrt_input) {
        // with respect to left operand
        derivative = F<ReduceSum>(d_target_wrt_this, F<NoBackProp>(wrt_input), true);
    } else if (all_inputs[1] == wrt_input) {
        // with respect to right_operand
        derivative = F<ReduceSum>(F<Negate>(d_target_wrt_this), F<NoBackProp>(wrt_input), true);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
    if (derivative->shape() != wrt_input->shape()) {
        derivative = ReshapeLike::make(derivative, wrt_input);
    }
    return derivative;
}


const NodeRef Multiply::apply_chain_rule(const NodeRef &wrt_input,
                                         const NodeRef &d_target_wrt_this,
                                         const NodeRefList &all_inputs) const {
    NodeRef derivative = nullptr;
    if (all_inputs[0] == wrt_input) {
        // with respect to left operand
        auto part_derivative = F<Multiply>(d_target_wrt_this, all_inputs[1]);
        derivative = F<ReduceSum>(part_derivative, F<NoBackProp>(wrt_input), true);
    } else if (all_inputs[1] == wrt_input) {
        // with respect to right_operand
        auto part_derivative = F<Multiply>(d_target_wrt_this, all_inputs[0]);
        derivative = F<ReduceSum>(part_derivative, F<NoBackProp>(wrt_input), true);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
    if (derivative->shape() != wrt_input->shape()) {
        derivative = ReshapeLike::make(derivative, wrt_input);
    }
    return derivative;
}

const NodeRef Divide::apply_chain_rule(const NodeRef &wrt_input,
                                       const NodeRef &d_target_wrt_this,
                                       const NodeRefList &all_inputs) const {
    NodeRef derivative = nullptr;
    if (all_inputs[0] == wrt_input) {
        // with respect to left operand
        auto part_derivative = F<Divide>(d_target_wrt_this, all_inputs[1]);
        derivative = F<ReduceSum>(part_derivative, wrt_input, true);
    } else if (all_inputs[1] == wrt_input) {
        // with respect to right_operand
        auto part_derivative = F<Multiply>(
            d_target_wrt_this,
            F<Negate>(F<Divide>(all_inputs[0],
                                F<Square>(all_inputs[1]))));
        derivative = F<ReduceSum>(part_derivative, F<NoBackProp>(wrt_input), true);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
    if (derivative->shape() != wrt_input->shape()) {
        derivative = ReshapeLike::make(derivative, wrt_input);
    }
    return derivative;
}


const NodeRef Power::apply_chain_rule(const NodeRef &wrt_input,
                                      const NodeRef &d_target_wrt_this,
                                      const NodeRefList &all_inputs) const {
    NodeRef part_derivative = nullptr;
    if (all_inputs[0] == wrt_input) {
        // with respect to left operand (named "x" below)
        // d(x^a)/dx = a * x^(a - 1)
        part_derivative = F<Multiply>(
            d_target_wrt_this,
            F<Multiply>(
                all_inputs[1],
                F<Power>(all_inputs[0],
                         F<Minus>(all_inputs[1],
                                  Constant::ones_like(all_inputs[1])))));
    } else if (all_inputs[1] == wrt_input) {
        // with respect to right_operand (name "a" below)
        // d(x^a)/da = x^a * log(x)
        part_derivative = F<Multiply>(
            d_target_wrt_this,
            // TODO: Can be improved by replacing F<Power>(...) with a reference
            // to this node (which would require to change the signature
            // of apply_chain_rule method everywhere)
            F<Multiply>(
                F<Power>(all_inputs[0], all_inputs[1]),
                F<Log>(all_inputs[0])));
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
    auto derivative = F<ReduceSum>(part_derivative, wrt_input, true);
    if (derivative->shape() != wrt_input->shape()) {
        derivative = ReshapeLike::make(derivative, wrt_input);
    }
    return derivative;
}

const NodeRef ElemWisePlus::apply_chain_rule(const NodeRef &wrt_input,
                                             const NodeRef &d_target_wrt_this,
                                             const NodeRefList &all_inputs) const {

    return d_target_wrt_this;
}

const NodeRef ElemWiseMultiply::apply_chain_rule(const NodeRef &wrt_input,
                                                 const NodeRef &d_target_wrt_this,
                                                 const NodeRefList &all_inputs) const {
    if (all_inputs[0] == wrt_input) {
        // with respect to left operand
        return F<ElemWiseMultiply>(d_target_wrt_this, all_inputs[1]);
    } else if (all_inputs[1] == wrt_input) {
        // with respect to right_operand
        return F<ElemWiseMultiply>(d_target_wrt_this, all_inputs[0]);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}
} // namespace
