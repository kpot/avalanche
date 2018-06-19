#include <iostream>

#include "CL_cust/cl2.hpp"
#include <fmt/format.h>

#include "avalanche/CodeCache.h"
#include "avalanche/terminal_nodes.h"
#include "avalanche/math_ops/simple_arithemic.h"
#include "avalanche/math_ops/messages.h"

#include "avalanche/math_ops/const_transformation.h"

namespace avalanche {

const NodeRef SPower::partial_derivative(const NodeRef &input) const {
    return FU<SPower>(input, params[0] * params[1], params[1] - 1);
}

const NodeRef Recip::partial_derivative(const NodeRef &input) const {
    return F<Negate>(FU<SPower>(input, 1.0, -2));
}

const NodeRef
Scale::apply_chain_rule(const NodeRef &wrt_input,
                        const NodeRef &d_target_wrt_this,
                        const NodeRefList &all_inputs) const {
    if (all_inputs[0] == wrt_input) {
        return std::make_shared<UnaryOp<Scale>>(d_target_wrt_this, params[0]);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}

const NodeRef Log::partial_derivative(const NodeRef &input) const {
    return F<Recip>(input);
}

class SigmoidDiff : public ConstTransform<0> {
public:
    SigmoidDiff(const NodeRef &input) :ConstTransform<0>(
        input, input->dtype(),
        {{"s", Sigmoid::opencl_expression(input->dtype())}},
        "s * (1-s)", "diffsigmoid(", ")", {}) {}
};

const NodeRef Sigmoid::partial_derivative(const NodeRef &input) const {
    // This isn't very beautiful to implement differential directly,
    // since it won't allow to calculate differentials of higher orders,
    // but it's effective
    return FU<SigmoidDiff>(input);
}

class TanhDiff : public ConstTransform<0> {
public:
    TanhDiff(const NodeRef &input) :ConstTransform<0>(
        input, input->dtype(),
        {{"s", Tanh::opencl_expression(input->dtype())}},
        "1 - s * s", "difftanh(", ")", {}) {}
};

const NodeRef Tanh::partial_derivative(const NodeRef &input) const {
    return FU<TanhDiff>(input);
}


class ReLUDiff : public ConstTransform<0> {
public:
    static std::string opencl_expression(ArrayType dtype) {
        const char *type_name = cl_type_name_of_array(dtype);
        return fmt::format("select(({0})0.0, ({0})1.0, isgreater(({0})v, 0))", type_name);
    }

    ReLUDiff(const NodeRef &input) :ConstTransform<0>(
        input, input->dtype(),
        {},
        opencl_expression(input->dtype()), "diffrelu(", ")", {}) {}
};

const NodeRef ReLU::partial_derivative(const NodeRef &input) const {
    return FU<ReLUDiff>(input);
}

const NodeRef Exp::partial_derivative(const NodeRef &input) const {
    return FU<Exp>(input);
}

const NodeRef Square::partial_derivative(const NodeRef &input) const {
    return FU<Scale>(input, 2);
}

const NodeRef Cast::apply_chain_rule(const NodeRef &wrt_input,
                                     const NodeRef &d_target_wrt_this,
                                     const NodeRefList &all_inputs) const {
    return FU<Cast>(d_target_wrt_this, wrt_input->dtype());
}

const NodeRef Sqrt::partial_derivative(const NodeRef &input) const {
    return F<Recip>(FU<Scale>(F<Sqrt>(input), 2)); //  = 1 / (2 * sqrt(x))
}

} // namespace
