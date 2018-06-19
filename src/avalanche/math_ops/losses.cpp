#include <fmt/format.h>

#include "avalanche/BaseNode.h"
#include "avalanche/math_ops/losses.h"
#include "avalanche/math_ops/messages.h"
#include "avalanche/math_ops/simple_arithemic.h"
#include "avalanche/math_ops/const_transformation.h"
#include "avalanche/terminal_nodes.h"

namespace avalanche {

class DiffBinaryCrossEntropyWithLogits : public ElemWiseBinaryOp {
public:
    DiffBinaryCrossEntropyWithLogits(const NodeRef &logits, const NodeRef &labels)
        : ElemWiseBinaryOp(
        logits, labels, "diff_binary_crossentropy_with_logits",
        opencl_expression(logits->dtype(), labels->dtype()),
        choose_common_array_type(logits->dtype(), labels->dtype())) {}

    std::string name() const { return "+"; }

    static std::string opencl_expression(ArrayType left_dtype,
                                         ArrayType right_dtype)  {
        return "(a - b) / (a - a * a)";
    }

    const NodeRef apply_chain_rule(
            const NodeRef &wrt_input,
            const NodeRef &d_target_wrt_this,
            const NodeRefList &all_inputs) const {
        throw std::runtime_error("Not implemented");
    }
};

const NodeRef
BinaryCrossEntropy::apply_chain_rule(const NodeRef &wrt_input,
                                               const NodeRef &d_target_wrt_this,
                                               const NodeRefList &all_inputs) const {
    if (all_inputs[0] == wrt_input) {
        // Derivative w.r.t. logits
        return F<DiffBinaryCrossEntropyWithLogits>(all_inputs[0], all_inputs[1]);
    } else if (all_inputs[1] == wrt_input) {
        // Derivative w.r.t. labels.
        // Not really necessary, but kinda completes the picture
        auto one = Constant::ones_like(wrt_input);
        return F<Log>(one - all_inputs[0]) - F<Log>(all_inputs[0]);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}

std::string BinaryCrossEntropy::opencl_expression(ArrayType left_dtype,
                                                  ArrayType right_dtype) {

    auto left_type_name = cl_type_name_of_array(left_dtype);
    return fmt::format("-b * log(a) - (1.0 - b) * log(({0})1.0 - a)", left_type_name);
}

} // namespace
