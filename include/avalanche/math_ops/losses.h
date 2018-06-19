#ifndef AVALANCHE_LOSSES_H
#define AVALANCHE_LOSSES_H

#include "avalanche/math_ops/ElemWiseBinaryOp.h"

namespace avalanche {

class BinaryCrossEntropy: public ElemWiseBinaryOp {
public:
    BinaryCrossEntropy(const NodeRef &logits, const NodeRef &labels)
        : ElemWiseBinaryOp(
        logits, labels, "binary_crossentropy",
        opencl_expression(logits->dtype(), labels->dtype()),
        choose_common_array_type(logits->dtype(), labels->dtype())) {}

    std::string name() const { return "+"; }

    static std::string opencl_expression(ArrayType left_dtype,
                                         ArrayType right_dtype);

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};

} // namespace

#endif //AVALANCHE_LOSSES_H
