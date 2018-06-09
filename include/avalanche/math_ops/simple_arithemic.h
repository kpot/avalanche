#ifndef AVALANCHE_SIMPLE_ARITHEMIC_H
#define AVALANCHE_SIMPLE_ARITHEMIC_H

#include "avalanche/base_ops_nodes.h"
#include "avalanche/math_ops/BroadcastedBinaryOp.h"

namespace avalanche {

struct Plus : BroadcastedBinaryOp {

    Plus(const NodeRef &left, const NodeRef &right)
        : BroadcastedBinaryOp(left, right) {}

    std::string name() const { return "+"; }
    const char* kernel_op_name() const final { return "plus"; }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};


struct Minus : BroadcastedBinaryOp {

    Minus(const NodeRef &left, const NodeRef &right)
        : BroadcastedBinaryOp(left, right) {}

    std::string name() const { return "-"; }
    const char* kernel_op_name() const final { return "minus"; }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};


struct Multiply : BroadcastedBinaryOp {

    Multiply(const NodeRef &left, const NodeRef &right)
        :BroadcastedBinaryOp(left, right) {}

    std::string name() const { return "*"; }
    const char* kernel_op_name() const final { return "multiply"; }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};


struct Divide : BroadcastedBinaryOp {

    Divide(const NodeRef &left, const NodeRef &right)
        :BroadcastedBinaryOp(left, right) {}

    std::string name() const { return "/"; }
    const char* kernel_op_name() const final { return "divide"; }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};


inline NodeRef operator+(const NodeRef &node1, const NodeRef &node2) {
    return F<Plus>(node1, node2);
}

inline NodeRef operator-(const NodeRef &node1, const NodeRef &node2) {
    return F<Minus>(node1, node2);
}

inline NodeRef operator*(const NodeRef &node1, const NodeRef &node2) {
    return F<Multiply>(node1, node2);
}

inline NodeRef operator/(const NodeRef &node1, const NodeRef &node2) {
    return F<Divide>(node1, node2);
}

} // namespace

#endif //AVALANCHE_SIMPLE_ARITHEMIC_H
