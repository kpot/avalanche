#ifndef AVALANCHE_SIMPLE_ARITHEMIC_H
#define AVALANCHE_SIMPLE_ARITHEMIC_H

#include "avalanche/base_ops_nodes.h"
#include "avalanche/math_ops/BroadcastedBinaryOp.h"

namespace avalanche {

class Plus : public BroadcastedBinaryOp {
public:
    Plus(const NodeRef &left, const NodeRef &right)
        : BroadcastedBinaryOp(left, right, "plus",
                              "a + b", left->dtype()) {}

    std::string name() const { return "+"; }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};


class Minus : public BroadcastedBinaryOp {
public:
    Minus(const NodeRef &left, const NodeRef &right)
        : BroadcastedBinaryOp(left, right, "minus",
                              "a - b", left->dtype()) {}

    std::string name() const { return "-"; }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};


class Multiply : public BroadcastedBinaryOp {
public:
    Multiply(const NodeRef &left, const NodeRef &right)
        : BroadcastedBinaryOp(left, right, "multiply",
                              "a * b", left->dtype()) {}

    std::string name() const { return "*"; }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};


class Divide : public BroadcastedBinaryOp {
public:
    Divide(const NodeRef &left, const NodeRef &right)
        : BroadcastedBinaryOp(left, right, "divide",
                              cl_operation_code(left->dtype()),
                              left->dtype()) {}

    std::string name() const { return "/"; }

    static std::string cl_operation_code(ArrayType dtype) {
        if (dtype == ArrayType::float32) {
            return "native_divide(a, b)";
        }
        return "a / b";
    }

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
