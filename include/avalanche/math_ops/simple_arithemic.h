#ifndef AVALANCHE_SIMPLE_ARITHEMIC_H
#define AVALANCHE_SIMPLE_ARITHEMIC_H

#include "avalanche/base_ops_nodes.h"
#include "avalanche/math_ops/BroadcastedBinaryOp.h"
#include "avalanche/math_ops/ElemWiseBinaryOp.h"

namespace avalanche {

class Plus : public BroadcastedBinaryOp {
public:
    Plus(const NodeRef &left, const NodeRef &right)
        : BroadcastedBinaryOp(
            left, right, "plus",
            "a + b",
            choose_common_array_type(left->dtype(), right->dtype())) {}

    std::string name() const { return "+"; }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};

class ElemWisePlus : public ElemWiseBinaryOp {
public:
    ElemWisePlus(const NodeRef &left, const NodeRef &right)
        : ElemWiseBinaryOp(
        left, right, "elem_wise_plus",
        "a + b",
        choose_common_array_type(left->dtype(), right->dtype())) {}

    std::string name() const { return "+"; }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};

class Minus : public BroadcastedBinaryOp {
public:
    Minus(const NodeRef &left, const NodeRef &right)
        : BroadcastedBinaryOp(
            left, right, "minus",
            "a - b",
            choose_common_array_type(left->dtype(), right->dtype())) {}

    std::string name() const { return "-"; }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};


class Multiply : public BroadcastedBinaryOp {
public:
    Multiply(const NodeRef &left, const NodeRef &right)
        : BroadcastedBinaryOp(
            left, right, "multiply",
            "a * b", choose_common_array_type(left->dtype(), right->dtype())) {}

    std::string name() const { return "*"; }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};


class ElemWiseMultiply : public ElemWiseBinaryOp {
public:
    ElemWiseMultiply(const NodeRef &left, const NodeRef &right)
        : ElemWiseBinaryOp(
        left, right, "elem_wise_multiply",
        "a * b",
        choose_common_array_type(left->dtype(), right->dtype())) {}

    std::string name() const { return "+"; }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};


class Divide : public BroadcastedBinaryOp {
public:
    Divide(const NodeRef &left, const NodeRef &right)
        : BroadcastedBinaryOp(
            left, right, "divide",
            cl_operation_code(left->dtype(), right->dtype()),
            choose_common_array_type(left->dtype(), right->dtype())) {}

    std::string name() const { return "/"; }

    static std::string
    cl_operation_code(ArrayType left_dtype, ArrayType right_dtype) {
        if (choose_common_array_type(left_dtype, right_dtype)
                == ArrayType::float32) {
            return "native_divide((float)a, (float)b)";
        }
        return "a / b";
    }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
};

class Power : public BroadcastedBinaryOp {
public:
    Power(const NodeRef &left, const NodeRef &right)
        : BroadcastedBinaryOp(
        left, right, "power",
        cl_operation_code(left->dtype(), right->dtype()),
        choose_common_array_type(left->dtype(), right->dtype())) {}

    std::string name() const { return "/"; }

    static std::string
    cl_operation_code(ArrayType left_dtype, ArrayType right_dtype) {
        if (is_floating_array_type(left_dtype) &&
                !is_floating_array_type(right_dtype)) {
            return "pown(a, (int)b)";
        } else {
            auto uni_type = choose_common_array_type(left_dtype, right_dtype);
            const char *uni_cl_type = cl_type_name_of_array(uni_type);
            return (
                std::string("pow((") + uni_cl_type +
                    ")a, (" + uni_cl_type + ")b)");
        }
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
