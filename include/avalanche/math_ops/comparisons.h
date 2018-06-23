#ifndef AVALANCHE_COMPARISONS_H
#define AVALANCHE_COMPARISONS_H

#include "avalanche/base_ops_nodes.h"
#include "avalanche/math_ops/BroadcastedBinaryOp.h"

namespace avalanche {

constexpr ArrayType BoolArrayType = ArrayType::int8;
using BoolArrayStaticType = std::int8_t;

class Comparison : public BroadcastedBinaryOp {
public:
    Comparison(const NodeRef &left,
               const NodeRef &right,
               const std::string &operation_name,
               const std::string &operation_cl_code)
    :BroadcastedBinaryOp(left, right, operation_name,
                         operation_cl_code, BoolArrayType)
    {}

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const {
        throw std::runtime_error("Not applicable");
    }

    bool use_in_back_propagation() const override { return false; }
};


class NotEqual : public Comparison {
public:
    NotEqual(const NodeRef &left, const NodeRef &right)
        : Comparison(left, right, "not_equal", "a != b") {}

    std::string name() const { return "!="; }
};


class Equal : public Comparison {
public:
    Equal(const NodeRef &left, const NodeRef &right)
        : Comparison(left, right, "equal", "a == b") {}

    std::string name() const { return "=="; }
};


class Greater : public Comparison {
public:
    Greater(const NodeRef &left, const NodeRef &right)
        : Comparison(left, right, "greater", "a > b") {}

    std::string name() const { return ">"; }
};


class GreaterEqual : public Comparison {
public:
    GreaterEqual(const NodeRef &left, const NodeRef &right)
        : Comparison(left, right, "greater_equal", "a >= b") {}

    std::string name() const { return ">="; }
};


class Less : public Comparison {
public:
    Less(const NodeRef &left, const NodeRef &right)
        : Comparison(left, right, "less", "a < b") {}

    std::string name() const { return "<"; }
};


class LessEqual : public Comparison {
public:
    LessEqual(const NodeRef &left, const NodeRef &right)
        : Comparison(left, right, "less_equal", "a <= b") {}

    std::string name() const { return "<="; }
};

} // namespace

#endif //AVALANCHE_COMPARISONS_H
