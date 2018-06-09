#ifndef AVALANCHE_MATMUL_H
#define AVALANCHE_MATMUL_H

#include <string>

#include "avalanche/MultiArray.h"
#include "avalanche/BaseNode.h"

namespace avalanche {

struct MatMul {
    bool transpose_left;
    bool transpose_right;
    Shape result_shape;
    ArrayType result_dtype;

    MatMul(const NodeRef &left, const NodeRef &right,
           bool transpose_left=false, bool transpose_right=false);

    Shape shape() const {
        return result_shape;
    }

    ArrayType dtype() const {
        return result_dtype;
    }

    std::string name() const {
        return std::string(transpose_left ? "T" : "")
               + "matmul"
               + (transpose_right ? "T" : "");
    }

    MultiArrayRef forward(const MultiArrayRef &v1,
                          const MultiArrayRef &v2) const;


    const NodeRef
    apply_chain_rule(const NodeRef &wrt_input, const NodeRef &d_target_wrt_this,
                     const NodeRefList &all_inputs) const;
};

} // namespace

#endif //AVALANCHE_MATMUL_H
