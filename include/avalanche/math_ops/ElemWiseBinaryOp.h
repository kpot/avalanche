#ifndef AVALANCHE_SIMPLEBINARYOP_H
#define AVALANCHE_SIMPLEBINARYOP_H

#include "avalanche/Shape.h"
#include "avalanche/ArrayType.h"
#include "avalanche/BaseNode.h"

namespace avalanche {

class ElemWiseBinaryOp {
public:
    Shape shape() const {
        return _result_shape;
    }

    ArrayType dtype() const {
        return _result_dtype;
    }

    ElemWiseBinaryOp(const NodeRef &left,
                     const NodeRef &right,
                     const std::string &operation_name,
                     const std::string &operation_cl_code,
                     ArrayType output_dtype);

    MultiArrayRef forward(const MultiArrayRef &v1,
                          const MultiArrayRef &v2) const;

    virtual bool use_in_back_propagation() const { return true; };

    static Shape infer_elemwise_shape(const Shape &shape1, const Shape &shape2);

private:
    Shape _result_shape;
    ArrayType _result_dtype;
    std::string _operation_name;
    std::string _kernel_name;
    std::string _kernel_source;

};

} // namespace

#endif //AVALANCHE_SIMPLEBINARYOP_H
