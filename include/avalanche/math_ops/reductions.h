#ifndef AVALANCHE_REDUCTIONS_H
#define AVALANCHE_REDUCTIONS_H

#include <vector>
#include <string>
#include <sstream>

#include "avalanche/Shape.h"
#include "avalanche/BaseNode.h"
#include "avalanche/math_ops/messages.h"

namespace avalanche {

class Reshape {
public:
    Reshape(const NodeRef &input, const Shape &new_shape);
    const Shape& shape() const { return _new_shape; }
    ArrayType dtype() const { return _result_dtype; }

    std::string lh_name() const { return "reshape("; }
    std::string rh_name() const { return ", " + _new_shape.to_string() + ")"; }

    MultiArrayRef forward(const MultiArrayRef &value) const;

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;
private:
    const Shape _new_shape;
    const ArrayType _result_dtype;
};

class Reduction {
public:
    Reduction(const NodeRef &input);
    /**
     * @param input a node
     * @param reduce_axis all dimensions that need to be reduces. If empty,
     *    this means that all of them must be reduced to a single scalar.
     */
    Reduction(const NodeRef &input,
              std::vector<ShapeDim> reduce_axis,
              bool keep_dims = false);

    const Shape& shape() const {
        return _keep_dims ? _result_shape_dims_kept : _result_shape_dims_cut;
    }
    ArrayType dtype() const { return _result_dtype; }

    std::string lh_name() const {
        return std::string("reduce") + kernel_op_name() + "(";
    }
    std::string rh_name() const;


    MultiArrayRef forward(const MultiArrayRef &value) const;

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;

    virtual const NodeRef partial_derivative(const NodeRef &input) const {
        throw std::logic_error("not implemented");
    }


protected:
    struct ReductionStep {
        // Size of the final array after reduction
        std::size_t result_size;
        // The size of the whole dimension we cut
        std::size_t source_stride;
        // How big each block of sub-elements in the dimensions we cut
        std::size_t source_block;
        // Size of the dimension we're cutting
        std::size_t dim_size;
    };

    Shape _result_shape_dims_cut;
    Shape _result_shape_dims_kept;
    ArrayType _result_dtype;
    std::vector<ReductionStep> _reduction_steps;
    std::vector<ShapeDim> _dims_to_cut;
    const bool _keep_dims;

    virtual std::string kernel_op_name() const =0;

private:
    mutable std::string _kernel_name;

    MultiArrayRef partial_reduction(const MultiArrayRef &value) const;
    MultiArrayRef full_reduction(const MultiArrayRef &value) const;

    const std::string& cached_kernel_name() const;
};

class ReduceSum : public Reduction {
public:
    using Reduction::Reduction;
    virtual std::string kernel_op_name() const override { return "sum"; };

    const NodeRef partial_derivative(const NodeRef &input) const override;
};

class ReduceProd : public Reduction {
public:
    using Reduction::Reduction;
    virtual std::string kernel_op_name() const override { return "prod"; };
};

class ReduceMean : public Reduction {
public:
    using Reduction::Reduction;
    virtual std::string kernel_op_name() const override { return "mean"; };
    const NodeRef partial_derivative(const NodeRef &wrt_input) const override;
};

class ReduceMin : public Reduction {
public:
    using Reduction::Reduction;
    virtual std::string kernel_op_name() const override { return "min"; };
    // TODO: Implement partial derivative
};

class ReduceMax : public Reduction {
public:
    using Reduction::Reduction;
    virtual std::string kernel_op_name() const override { return "max"; };
    // TODO: Implement partial derivative
};

const NodeRef softmax(const NodeRef &node, ShapeDim axis=-1);

} // namespace

#endif //AVALANCHE_REDUCTIONS_H
