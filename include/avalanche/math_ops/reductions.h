#ifndef AVALANCHE_REDUCTIONS_H
#define AVALANCHE_REDUCTIONS_H

#include <vector>
#include <string>
#include <sstream>

#include "avalanche/Shape.h"
#include "avalanche/BaseNode.h"
#include "avalanche/math_ops/messages.h"
#include "avalanche/shape_nodes.h"

namespace avalanche {

class ShapeDimsToOnes {
public:
    ShapeDimsToOnes(const NodeRef &shape_node, std::vector<ShapeDim> &dims_to_replace)
    :_result_shape {shape_node->shape()}, _dims_to_replace{dims_to_replace}
    {
        if (shape_node->shape().rank() != 1 || shape_node->dtype() != ShapeOf::DType) {
            throw std::invalid_argument(
                "Node " + shape_node->repr() +
                " doesn't look like a shape node.");
        }
    }

    const Shape& shape() const { return _result_shape; }
    ArrayType dtype() const { return ShapeOf::DType; }

    bool use_in_back_propagation() const {
        return false;
    };

    MultiArrayRef forward(const MultiArrayRef &value) const {
        auto extracted_dims = ShapeOf::extract_shape_from_metadata(value);
        auto extracted_shape = Shape(extracted_dims);
        auto dims_to_replace = extracted_shape.normalize_dims(_dims_to_replace);
        for (std::size_t i = 0; i < dims_to_replace.size(); ++i) {
            extracted_dims[dims_to_replace[i]] = 1;
        }
        auto pool = value->buffer_unsafe()->pool();
        auto result = pool->make_array(
            Shape({static_cast<ShapeDim >(extracted_dims.size())}),
            ShapeOf::DType);
        return result;
    }

private:
    Shape _result_shape;
    std::vector<ShapeDim> _dims_to_replace;
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

    Reduction(const NodeRef &input, const NodeRef &to_be_like, bool keep_dims);

    const Shape& shape() const {
        return _keep_dims ? _result_shape_dims_kept : _result_shape_dims_cut;
    }
    ArrayType dtype() const { return _result_dtype; }

    std::string lh_name() const {
        return std::string("reduce") + kernel_op_name() + "(";
    }
    std::string rh_name() const;
    std::string name() const;

    MultiArrayRef forward(const MultiArrayRef &value) const;
    MultiArrayRef forward(const MultiArrayRef &value,
                          const MultiArrayRef &to_be_like_value) const;

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;

    virtual const NodeRef partial_derivative(const NodeRef &input) const {
        throw std::logic_error("not implemented");
    }

    bool use_in_back_propagation() const { return true; };

    std::string repr_extra() const;

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

    // The variables below are merely estimations made when the graph
    // was being constructed
    Shape _result_shape_dims_cut;
    Shape _result_shape_dims_kept;
    ArrayType _result_dtype;
    std::vector<ShapeDim> _dims_to_cut;
    const bool _keep_dims;
    const NodeRef _to_be_like;

    virtual std::string kernel_op_name() const =0;

private:

    MultiArrayRef partial_reduction(
        const MultiArrayRef &value,
        const std::vector<ReductionStep> &reduction_steps,
        const Shape &result_shape_dims_cut,
        const Shape &result_shape_dims_kept) const;
    MultiArrayRef full_reduction(const MultiArrayRef &value) const;

    const std::string get_kernel_name(bool is_partial_reduction) const;
    void estimate_steps_and_dimensions(
        const Shape &input_shape,
        const std::vector<ShapeDim> &dims_to_cut,
        std::vector<ReductionStep> &reduction_steps,
        Shape &result_shape_dims_cut,
        Shape &result_shape_dims_kept) const;

    std::vector<ShapeDim> estimate_dims_to_cut(
        const Shape &input_shape, const Shape &to_be_like_shape) const;
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
    // TODO: Implement partial derivative
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
