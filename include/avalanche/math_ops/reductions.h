#ifndef AVALANCHE_REDUCTIONS_H
#define AVALANCHE_REDUCTIONS_H

#include <vector>
#include <string>

#include "avalanche/Shape.h"
#include "avalanche/BaseNode.h"
#include "avalanche/math_ops/messages.h"

namespace avalanche {

class Reduction {
public:
    Reduction(const NodeRef &input);
    Reduction(const NodeRef &input, std::vector<ShapeDim> dims_to_cut);

    const Shape& shape() const { return _result_shape; }
    ArrayType dtype() const { return _result_dtype; }

    // TODO: rh_name() needs the actual dimensions
    std::string lh_name() const {
        return std::string("reduce") + kernel_op_name() + "(";
    }
    std::string rh_name() const { return ", ...)"; }


    MultiArrayRef forward(const MultiArrayRef &value) const;

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;;

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

    Shape _result_shape;
    ArrayType _result_dtype;
    std::vector<ReductionStep> _reduction_steps;

    virtual std::string kernel_op_name() const =0;

private:
    mutable std::string _kernel_name;

    MultiArrayRef partial_reduction(const MultiArrayRef &value) const;
    MultiArrayRef full_reduction(const MultiArrayRef &value) const;

    const std::string& cached_kernel_name() const {
        if (_kernel_name.empty()) {
            bool is_partial_reduction = _result_shape.rank() > 0;
            auto op_name = kernel_op_name();
            if (is_partial_reduction) {
                _kernel_name = (
                    std::string("reduce_") + op_name
                    + "_" + array_type_name(_result_dtype));
            } else {
                _kernel_name = (
                    std::string("step_of_full_reduce_") + op_name
                    + "_" + array_type_name(_result_dtype));
            }
        }
        return _kernel_name;
    }
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
};

class ReduceMax : public Reduction {
public:
    using Reduction::Reduction;
    virtual std::string kernel_op_name() const override { return "max"; };
};

} // namespace

#endif //AVALANCHE_REDUCTIONS_H
