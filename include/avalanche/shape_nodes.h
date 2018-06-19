#ifndef AVALANCHE_SHAPENODE_H
#define AVALANCHE_SHAPENODE_H

/**
 * This module contains anything transforming the shape or purpose of the data
 * without changing the content. This includes reshaping, fetching the shape
 * as a tensor,  concatenation of several nodes, etc.
 */

#include "avalanche/BaseNode.h"
#include "avalanche/Shape.h"

namespace avalanche {

/**
 * Absolutely transparent operation which passes any input values through it
 * but blocks back-propagation. Useful when you need to build a terminal node
 * which relies on some other differentiable inputs. For example, a constant
 * which size depends on a shape of a differentiable variable: in such case
 * any change in the variable's content will not affect the constant's content,
 * so this path is not differentiable. Yet we cannot declare the variable as
 * non-differentiable. And we still need to know that the constant depends from
 * it. Hence this class, which allows the constant to know about the variable
 * and allows effective caching (because both nodes are connected and
 * we can take this into account), without producing incorrect derivatives.
 */
class NoBackProp {
public:
    NoBackProp(const NodeRef &input)
        :_result_shape(input->shape()),
         _result_dtype(input->dtype())
    {
    }

    std::string lh_name() const { return "NoBackProp("; }
    std::string rh_name() const { return ")"; }

    const Shape& shape() const { return _result_shape; }
    ArrayType dtype() const { return _result_dtype; }

    MultiArrayRef forward(const MultiArrayRef &value) const { return value; }

    bool use_in_back_propagation() const { return false; }

    const NodeRef apply_chain_rule(const NodeRef &wrt_input,
                                   const NodeRef &d_target_wrt_this,
                                   const NodeRefList &all_inputs) const {
        return nullptr;
    };


private:
    ArrayType _result_dtype;
    Shape _result_shape;
};


class ShapeOf {
public:
    ShapeOf(const NodeRef &input)
        :_result_shape({static_cast<ShapeDim>(input->shape().rank())}) {
    }

    std::string lh_name() const { return "ShapeOf("; }
    std::string rh_name() const { return ")"; }

    const Shape& shape() const { return _result_shape; }
    static ArrayType dtype() { return ArrayType::int64; }

    MultiArrayRef forward(const MultiArrayRef &value) const;

    bool use_in_back_propagation() const { return false; }

    const NodeRef apply_chain_rule(const NodeRef &wrt_input,
                                   const NodeRef &d_target_wrt_this,
                                   const NodeRefList &all_inputs) const;

private:
    Shape _result_shape;
};


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

    bool use_in_back_propagation() const { return true; }
private:
    const Shape _new_shape;
    const ArrayType _result_dtype;
};

/**
 * Node that outputs a product of sizes of given dimensions for a node.
 * For example, if a node has shape (1, 2, 3, 4) and given dimensions
 * are [-1, 1], the result will be a value equal to 4 * 2 = 8
 * Given an empty list of dimensions, calculates products of all dimensions
 * of the input.
 */
class ProductOfDims {
public:
    ProductOfDims(const NodeRef &input, const std::vector<ShapeDim> &dims,
                  ArrayType output_dtype)
        :_result_shape(),
         _dims{input->shape().normalize_dims(dims)},
         _result_dtype{output_dtype} {

    }

    const Shape& shape() const { return _result_shape; }
    ArrayType dtype() const { return _result_dtype; }

    std::string lh_name() const { return "ProductOfDims("; }
    std::string rh_name() const { return ", " + Shape::dims_to_string(_dims) + ")"; }

    MultiArrayRef forward(const MultiArrayRef &value) const;

    const NodeRef apply_chain_rule(
            const NodeRef &wrt_input,
            const NodeRef &d_target_wrt_this,
            const NodeRefList &all_inputs) const {
        return nullptr;
    }

    bool use_in_back_propagation() const { return false; }
private:
    const Shape _result_shape;
    const ArrayType _result_dtype;
    std::vector<ShapeDim> _dims;
};


class SliceAxis {
public:
    SliceAxis(const NodeRef &input, ShapeDim axis,
              ShapeDim range_start, ShapeDim range_end);

    const Shape& shape() const { return _result_shape; }
    ArrayType dtype() const { return _result_dtype; }

    MultiArrayRef forward(const MultiArrayRef &value) const;
    std::string lh_name() const { return "SliceAxis("; }
    std::string rh_name() const;

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;;

    bool use_in_back_propagation() const { return true; }

private:
    Shape _result_shape;
    ArrayType _result_dtype;
    ShapeDim _axis;
    Range _range;
};

/**
 * Copies the first node in the midst of a copy of a second node
 * (which must have the same size or larger) along some axis,
 * overwriting previous values.
 * Useful for back-propagation through slicing.
 */
class ProjectOnto {
public:
    ProjectOnto(const NodeRef &input, const NodeRef &to_node,
                ShapeDim axis, ShapeDim dest_range_start);

    const Shape& shape() const { return _result_shape; }
    ArrayType dtype() const { return _result_dtype; }

    MultiArrayRef forward(const MultiArrayRef &left,
                          const MultiArrayRef &right) const;
    std::string name() const;

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const;;

    bool use_in_back_propagation() const { return true; }

private:
    Shape _result_shape;
    ArrayType _result_dtype;
    ShapeDim _axis;
    ShapeDim _dest_range_start;
};

/** Concatenates multiple nodes along a given axis */
class Concatenate : public BaseNode {
public:

    MultiArrayRef eval(Context &context, ExecutionCache &cache) const override;;

    const NodeRef apply_chain_rule(
            const NodeRef &wrt_input,
            const NodeRef &d_target_wrt_this,
            const NodeRefList &all_inputs) const override;;

    std::string to_string() const override;
    std::string repr() const override;

    NodeRefList inputs() const override { return _all_nodes; };

    MultiArrayRef forward(BufferPoolRef &pool,
                          const ArrayRefList &evaluated_inputs) const;

    static NodeRef make(const NodeRefList &nodes, ShapeDim axis = -1) {
        return std::static_pointer_cast<BaseNode>(
            std::shared_ptr<Concatenate>(new Concatenate(nodes, axis)));
    }
private:
    const NodeRefList _all_nodes;
    ShapeDim _axis;

    explicit Concatenate(const NodeRefList &nodes, ShapeDim axis = -1);
};

} // namespace

#endif //AVALANCHE_SHAPENODE_H
