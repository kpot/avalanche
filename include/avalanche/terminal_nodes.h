#ifndef AVALANCHE_TERMINAL_NODES_H
#define AVALANCHE_TERMINAL_NODES_H

#include <string>
#include <functional>
#include <utility>
#include "avalanche/BaseNode.h"
#include "avalanche/Context.h"
#include "ExecutionCache.h"

namespace avalanche {

using Initializer = std::function<MultiArrayRef(Context &context)>;


/**
 * Creates an initializer which can be used to automatically initialize
 * any variable the first time it's used.
 * @tparam T type of the data
 * @param data a vector of data that should be written into the variable.
 * @param shape shape of the data. Must match the shape of the variable.
 * @return a new initializer object
 */
template <typename T>
Initializer value_initializer(
        const std::vector<T> &data,
        const Shape &shape) {
    Initializer initializer = [data, shape](Context &context) {
        auto result = context.device_pool()->make_array(shape, dtype_of_static_type<T>);
        result->write_from_vector(data);
        return result;
    };
    return initializer;
}


class Variable : public BaseNode {
public:
    const std::string name;

    // A variable can be evaluated only if has an associated MultiArray
    // within the current context.
    // In comparison with TensorFlow, the MultiArray is the tf.Variable
    // (some data living on GPU) and the avalanche::Variable is the equivalent
    // of a tf.placeholder - something that we use to indicate which data
    // we should actually use as the input.
    MultiArrayRef eval(Context &context, ExecutionCache &cache) const override;

    std::string to_string() const override {
        return name;
    }

    std::string repr() const override {
        return format_repr("Variable", name);
    }

    NodeRefList inputs() const override {
        return NodeRefList();
    }

    const NodeRef
    apply_chain_rule(const NodeRef &wrt_input, const NodeRef &d_target_wrt_this,
                     const NodeRefList &all_inputs) const override {
        return d_target_wrt_this;
    }

    static NodeRef make(const std::string &name,
                        const std::vector<ShapeDim> &shape_dims,
                        ArrayType dtype=ArrayType::float32,
                        Initializer initializer=nullptr) {
        Variable *raw_ptr = new Variable(
            name, std::move(initializer), Shape(shape_dims), dtype);
        return std::static_pointer_cast<BaseNode>(
            std::shared_ptr<Variable>(raw_ptr));
    }

private:
    Initializer _initializer;

    explicit Variable(std::string name, Initializer initializer,
                      const Shape &shape, ArrayType dtype=ArrayType::float32)
        :name{std::move(name)}, _initializer{std::move(initializer)}
    {
        set_shape(shape);
        set_dtype(dtype);
    }
};


class Constant : public BaseNode {
public:

    explicit Constant(std::string name, Initializer initializer,
                      Shape shape, ArrayType dtype)
        :_initializer{initializer},
         _name{name}
    {
        set_dtype(dtype);
        set_shape(shape);
    }

    MultiArrayRef eval(Context &context, ExecutionCache &cache) const override;

    std::string to_string() const override {
        return "Constant(" + _name + ")";
    }

    std::string repr() const override {
        return format_repr("Constant", _name);
    }

    NodeRefList inputs() const override {
        return NodeRefList();
    }

    const NodeRef
    apply_chain_rule(const NodeRef &wrt_input, const NodeRef &d_target_wrt_this,
                     const NodeRefList &all_inputs) const override {
        return d_target_wrt_this;
    }

    static const NodeRef fill(Shape shape, ArrayType dtype, float value);

    template <typename T>
    static const NodeRef scalar(T value) {
        return fill(Shape(), dtype_of_static_type<T>,
                    static_cast<float>(value));
    }
    static const NodeRef scalar(ArrayType dtype, float value) {
        return fill(Shape(), dtype, value);
    }

    template <typename T>
    static const NodeRef tensor(const std::vector<T> &value,
                                const Shape &shape) {
        if (shape.size() != value.size()) {
            throw std::invalid_argument(
                "The length of the vector must match the size of the shape");
        }
        return std::make_shared<Constant>(
            "tensor",
            [shape, value](Context &context)->MultiArrayRef {
                auto result = context.device_pool()->make_array(
                    shape, dtype_of_static_type<T>);
                // write_from_vector will update the result's completion event
                result->buffer_unsafe()->write_from_vector(value);
                return result;
            },
            shape,
            dtype_of_static_type<T>);
    }

    static const NodeRef ones(Shape shape, ArrayType dtype) {
        return fill(shape, dtype, 1);
    }

    static const NodeRef one(ArrayType dtype) {
        return fill(Shape(), dtype, 1);
    }

    static const NodeRef zero(ArrayType dtype) {
        return fill(Shape(), dtype, 0);
    }

    static const NodeRef ones_like(const NodeRef &other_node) {
        return ones(other_node->shape(), other_node->dtype());
    }

    static const NodeRef minus_one(Shape shape, ArrayType dtype) {
        return fill(shape, dtype, -1);
    }

    static const NodeRef zeros(Shape shape, ArrayType dtype) {
        return fill(shape, dtype, 0);
    }

    static const NodeRef zeros_like(const NodeRef &other_node) {
        return zeros(other_node->shape(), other_node->dtype());
    }

private:
    Initializer _initializer;
    std::string _name;

};


} // namespace


#endif //AVALANCHE_TERMINAL_NODES_H
