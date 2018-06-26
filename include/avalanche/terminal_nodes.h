#ifndef AVALANCHE_TERMINAL_NODES_H
#define AVALANCHE_TERMINAL_NODES_H

#include <string>
#include <functional>
#include <utility>

#include "avalanche/BaseNode.h"
#include "avalanche/Context.h"
#include "avalanche/ExecutionCache.h"

namespace avalanche {

using InitializerFunc = std::function<
    MultiArrayRef(Context &context, ExecutionCache &cache,
                  ArrayRefList &dependencies)>;
using InitializerValidityCheck = std::function<
    bool(const MultiArrayRef &cached_value, const ArrayRefList &dependencies)>;


/**
 * Represents all we need to initialize a constant or a variable in runtime.
 * Can be non-functional if the code references are nullptr
 * (see operator bool())
 */
struct Initializer {
    // Generates a new value of a variable or a constant. Usually called
    // only once, after that the value gets cached
    InitializerFunc code = nullptr;
    // Checks if for some reason the existing cache of a constant has become
    // outdated. Check Constant::eval for more detail.
    InitializerValidityCheck is_cache_valid = nullptr;
    // The type of values the cache generates. Helps in validating initializers
    // before the first use
    ArrayType dtype = ArrayType::int8;
    NodeRefList dependencies;

    explicit operator bool() const {
        return static_cast<bool>(code);
    }
};


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
    Initializer initializer{
        [data, shape](Context &context, ExecutionCache &cache,
                      ArrayRefList &dependencies) {
            auto result = context.device_pool()->make_array(
                shape,
                dtype_of_static_type<T>);
            result->write_from_vector(data);
            return result;
        },
        nullptr,
        dtype_of_static_type<T>,
        {}
    };
    return initializer;
}


/** Initializes variables before the first use */
Initializer node_initializer(const NodeRef &node);


/**
 * The most important terminal node representing some permanent GPU array.
 * Comparing with TF `Variable` plays both roles of `tf.Variable`
 * and `tf.placeholder`.
 *
 * Since we can have multiple GPUs, this node doesn't actually store anything:
 * the real data attached to the `Context` class,
 * so the Variable is more like a pointer or a unique identifier for those.
 *
 * If the variable doesn't have an initializer, it will have to be initialized
 * lately by calling `Context::init`. In such case it's more like
 * a `placeholder` from TF.
 *
 * Once initialized by any means, the variable's data will keep living attached
 * to the Context until it's destroyed.
 */
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
        return format_repr("Variable", name, "");
    }

    NodeRefList inputs() const override {
        return _initializer ? _initializer.dependencies : NodeRefList();
    }

    const NodeRef
    apply_chain_rule(const NodeRef &wrt_input, const NodeRef &d_target_wrt_this,
                     const NodeRefList &all_inputs) const override {
        return d_target_wrt_this;
    }

    static NodeRef make_from_node(const std::string &name,
                                  const NodeRef &initialize_from_node);

    static NodeRef make(const std::string &name,
                        const std::vector<ShapeDim> &shape_dims,
                        ArrayType dtype=ArrayType::float32,
                        Initializer initializer={});

protected:
    Initializer _initializer;

    explicit Variable(std::string name, Initializer initializer,
                      const Shape &shape,
                      ArrayType dtype)
        :name{std::move(name)},
         _initializer{std::move(initializer)}
    {
        set_shape(shape);
        set_dtype(dtype);
    }
};


class Placeholder : public Variable {
public:

    explicit Placeholder(std::string name,
                         Initializer initializer,
                         const Shape &shape,
                         ArrayType dtype)
                             :Variable(name, initializer, shape, dtype) {}

    static NodeRef make(const std::string &name,
                        const std::vector<ShapeDim> &shape_dims,
                        ArrayType dtype=ArrayType::float32,
                        Initializer initializer={});

    std::string repr() const override {
        return format_repr("Placeholder", name, "");
    }

    MultiArrayRef eval(Context &context, ExecutionCache &cache) const override;
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
        return format_repr("Constant", _name, "");
    }

    NodeRefList inputs() const override {
        return _initializer ? _initializer.dependencies : NodeRefList();
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
        return tensor(
            "tensor",
            value.data(),
            sizeof(typename std::vector<T>::value_type) * value.size(),
            dtype_of_static_type<T>,
            shape);
    }

    static const NodeRef tensor(const std::string &name, const void *data,
                                std::size_t num_bytes, ArrayType dtype,
                                const Shape &shape);

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
        return fill_like(other_node, 1);
    }

    static const NodeRef minus_one(Shape shape, ArrayType dtype) {
        return fill(shape, dtype, -1);
    }

    static const NodeRef zeros(Shape shape, ArrayType dtype) {
        return fill(shape, dtype, 0);
    }

    static const NodeRef zeros_like(const NodeRef &other_node) {
        return fill_like(other_node, 0);
    }

    static const NodeRef zeros_like_with_type(const NodeRef &other_node,
                                              ArrayType dtype);
    static const NodeRef ones_like_with_type(const NodeRef &other_node,
                                             ArrayType dtype);

    static const NodeRef fill_shape(const NodeRef &shape_node, ArrayType dtype,
                                    float value);

    static const NodeRef fill_like(const NodeRef &other_node, float value);
    static const NodeRef fill_like_with_type(const NodeRef &other_node, ArrayType dtype, float value);

private:
    Initializer _initializer;
    std::string _name;
};


} // namespace


#endif //AVALANCHE_TERMINAL_NODES_H
