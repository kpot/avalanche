#include "avalanche/shape_nodes.h"
#include "avalanche/base_ops_nodes.h"
#include "avalanche/math_ops/simple_arithemic.h"
#include "avalanche/terminal_nodes.h"

namespace avalanche {

const NodeRef ShapeOf::apply_chain_rule(const NodeRef &wrt_input,
                                          const NodeRef &d_target_wrt_this,
                                          const NodeRefList &all_inputs) const {
    return nullptr;
}

MultiArrayRef ShapeOf::forward(const MultiArrayRef &value) const {
    auto pool = value->buffer_unsafe()->pool();
    auto array = pool->make_array(
        Shape({static_cast<ShapeDim>(value->shape().rank())}),
        dtype());
    array->add_dependencies({value});
    array->buffer_unsafe()->write_from_vector(value->shape().dims());
    return array;
}


Reshape::Reshape(const NodeRef &input, const Shape &new_shape)
    :_new_shape{input->shape().reshape(new_shape.dims())},
     _result_dtype{input->dtype()}
{
}

MultiArrayRef Reshape::forward(const MultiArrayRef &value) const {
    return value->reshape(_new_shape.dims());
}

const NodeRef Reshape::apply_chain_rule(const NodeRef &wrt_input,
                                        const NodeRef &d_target_wrt_this,
                                        const NodeRefList &all_inputs) const {
    return FU<Reshape>(
        F<Multiply>(d_target_wrt_this,
                    Constant::ones(_new_shape, _result_dtype)),
        wrt_input->shape());
}

MultiArrayRef ProductOfDims::forward(const MultiArrayRef &value) const {
    // TODO: Writing a scalar to a GPU each time is terribly ineffective
    auto pool = value->buffer_unsafe()->pool();
    auto result = pool->make_array(Shape(), _result_dtype);
    result->set_label("ProductOfDims");
    ShapeDim product = 1;
    if (_dims.empty()) {
        for (auto dim: value->shape().dims()) {
            product *= dim;
        }
    } else {
        for (auto i: _dims) {
            product *= value->shape().dim(i);
        }
    }
    std::uint64_t casted_product = cast_to_value_of_array_type(
        _result_dtype, product);
    auto event = result->buffer_unsafe()->write_data(
        &casted_product, array_type_size(_result_dtype));
    // We have to wait to make sure the local buffers have been transferred
    // before the function ends
    result->wait_until_ready();
    return result;
}

} // namespace
