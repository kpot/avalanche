#include <sstream>

#include <fmt/format.h>

#include "avalanche/Shape.h"
#include "avalanche/shape_nodes.h"
#include "avalanche/base_ops_nodes.h"
#include "avalanche/math_ops/simple_arithemic.h"
#include "avalanche/terminal_nodes.h"
#include "avalanche/math_ops/messages.h"

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
    array->buffer_unsafe()->write_from_vector(value->shape().dims(), 0);
    return array;
}


Reshape::Reshape(const NodeRef &input, const Shape &new_shape)
    :_new_shape{new_shape},
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
        &casted_product, array_type_size(_result_dtype), 0);
    // We have to wait to make sure the local buffers have been transferred
    // before the function ends
    result->wait_until_ready();
    return result;
}

Concatenate::Concatenate(const NodeRefList &nodes, ShapeDim axis)
    :_all_nodes{nodes},
     _axis{0} // will be reassigned later
{
    if (_all_nodes.empty()) {
        throw std::invalid_argument("Nothing to concatenate");
    }

    auto first_node_shape = _all_nodes[0]->shape();
    auto shape_rank = first_node_shape.rank();
    std::vector<ShapeDim> result_shape_dims = first_node_shape.dims();
    // Validating axis
    _axis = static_cast<ShapeDim>(first_node_shape.dim_real_index(axis));
    if (_axis >= first_node_shape.rank()) {
        throw std::invalid_argument(
            fmt::format(
                "Cannot concatenate along given axis {} because it "
                "exceeds the shape ({}) of at least one of the nodes.",
                _axis, first_node_shape.to_string()));
    }
    // Checking that all arrays have identical rank
    for (const auto &node: _all_nodes) {
        if (node->shape().rank() != shape_rank) {
            throw std::invalid_argument(
                "The shapes of all nodes must have the same rank");
        }
    }
    // Estimating the size of the concatenated dimension
    ShapeDim axis_final_size = 0;
    for (const auto &node: _all_nodes) {
        auto d = node->shape().dim(_axis);
        if (d != UnknownDim) {
            axis_final_size += d;
        } else {
            // without that we cannot have back-prop for concatenation
            throw std::invalid_argument(
                fmt::format(
                    "Dimension {} in node {} (shape {}) must be fully defined.",
                    _axis, node->repr(), node->shape().to_string()));
        }
    }
    // Validating the rest and estimating what the resulting shape would be
    auto result_dtype = _all_nodes[0]->dtype();
    result_shape_dims[_axis] = UnknownDim; // for infer_elemwise_shape to work
    for (const auto &node: _all_nodes) {
        if (shape_rank != node->shape().rank()) {
            throw std::invalid_argument(
                fmt::format(
                    "Cannot concatenate nodes of different ranks: {} and {}",
                    _all_nodes[0]->repr(), node->repr()));
        }
        if (result_dtype != node->dtype()) {
            throw std::invalid_argument(
                fmt::format(
                    "Cannot concatenate nodes of different types: {} and {}",
                    _all_nodes[0]->repr(), node->repr()));
        }
        if (!node->shape().agrees_with(result_shape_dims)) {
            throw std::invalid_argument(
                fmt::format(
                    "For concatenation to work, every node must have identical "
                    "known dimensions except the axis being concatenated. "
                    "These two shapes however do not agree: {} and {}",
                    node->shape().to_string(),
                    Shape::dims_to_string(result_shape_dims)));
        }
        // We try to figure out the resulting shape as best as possible
        // using all the nodes available. If one node has shape (?, ?)
        // and another (5, 2), and the axis = -1, the resulting shape will
        // be (5, ?). We update this estimation with each iteration of the loop
        result_shape_dims = ElemWiseBinaryOp::infer_elemwise_shape(
            result_shape_dims, node->shape()).dims();
        // for infer_elemwise_shape to work
        result_shape_dims[_axis] = UnknownDim;
    }
    result_shape_dims[_axis] = axis_final_size;
    set_shape(Shape(result_shape_dims));
    set_dtype(result_dtype);
}

MultiArrayRef Concatenate::eval(Context &context, ExecutionCache &cache) const {
    MultiArrayRef result;
    if (!cache.get(id, result)) {
        ArrayRefList evaluated_inputs;
        for (auto const &node: _all_nodes) {
            evaluated_inputs.emplace_back(std::move(node->eval(context, cache)));
        }
        BufferPoolRef pool = context.device_pool();
        result = forward(pool, evaluated_inputs);
        cache.put(id, result);
    }
    return result;
}

MultiArrayRef Concatenate::forward(BufferPoolRef &pool,
                                   const ArrayRefList &evaluated_inputs) const {
    // Calculating the result's shape
    auto final_axis_size = 0;
    auto required_shape_dims = evaluated_inputs[0]->shape().dims();
    auto elem_size = array_type_size(evaluated_inputs[0]->dtype());
    required_shape_dims[_axis] = UnknownDim;
    for (const auto &array: evaluated_inputs) {
        if (!array->shape().agrees_with(required_shape_dims)) {
            throw std::invalid_argument(
                fmt::format("One of the arrays for concatenation doesn't have "
                            "the right shape: {} instead of expected {}",
                            array->shape().to_string(),
                            Shape::dims_to_string(required_shape_dims)));
        }
        final_axis_size += array->shape().dim(_axis);
    }
    required_shape_dims[_axis] = final_axis_size;
    Shape required_shape(required_shape_dims);
    // Let's assume that each array has shape
    // (d_1, d_2, ..., d_axis, ... d_n-1, d_n)
    // We move to another view, seeing each array as a 3D array the size of
    // (d1*d2..., d_axis, ... *d_n-1*d_n) = (outer_size, d_axis, inner_size)
    // where the outer_size and the inner_size are the same for all arrays
    // being processed and d_axis is unique for each.
    cl::size_type number_of_outer_blocks = 1;
    for (ShapeDim i = 0; i < _axis; ++i) {
        number_of_outer_blocks *= required_shape.dim(i);
    }

    cl::size_type inner_block_size = 1;
    for (ShapeDim i = _axis + 1; i < required_shape.rank(); ++i) {
        inner_block_size *= required_shape.dim(i);
    }
    // Preparing the output buffer
    auto result = pool->make_array(required_shape, dtype());
    result->add_dependencies(evaluated_inputs);
    // Doing the copying itself
    cl::size_type buffer_sub_offset = 0;
    std::vector<cl::Event> copying_is_done_events;
    for (const auto &array: evaluated_inputs) {
        cl::array<cl::size_type, 3> src_origin({array->buffer_offset() * elem_size, 0, 0});
        cl::array<cl::size_type, 3> dst_origin({buffer_sub_offset * elem_size, 0, 0});

        cl::size_type array_block_size = array->shape().dim(_axis) * inner_block_size;

        cl::array<cl::size_type, 3> region(
            {array_block_size * elem_size, number_of_outer_blocks, 1});
        std::vector<cl::Event> events_to_wait(
            {array->buffer_unsafe()->completion_event()});
        cl::Event ready_event;
        pool->cl_queue().enqueueCopyBufferRect(
            array->cl_buffer_unsafe(),
            result->cl_buffer_unsafe(),
            src_origin,
            dst_origin,
            region,
            0,
            0,
            final_axis_size * inner_block_size * elem_size,
            0,
            &events_to_wait,
            &ready_event);
        copying_is_done_events.emplace_back(std::move(ready_event));
        buffer_sub_offset += array_block_size;
    }
    // Here we have multiple events to wait for, but the buffer
    // can monitor only one, so we give "pack" all events into one
    cl::Event one_event_for_everything;
    pool->cl_queue().enqueueMarkerWithWaitList(
        &copying_is_done_events, &one_event_for_everything);
    result->set_completion_event(one_event_for_everything);
    return result;
}

std::string Concatenate::to_string() const {
    std::ostringstream out;
    out << "(Concatenation along axis " << _axis << " of ";
    for (auto const &node: _all_nodes) {
        out << node->to_string() << ", ";
    }
    out << ")";
    return out.str();
}

std::string Concatenate::repr() const {
    return format_repr("Concatenate", "");
}

const NodeRef Concatenate::apply_chain_rule(const NodeRef &wrt_input,
                                            const NodeRef &d_target_wrt_this,
                                            const NodeRefList &all_inputs) const {
    ShapeDim input_offset = 0;
    for (auto const &node: all_inputs) {
        if (node == wrt_input) {
            break;
        }
        input_offset += node->shape().dim(_axis);
    }
    return FU<SliceAxis>(
        d_target_wrt_this, _axis,
        input_offset, input_offset + wrt_input->shape().dim(_axis) - 1);
}

SliceAxis::SliceAxis(const NodeRef &input, ShapeDim axis, ShapeDim range_start,
                     ShapeDim range_end)
    :_result_dtype{input->dtype()}
{
    input->shape().normalize_range(axis, {range_start, range_end},
                                   _axis, _range);
    _axis = static_cast<ShapeDim>(input->shape().dim_real_index(axis));
    auto result_shape_dims = input->shape().dims();
    if (result_shape_dims[_axis] != UnknownDim) {
        result_shape_dims[_axis] = _range.end - _range.start + 1;
    }
    _result_shape = Shape(result_shape_dims);
}

MultiArrayRef SliceAxis::forward(const MultiArrayRef &value) const {
    ShapeDim real_axis;
    Range real_range;
    value->shape().normalize_range(_axis, _range, real_axis, real_range);
    auto result_shape_dims = value->shape().dims();
    result_shape_dims[real_axis] = real_range.end - real_range.start + 1;
    cl::size_type inner_block_size = static_cast<cl::size_type>(
        Shape::dims_product(
            value->shape().dims(), real_axis + 1,
            static_cast<ShapeDim>(value->shape().rank()) - 1));
    if (real_axis == 0) {
        // It's a continuous slice, we just need to calculate an offset
        // and return a new array linked to the same buffer with the offset
        return MultiArray::from_buffer(
            value->buffer_unsafe(),
            Shape(result_shape_dims),
            value->dtype(),
            value->buffer_offset() + real_range.start * inner_block_size);
    } else {
        // It's a strided slice, we need to make a copy
        cl::size_type num_outer_blocks = static_cast<cl::size_type>(
            Shape::dims_product(value->shape().dims(), 0, real_axis - 1));
        auto elem_size = array_type_size(value->dtype());
        auto pool = value->buffer_unsafe()->pool();
        auto result = pool->make_array(Shape(result_shape_dims), dtype());
        result->add_dependencies({value});

        cl::array<cl::size_type, 3> src_origin({(value->buffer_offset() + real_range.start * inner_block_size) * elem_size, 0, 0});
        cl::array<cl::size_type, 3> dst_origin({0, 0, 0});

        cl::size_type copied_block_size = result_shape_dims[real_axis] * inner_block_size;

        cl::array<cl::size_type, 3> region({copied_block_size * elem_size, num_outer_blocks, 1});
        std::vector<cl::Event> events_to_wait({value->buffer_unsafe()->completion_event()});
        cl::Event ready_event;
        pool->cl_queue().enqueueCopyBufferRect(
            value->cl_buffer_unsafe(),
            result->cl_buffer_unsafe(),
            src_origin,
            dst_origin,
            region,
            inner_block_size * value->shape().dim(real_axis) * elem_size,
            0,
            0,
            0,
            &events_to_wait,
            &ready_event);
        result->set_completion_event(ready_event);
        return result;
    }
}

std::string SliceAxis::rh_name() const {
    return fmt::format(", axis={}, start={}, end={})",
                       _axis, _range.start, _range.end);
}

const NodeRef SliceAxis::apply_chain_rule(const NodeRef &wrt_input,
                                          const NodeRef &d_target_wrt_this,
                                          const NodeRefList &all_inputs) const {
    return F<ProjectOnto>(d_target_wrt_this,
                          F<NoBackProp>(Constant::zeros_like(wrt_input)),
                          _axis, _range.start);
}

ProjectOnto::ProjectOnto(const NodeRef &input, const NodeRef &to_node,
                         ShapeDim axis, ShapeDim dest_range_start)
:_result_dtype{input->dtype()},
 _dest_range_start{dest_range_start}
{
    if (to_node->dtype() != input->dtype()) {
        throw std::invalid_argument(
            fmt::format(
                "Both the projected node and the target node should have "
                "the same data types. Currently it is {} and {}.",
                array_type_name(input->dtype()),
                array_type_name(to_node->dtype())));
    }
    auto input_shape = input->shape();
    auto shape_rank = input_shape.rank();
    std::vector<ShapeDim> result_shape_dims = input_shape.dims();
    // Validating axis
    _axis = static_cast<ShapeDim>(input_shape.dim_real_index(axis));
    if (_axis >= input_shape.rank()) {
        throw std::invalid_argument(
            fmt::format(
                "Cannot project to the given axis {} because it "
                "exceeds the shape ({}) of at least one of the input nodes.",
                _axis, input_shape.to_string()));
    }
    // Checking that all arrays have identical rank
    if (to_node->shape().rank() != shape_rank) {
        throw std::invalid_argument(
            "The shapes of both projected and target nodes must "
            "have the same rank");
    }
    if (input_shape[_axis] == UnknownDim) {
        throw std::invalid_argument(
            "Input node not must have known size along the projected dimension");
    }
    // Estimating the size of the concatenated dimension
    ShapeDim axis_final_size = to_node->shape().dim(_axis);
    // Validating the rest and estimating what the resulting shape would be
    result_shape_dims[_axis] = UnknownDim; // for infer_elemwise_shape to work
    if (!input_shape.agrees_with(result_shape_dims)) {
        throw std::invalid_argument(
            fmt::format(
                "For projection to work, both projected and target node's "
                "known shape must be identical except for the axis "
                "being projected."
                "These two shapes however do not agree: {} and {}",
                input->shape().to_string(),
                Shape::dims_to_string(result_shape_dims)));
    }
    result_shape_dims = ElemWiseBinaryOp::infer_elemwise_shape(
        result_shape_dims, to_node->shape()).dims();
    result_shape_dims[_axis] = axis_final_size;
    _result_shape = Shape(result_shape_dims);
}

const NodeRef ProjectOnto::apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const {
    if (wrt_input == all_inputs[0]) {
        return FU<SliceAxis>(
            d_target_wrt_this, _axis,
            _dest_range_start,
            _dest_range_start + wrt_input->shape().dim(_axis) - 1);
    } else if (wrt_input == all_inputs[1]) {
        // Normally this branch should not be calculated, because
        // it doesn't make practical sense and likely this input is wrapped
        // with NoBackProp
        return Constant::zeros_like(wrt_input);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}

std::string ProjectOnto::name() const {
    return fmt::format("ProjectOnto(axis={}, dest_star={})",
                       _axis, _dest_range_start);
}

MultiArrayRef ProjectOnto::forward(const MultiArrayRef &left,
                                   const MultiArrayRef &right) const {
    // Finding and normalizing the final axis and destination range
    ShapeDim real_axis;
    Range real_range;
    auto dst_axis_size = right->shape().dim(_axis);
    ShapeDim norm_range_start = (
        (_dest_range_start < 0) ? dst_axis_size + _dest_range_start
                                : _dest_range_start);
    right->shape().normalize_range(
        _axis,
        {norm_range_start, norm_range_start + left->shape().dim(_axis) - 1},
        real_axis, real_range);

    // Preparing the output array of a right size depending both
    // on the left and the right nodes, to keep them alive until the copying
    // is done
    auto elem_size = array_type_size(dtype());
    auto pool = right->buffer_unsafe()->pool();
    auto result = pool->make_array(right->shape(), dtype());
    result->add_dependencies({right, left});

    // First we need to copy the right buffer into the result
    cl::Event right_is_copied;
    std::vector<cl::Event> events_to_wait({right->buffer_unsafe()->completion_event()});
    pool->cl_queue().enqueueCopyBuffer(right->cl_buffer_unsafe(), result->cl_buffer_unsafe(), left->buffer_offset() * elem_size, 0, left->size() * elem_size, &events_to_wait, &right_is_copied);
    events_to_wait.clear();

    // Now we need to copy the left buffer into the result as soon as the
    // first copying is finished
    events_to_wait.emplace_back(std::move(right_is_copied));
    cl::size_type inner_block_size = static_cast<cl::size_type>(
        Shape::dims_product(
            left->shape().dims(), real_axis + 1,
            static_cast<ShapeDim>(left->shape().rank()) - 1));
    cl::size_type num_outer_blocks = static_cast<cl::size_type>(
        Shape::dims_product(left->shape().dims(), 0, real_axis - 1));

    cl::array<cl::size_type, 3> src_origin({left->buffer_offset() * elem_size,
                                            0, 0});
    cl::array<cl::size_type, 3> dst_origin({inner_block_size * real_range.start * elem_size,
                                            0, 0});
    cl::size_type left_block_size = left->shape().dim(_axis) * inner_block_size;
    cl::array<cl::size_type, 3> region({left_block_size * elem_size,
                                        num_outer_blocks,
                                        1});
    cl::Event left_is_copied;
    pool->cl_queue().enqueueCopyBufferRect(
        left->cl_buffer_unsafe(),
        result->cl_buffer_unsafe(),
        src_origin,
        dst_origin,
        region,
        0,
        0,
        dst_axis_size * inner_block_size * elem_size,
        0,
        &events_to_wait,
        &left_is_copied);
    result->set_completion_event(left_is_copied);
    return result;
}

} // namespace
