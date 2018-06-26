#include <sstream>

#include <fmt/format.h>

#include "avalanche/Shape.h"
#include "avalanche/shape_nodes.h"
#include "avalanche/base_ops_nodes.h"
#include "avalanche/math_ops/simple_arithemic.h"
#include "avalanche/terminal_nodes.h"
#include "avalanche/math_ops/messages.h"
#include "avalanche/CodeCache.h"
#include "avalanche/opencl_utils.h"

namespace avalanche {

constexpr const std::size_t WORK_GROUP_SIZE = 64;


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
    return ReshapeLike::make(
        F<Multiply>(d_target_wrt_this,
                    Constant::ones(_new_shape, _result_dtype)),
        wrt_input);
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
    return format_repr("Concatenate", "", fmt::format("axis: {}", _axis));
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
                     ShapeDim range_end, bool keep_dims)
    :_keep_dims{keep_dims},
     _result_dtype{input->dtype()}
{
    input->shape().normalize_range(axis, {range_start, range_end},
                                   _axis, _range);
    _axis = static_cast<ShapeDim>(input->shape().dim_real_index(axis));
    auto result_shape_dims = input->shape().dims();
    if (result_shape_dims[_axis] != UnknownDim) {
        result_shape_dims[_axis] = _range.end - _range.start + 1;
    }
    if (!_keep_dims && result_shape_dims[_axis] == 1) {
        result_shape_dims.erase(result_shape_dims.begin() + _axis);
    }
    _result_shape = Shape(result_shape_dims);
}

MultiArrayRef SliceAxis::forward(const MultiArrayRef &value) const {
    ShapeDim real_axis;
    Range real_range;
    value->shape().normalize_range(_axis, _range, real_axis, real_range);
    auto result_shape_dims_kept = value->shape().dims();
    result_shape_dims_kept[real_axis] = real_range.end - real_range.start + 1;
    auto result_shape_dims_cut = result_shape_dims_kept;
    if (!_keep_dims && result_shape_dims_kept[_axis] == 1) {
        result_shape_dims_cut.erase(result_shape_dims_cut.begin() + _axis);
    }
    cl::size_type inner_block_size = static_cast<cl::size_type>(
        Shape::dims_product(
            value->shape().dims(), real_axis + 1,
            static_cast<ShapeDim>(value->shape().rank()) - 1));
    if (real_axis == 0) {
        // It's a continuous slice, we just need to calculate an offset
        // and return a new array linked to the same buffer with the offset
        return MultiArray::from_buffer(
            value->buffer_unsafe(),
            Shape(result_shape_dims_kept),
            value->dtype(),
            value->buffer_offset() + real_range.start * inner_block_size);
    } else {
        // It's a strided slice, we need to make a copy
        cl::size_type num_outer_blocks = static_cast<cl::size_type>(
            Shape::dims_product(value->shape().dims(), 0, real_axis - 1));
        auto elem_size = array_type_size(value->dtype());
        auto pool = value->buffer_unsafe()->pool();
        auto result = pool->make_array(Shape(result_shape_dims_cut), dtype());
        result->add_dependencies({value});

        cl::array<cl::size_type, 3> src_origin(
            {(value->buffer_offset() + real_range.start * inner_block_size)
             * elem_size,
             0, 0});
        cl::array<cl::size_type, 3> dst_origin({0, 0, 0});

        cl::size_type copied_block_size =
            result_shape_dims_kept[real_axis] * inner_block_size;
        cl::array<cl::size_type, 3> region({copied_block_size * elem_size,
                                            num_outer_blocks, 1});
        std::vector<cl::Event> events_to_wait(
            {value->buffer_unsafe()->completion_event()});
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
    return fmt::format(", axis={}, start={}, end={}, keep_dims={})",
                       _axis, _range.start, _range.end, _keep_dims);
}

const NodeRef SliceAxis::apply_chain_rule(const NodeRef &wrt_input,
                                          const NodeRef &d_target_wrt_this,
                                          const NodeRefList &all_inputs) const {
    auto d_target = _keep_dims ? d_target_wrt_this : FU<ExpandDims>(d_target_wrt_this, _axis);
    return F<ProjectOnto>(d_target,
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
    pool->cl_queue().enqueueCopyBuffer(
        right->cl_buffer_unsafe(), result->cl_buffer_unsafe(),
        right->buffer_offset() * elem_size, 0, right->size() * elem_size,
        &events_to_wait, &right_is_copied);
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

Shape tile_forward_shape(const Shape &shape,
                         const std::vector<ShapeDim> &multiplies) {
    std::vector<ShapeDim> result_shape_dims = shape.dims();
    for (size_t i = 0; i < shape.rank(); ++i) {
        result_shape_dims[i] *= multiplies[i];
    }
    return Shape(result_shape_dims);
}

Shape tile_backward_shape(const Shape &shape,
                         const std::vector<ShapeDim> &multiplies) {
    std::vector<ShapeDim> result_shape_dims = shape.dims();
    for (size_t i = 0; i < shape.rank(); ++i) {
        result_shape_dims[i] /= multiplies[i];
    }
    return Shape(result_shape_dims);
}

std::vector<cl_ulong> inner_block_sizes(const Shape &shape) {
    std::vector<cl_ulong> result(shape.rank());
    cl_ulong prod = 1;
    for (ShapeDim i = static_cast<ShapeDim>(shape.rank()) - 1; i >= 0; --i) {
        result[i] = prod;
        prod *= shape[i];
    }
    return result;
}


std::string tiling_kernel_name(ArrayType orig_dtype, ArrayType tiled_dtype,
                               bool forward) {
    return fmt::format(
        "tiling_{}_{}_{}",
        forward ? "forward" : "backward",
        cl_type_name_of_array(orig_dtype),
        cl_type_name_of_array(tiled_dtype));
}


/**
 * Generates a kernel performing tiling operation. Since almost the same
 * code can be used both for the tiling itself and for calculating its
 * derivative, this function can generate code for both operations,
 * depending on the parameter `forward`.
 *
 * The kernel maps each value of the original array to each value
 * of the destination.
 *
 * @param orig_dtype ArrayType of the original array being tiled.
 * @param tiled_dtype ArrayType of the tiled array
 * @param forward If true, the code will be generated for the forward tiling
 *    operation (which actually replicates the things). When false,
 *    the generated kernel will perform summation for all replicas of each
 *    value, outputing the result to the `origin` (so the output for the forward
 *    operation becomes the input for the backward).
 * @param work_group_size OpenCL work group size
 * @return a string containing the kernel
 */
std::string tiling_kernel_code(ArrayType orig_dtype, ArrayType tiled_dtype,
                               bool forward, int work_group_size) {

    constexpr const char *kernel_template = R"clkernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size({work_group_size}, 1, 1)))
void {kernel_name}(
         __global {orig_dtype} *origin,
         __global {tiled_dtype} *tiled,
         uint rank,
         __constant ulong *orig_shape,
         __constant ulong *multiplies,
         __constant ulong *orig_inner_sizes,
         __constant ulong *tiled_inner_sizes,
         __local ulong *all_counters,
         __local ulong *all_current_locations,
         ulong origin_size) {{
    if (get_global_id(0) >= origin_size) return;
    // choosing which part of the scratchpad can be used by this thread
    __local ulong *counters = all_counters + rank * get_local_id(0);
    __local ulong *current_location = all_current_locations + rank * get_local_id(0);
    // Identifying coordinates of the current thread within the original array
    for (uint i = 0; i < rank; ++i) counters[i] = 0;
    ulong addr_left = get_global_id(0);
    for (uint i = 0; i < rank; ++i) {{
        current_location[i] = addr_left / orig_inner_sizes[i];
        addr_left = addr_left % orig_inner_sizes[i];
    }}

    {initialization}

    // Iterating through all possible replicated blocks, storing block
    // coordinates in counters
    char the_end = 0;
    while (!the_end) {{
        // calculating the offset for this thread within
        // the next replicated block
        ulong tiled_offset = 0;
        for (uint ci = 0; ci < rank; ++ci) {{
            tiled_offset += (counters[ci] * orig_shape[ci] + current_location[ci]) * tiled_inner_sizes[ci];
        }}

        {mapping}

        // Switching to next block
        for (int ci = rank - 1; ci >= 0; --ci) {{
            ++counters[ci];
            if (counters[ci] < multiplies[ci]) {{
                break;
            }} else {{
                the_end = (ci == 0);
                counters[ci] = 0;
            }}
        }}
    }}

    {finalization}
}}
    )clkernel";

    std::string initialization, mapping, finalization;
    if (forward) {
        initialization = fmt::format(
            "{} replicate_value = origin[get_global_id(0)];",
            cl_type_name_of_array(orig_dtype));
        mapping = "tiled[tiled_offset] = replicate_value;";
        finalization = "";
    } else {
        initialization = fmt::format(
            "{} accumulated_value = 0;",
            cl_type_name_of_array(orig_dtype));
        mapping = "accumulated_value += tiled[tiled_offset];";
        finalization = "origin[get_global_id(0)] = accumulated_value;";
    }

    return fmt::format(
        kernel_template,
        fmt::arg("kernel_name",
                 tiling_kernel_name(orig_dtype, tiled_dtype, forward)),
        fmt::arg("orig_dtype", cl_type_name_of_array(orig_dtype)),
        fmt::arg("tiled_dtype", cl_type_name_of_array(tiled_dtype)),
        fmt::arg("work_group_size", work_group_size),
        fmt::arg("initialization", initialization),
        fmt::arg("mapping", mapping),
        fmt::arg("finalization", finalization));
}


Tile::Tile(const NodeRef &input, const std::vector<ShapeDim> &multiples,
           bool run_forward)
:_result_dtype{input->dtype()},
 _multiples{multiples},
 _kernel_name{
    tiling_kernel_name(input->dtype(), input->dtype(), run_forward)},
 _kernel_source{
    tiling_kernel_code(input->dtype(), input->dtype(),
                       run_forward, WORK_GROUP_SIZE)},
 _is_forward_op{run_forward}
{
    if (input->shape().rank() != multiples.size()) {
        throw std::invalid_argument(
            fmt::format("Cannot tile node {} because the list of multiplies has"
                        " different size ({}) from the rank of the node ({})",
                        input->repr(), multiples.size(),
                        input->shape().rank()));
    }
    for (auto m: multiples) {
        if (m < 1) {
            throw std::invalid_argument(
                fmt::format("Cannot tile node {}: the list of multiplies"
                            "can contain only values >= 1. Currently: {}",
                            input->repr(),
                            Shape::dims_to_string(multiples, false)));
        }
    }
    if (_is_forward_op) {
        std::vector<ShapeDim> result_dims = input->shape().dims();
        for (std::size_t i = 0; i < multiples.size(); ++i) {
            if (result_dims[i] != UnknownDim) {
                result_dims[i] *= multiples[i];
            }
        }
        _tiled_shape = Shape(result_dims);
        _orig_shape = input->shape();
    } else {
        std::vector<ShapeDim> result_dims = input->shape().dims();
        for (std::size_t i = 0; i < multiples.size(); ++i) {
            if (result_dims[i] != UnknownDim) {
                result_dims[i] /= multiples[i];
            }
        }
        _tiled_shape = input->shape();
        _orig_shape = Shape(result_dims);
    }
}

std::string Tile::rh_name() const {
    return fmt::format(", multiplies={}, forward={})",
                       Shape::dims_to_string(_multiples, false),
                       _is_forward_op);
}

MultiArrayRef Tile::forward(const MultiArrayRef &value) const {
    auto rank = value->shape().rank();
    if (rank != _multiples.size()) {
        // just to be sure
        std::invalid_argument(
            "The number of multiplies must match the node's rank");
    }
    Shape result_shape, orig_shape;
    std::vector<cl_ulong> orig_inner_sizes;
    std::vector<cl_ulong> tiled_inner_sizes;
    if (_is_forward_op) {
        result_shape = tile_forward_shape(value->shape(), _multiples);
        orig_inner_sizes = inner_block_sizes(value->shape());
        tiled_inner_sizes = inner_block_sizes(result_shape);
        orig_shape = value->shape();
    } else {
        result_shape = tile_backward_shape(value->shape(), _multiples);
        orig_inner_sizes = inner_block_sizes(result_shape);
        tiled_inner_sizes = inner_block_sizes(value->shape());
        orig_shape = result_shape;
    }
    auto pool = value->buffer_unsafe()->pool();
    auto multiplies_buffer = pool->reserve_buffer_for_vector(_multiples);
    auto orig_shape_buffer = pool->reserve_buffer_for_vector(orig_shape.dims());
    auto orig_inner_buffer = pool->reserve_buffer_for_vector(orig_inner_sizes);
    auto tiled_inner_buffer = pool->reserve_buffer_for_vector(tiled_inner_sizes);
    auto constants_are_ready = make_event_list(
        {multiplies_buffer->write_from_vector(_multiples, 0),
         orig_shape_buffer->write_from_vector(orig_shape.dims(), 0),
         orig_inner_buffer->write_from_vector(orig_inner_sizes, 0),
         tiled_inner_buffer->write_from_vector(tiled_inner_sizes, 0)});
    auto all_data_are_ready = make_event_list(
        {value->buffer_unsafe()->completion_event()});
    std::copy(constants_are_ready.begin(), constants_are_ready.end(),
              std::back_inserter(all_data_are_ready));
    auto queue = pool->cl_queue();
    auto result = pool->make_array(result_shape, dtype());
    result->add_dependencies({value});
    result->add_dependencies({multiplies_buffer, orig_shape_buffer,
                              orig_inner_buffer, tiled_inner_buffer});
    auto program = CodeCache::get_default().get_program(
        pool->cl_context(), queue,
        _kernel_name, _kernel_source, "");
    auto kernel = cl::Kernel(program, _kernel_name.c_str());
    kernel.setArg(0, _is_forward_op ? value->cl_buffer_unsafe()
                                    : result->cl_buffer_unsafe());
    kernel.setArg(1, _is_forward_op ? result->cl_buffer_unsafe()
                                    : value->cl_buffer_unsafe());
    kernel.setArg(2, static_cast<cl_uint>(value->shape().rank()));
    kernel.setArg(3, orig_shape_buffer->cl_buffer_unsafe());
    kernel.setArg(4, multiplies_buffer->cl_buffer_unsafe());
    kernel.setArg(5, orig_inner_buffer->cl_buffer_unsafe());
    kernel.setArg(6, tiled_inner_buffer->cl_buffer_unsafe());
    kernel.setArg(7, static_cast<cl_ulong>(rank * sizeof(cl_ulong) * WORK_GROUP_SIZE), nullptr);
    kernel.setArg(8, static_cast<cl_ulong>(rank * sizeof(cl_ulong) * WORK_GROUP_SIZE), nullptr);
    kernel.setArg(9, static_cast<cl_ulong>(value->shape().size()));
    const auto work_items = make_divisible_by(WORK_GROUP_SIZE, value->shape().size());
    cl::Event operation_is_done;
    queue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange(work_items),
        cl::NDRange(WORK_GROUP_SIZE),
        &all_data_are_ready,
        &operation_is_done);
    cl::WaitForEvents(constants_are_ready);
    result->set_completion_event(operation_is_done);
    operation_is_done.wait();
    return result;
}

const NodeRef Tile::apply_chain_rule(const NodeRef &wrt_input,
                                     const NodeRef &d_target_wrt_this,
                                     const NodeRefList &all_inputs) const {
    if (_is_forward_op) {
        return FU<Tile>(d_target_wrt_this, _multiples, false);
    } else {
        throw std::runtime_error("Not implemented");
    }
}

const NodeRef NoBackProp::apply_chain_rule(const NodeRef &wrt_input,
                                           const NodeRef &d_target_wrt_this,
                                           const NodeRefList &all_inputs) const {
    return Constant::zeros_like(wrt_input);
}

ShapeOf::ShapeOf(const NodeRef &input)
    :Constant(
    "Shape of " + input->repr(),
    Initializer{
        [](Context &context, ExecutionCache &cache,
           ArrayRefList &dependencies) {
            auto value = dependencies[0];
            auto pool = value->buffer_unsafe()->pool();
            auto array = pool->make_array(
                Shape({static_cast<ShapeDim>(value->shape().rank())}),
                ShapeOf::DType);
            array->add_dependencies({value});
            array->buffer_unsafe()->write_from_vector(
                value->shape().dims(), 0);
            // we preserve the shape inside the host memory
            // to simplify further cache checking
            array->write_metadata(value->shape().dims());
            return array;
        },
        [](const MultiArrayRef &cached_value,
           const ArrayRefList &dependencies) {
            // comparing cached value agains the current shape
            auto dims_from_metadata = extract_shape_from_metadata(cached_value);
            return dims_from_metadata == dependencies[0]->shape().dims();
        },
        DType,
        {F<NoBackProp>(input)}
    },
    {static_cast<ShapeDim>(input->shape().rank())},
    DType)
{
}

std::vector<ShapeDim> ShapeOf::extract_shape_from_metadata(
        const MultiArrayRef &cached_value) {
    std::vector<ShapeDim> dims_from_metadata;
    cached_value->read_metadata(dims_from_metadata);
    return dims_from_metadata;
}

ExpandDims::ExpandDims(const NodeRef &input, ShapeDim axis)
    :_result_dtype{input->dtype()}
{
    _axis = normalize_axis(input->shape(), axis);
    auto result_shape_dims = input->shape().dims();
    result_shape_dims.insert(result_shape_dims.begin() + _axis, 1);
    _result_shape = Shape(result_shape_dims);
}

std::string ExpandDims::rh_name() const {
    return fmt::format(", axis={})", _axis);
}

MultiArrayRef ExpandDims::forward(const MultiArrayRef &value) const {
    auto result_shape_dims = value->shape().dims();
    result_shape_dims.insert(result_shape_dims.begin() + _axis, 1);
    return value->reshape(result_shape_dims);
}

const NodeRef ExpandDims::apply_chain_rule(const NodeRef &wrt_input,
                                           const NodeRef &d_target_wrt_this,
                                           const NodeRefList &all_inputs) const {
    return FU<Squeeze>(d_target_wrt_this, _axis);
}

ShapeDim ExpandDims::normalize_axis(const Shape &shape, ShapeDim axis) {
    ShapeDim result = (axis < 0) ? static_cast<ShapeDim>(shape.rank()) + axis + 1 : axis;
    if (result < 0 || result > shape.rank()) {
        throw std::invalid_argument(
            fmt::format(
                "It is impossible to insert a dimension with index {} "
                "into the shape {}",
                axis, shape.to_string()));
    }
    return result;
}


Squeeze::Squeeze(const NodeRef &input, ShapeDim axis)
    :_result_dtype{input->dtype()}
{
    _axis = static_cast<ShapeDim>(input->shape().dim_real_index(axis));
    if (_axis < 0 || _axis > input->shape().rank() - 1) {
        throw std::invalid_argument(
            fmt::format(
                "It is impossible to remove the dimension with index {} "
                "from the node {}", axis, input->repr()));
    }
    auto result_shape_dims = input->shape().dims();
    if (result_shape_dims[_axis] != UnknownDim
            && result_shape_dims[_axis] != 1) {
        throw std::invalid_argument(
            fmt::format(
                "It is impossible to remove the dimension with index {} from "
                "the node {} because the dimension's size is greater than 1",
                axis, input->repr()));
    }
    result_shape_dims.erase(result_shape_dims.begin() + _axis);
    _result_shape = Shape(result_shape_dims);
}

std::string Squeeze::rh_name() const {
    return fmt::format(", axis={})", _axis);
}

MultiArrayRef Squeeze::forward(const MultiArrayRef &value) const {
    auto result_shape_dims = value->shape().dims();
    if (result_shape_dims[_axis] != 1) {
        throw std::invalid_argument("Cannot remove a dimension of a size > 1");
    }
    result_shape_dims.erase(result_shape_dims.begin() + _axis);
    return value->reshape(result_shape_dims);
}

const NodeRef Squeeze::apply_chain_rule(const NodeRef &wrt_input,
                                        const NodeRef &d_target_wrt_this,
                                        const NodeRefList &all_inputs) const {
    return FU<ExpandDims>(d_target_wrt_this, _axis);
}

NodeRef stack_nodes(const NodeRefList &nodes, ShapeDim axis) {
    NodeRefList reshaped_nodes;
    for (auto const &node: nodes) {
        reshaped_nodes.push_back(FU<ExpandDims>(node, axis));
    }
    return Concatenate::make(reshaped_nodes, axis);
}

ReshapeLike::ReshapeLike(const NodeRef &input, const NodeRef &like_node)
:_input{input},
 _shape_node{ShapeOf::make(like_node)},
 _replace_dims_to_ones{false}
{
    set_dtype(input->dtype());
    set_shape(like_node->shape());
}

ReshapeLike::ReshapeLike(const NodeRef &input, const NodeRef &like_node,
                         const std::vector<ShapeDim> dims_to_ones)
    :ReshapeLike(input, like_node)
{
    _replace_dims_to_ones = true;
    _dims_to_ones = dims_to_ones;
}

const NodeRef ReshapeLike::apply_chain_rule(const NodeRef &wrt_input,
                                            const NodeRef &d_target_wrt_this,
                                            const NodeRefList &all_inputs) const {
    if (wrt_input == all_inputs[0]) {
        return ReshapeLike::make(d_target_wrt_this, wrt_input);
    } else if (wrt_input == all_inputs[1]) {
        // Normally this branch should not be calculated, because
        // it doesn't make practical sense
        return Constant::zeros_like(wrt_input);
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}

NodeRefList ReshapeLike::inputs() const {
    return avalanche::NodeRefList({_input, _shape_node});
}

MultiArrayRef ReshapeLike::eval(Context &context, ExecutionCache &cache) const {
    MultiArrayRef result;
    if (!cache.get(id, result)) {
        auto input_value = _input->eval(context, cache);
        auto shape_value = _shape_node->eval(context, cache);
        auto shape_dims = ShapeOf::extract_shape_from_metadata(shape_value);

        if (_replace_dims_to_ones) {
            auto extracted_shape = Shape(shape_dims);
            auto dims_to_replace = extracted_shape.normalize_dims(_dims_to_ones);
            if (dims_to_replace.empty()) {
                // This means all dimensions must be replaced
                for (auto &d: shape_dims) {
                    d = 1;
                }
            } else {
                // only particular dimensions must be replaced
                for (auto i: dims_to_replace) {
                    shape_dims[i] = 1;
                }
            }
        }

        auto shape_is_different = input_value->shape().dims() != shape_dims;
        result = shape_is_different ? input_value->reshape(shape_dims)
                                    : input_value;
        cache.put(id, result);
    }
    return result;
}

std::string ReshapeLike::to_string() const {
    return fmt::format("({} ReshapeLike {})",
                       _input->to_string(), _shape_node->to_string());
}

NodeRef ReshapeLike::make(const NodeRef &input, const NodeRef &like_node) {
    auto *raw_ptr = new ReshapeLike(input, like_node);
    return std::static_pointer_cast<BaseNode>(
        std::shared_ptr<ReshapeLike>(raw_ptr));
}

NodeRef ReshapeLike::make(const NodeRef &input, const NodeRef &like_node,
                          const std::vector<ShapeDim> &dims_to_ones) {
    auto *raw_ptr = new ReshapeLike(input, like_node, dims_to_ones);
    return std::static_pointer_cast<BaseNode>(
        std::shared_ptr<ReshapeLike>(raw_ptr));
}

std::string ReshapeLike::repr() const {
    std::string extra;
    if (_replace_dims_to_ones) {
        extra = fmt::format(
            "like: {}, dims_to_ones: {}",
            _shape_node->to_string(),
            Shape::dims_to_string(_dims_to_ones));
    } else {
        extra = fmt::format("like: {}", _shape_node->to_string());
    }
    return format_repr("ReshapeLike", "", extra);
}

} // namespace
