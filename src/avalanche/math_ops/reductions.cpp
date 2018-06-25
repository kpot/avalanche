#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>

#include "clblast.h"
#include <fmt/format.h>

#include "avalanche/opencl_utils.h"
#include "avalanche/CodeCache.h"
#include "avalanche/math_ops/reductions.h"
#include "avalanche/math_ops/simple_arithemic.h"
#include "avalanche/terminal_nodes.h"
#include "avalanche/math_ops/const_transformation.h"
#include "avalanche/math_ops/messages.h"
#include "avalanche/shape_nodes.h"

namespace avalanche {

constexpr char cl_sources_of_random_generators[] = {
#include "avalanche/kernels/reductions.hex"
};

constexpr std::size_t WORK_GROUP_SIZE = 64;


Reduction::Reduction(const NodeRef &input)
    : _result_shape_dims_cut{},
      _result_shape_dims_kept{},
      _result_dtype{input->dtype()},
      _dims_to_cut{},
      _keep_dims{false}
{
}

void Reduction::estimate_steps_and_dimensions(
        const Shape &input_shape,
        const std::vector<ShapeDim> &dims_to_cut,
        // Outputs
        std::vector<ReductionStep> &reduction_steps,
        Shape &result_shape_dims_cut,
        Shape &result_shape_dims_kept) const {

    if (dims_to_cut.empty() || dims_to_cut.size() == input_shape.rank()) {
        // It seems we need to cut all dimensions, completely (full reduction)
        reduction_steps.clear();
        result_shape_dims_cut = Shape();
    } else {
        // Preparing list of intermediate steps necessary to reduce
        // all requested dimensions (one by one)
        reduction_steps.reserve(dims_to_cut.size());
        std::vector<ShapeDim> result_dims = input_shape.dims();
        std::vector<ShapeDim> result_dims_preserved = input_shape.dims();
        for (auto dim = dims_to_cut.rbegin();
             dim != dims_to_cut.rend(); ++dim) {
            ReductionStep step{};
            step.dim_size = static_cast<size_t>(result_dims[*dim]);
            step.source_block = 1;
            for (auto next_dim = result_dims.begin() + *dim + 1;
                 next_dim != result_dims.end(); ++next_dim) {
                step.source_block *= *next_dim;
            }
            step.source_stride = step.source_block * step.dim_size;
            std::move(result_dims.begin() + *dim + 1, result_dims.end(),
                      result_dims.begin() + *dim);
            result_dims.resize(result_dims.size() - 1);
            step.result_size = 1;
            for (auto d: result_dims) { step.result_size *= d; }
            reduction_steps.push_back(step);
            result_dims_preserved[*dim] = 1;
        }
        result_shape_dims_cut = Shape(result_dims);
        result_shape_dims_kept = Shape(result_dims_preserved);
    }
}


Reduction::Reduction(const NodeRef &input,
                     std::vector<ShapeDim> reduce_axis,
                     const bool keep_dims)
    :_result_shape_dims_cut{},
     _result_shape_dims_kept{},
     _result_dtype{input->dtype()},
     _dims_to_cut{input->shape().normalize_dims(reduce_axis)},
     _keep_dims{keep_dims},
     _to_be_like{nullptr}
{
    std::vector<ReductionStep> tmp_reduction_steps;
    estimate_steps_and_dimensions(
        input->shape(), _dims_to_cut,
        tmp_reduction_steps, _result_shape_dims_cut, _result_shape_dims_kept);
}

Reduction::Reduction(const NodeRef &input,
                     const NodeRef &to_be_like,
                     bool keep_dims)
    :_result_shape_dims_cut{},
     _result_shape_dims_kept{},
     _result_dtype{input->dtype()},
     _dims_to_cut{estimate_dims_to_cut(input->shape(), to_be_like->shape())},
     _keep_dims{keep_dims},
     _to_be_like{to_be_like}
{
    std::vector<ReductionStep> tmp_reduction_steps;
    if (input->shape() == to_be_like->shape() || _dims_to_cut.empty()) {
        // Handles cases when the input already looks like the target
        // or both shapes are already equivalent (like {5} and {1, 5})
        // so and we need nothing to do
        _result_shape_dims_cut = to_be_like->shape();
        _result_shape_dims_kept = to_be_like->shape();
    } else {
        estimate_steps_and_dimensions(
            input->shape(), _dims_to_cut,
            tmp_reduction_steps, _result_shape_dims_cut,
            _result_shape_dims_kept);
    }
}

std::vector<ShapeDim> Reduction::estimate_dims_to_cut(
        const Shape &input_shape, const Shape &to_be_like_shape) const {
    Shape input_shape_aligned, sample_shape_aligned, output_shape_aligned;
    Shape::align_for_broadcasting(input_shape, to_be_like_shape,
                                  input_shape_aligned, sample_shape_aligned,
                                  output_shape_aligned);
    std::vector<ShapeDim> dims_to_cut;
    for (ShapeDim i = 0; i < input_shape_aligned.rank(); ++i) {
        if (input_shape_aligned.dim(i) != sample_shape_aligned.dim(i)
                && (sample_shape_aligned.dim(i) == 1
                    || sample_shape_aligned.dim(i) == UnknownDim)) {
            dims_to_cut.push_back(i);
        }
    }
    return dims_to_cut;
}

cl::Program load_reduction_program(cl::CommandQueue &queue) {
    auto context = get_context_from_queue(queue);
    return CodeCache::get_default().get_program(
        context,
        queue,
        "reductions",
        cl_sources_of_random_generators,
        "");
}

MultiArrayRef Reduction::partial_reduction(
        const MultiArrayRef &value,
        const std::vector<ReductionStep> &reduction_steps,
        const Shape &result_shape_dims_cut,
        const Shape &result_shape_dims_kept) const {
    if (reduction_steps.empty()) {
        // If we pass empty array as list of dimensions to reduce,
        // then we need to do nothing.
        return value;
    }
    auto pool = value->buffer_unsafe()->pool();
    auto queue = pool->cl_queue();
    auto program = load_reduction_program(queue);
    using KernelType = cl::KernelFunctor<
        const cl::Buffer&, cl_ulong, const cl::Buffer&,
        cl_ulong, cl_ulong, cl_ulong, cl_ulong>;
    KernelType kernel(program, get_kernel_name(true));
    CLBufferRef result_buffer;
    CLBufferRef source_buffer = value->buffer_unsafe();
    std::vector<cl::Event> wait_for_events(1);
    for (auto &step: reduction_steps) {
        wait_for_events[0] = source_buffer->completion_event();
        result_buffer = pool->reserve_buffer(
            step.result_size * array_type_size(_result_dtype));
        result_buffer->add_dependencies({source_buffer});
        const auto work_items = make_divisible_by(
            WORK_GROUP_SIZE, step.result_size);

        cl::Event reduction_is_done = kernel(
            cl::EnqueueArgs(queue,
                            wait_for_events,
                            cl::NDRange(work_items),
                            cl::NDRange(WORK_GROUP_SIZE)),
            source_buffer->cl_buffer_unsafe(),
            static_cast<cl_ulong>(value->buffer_offset()),
            result_buffer->cl_buffer_unsafe(),
            static_cast<cl_ulong>(step.result_size),
            static_cast<cl_ulong>(step.source_stride),
            static_cast<cl_ulong>(step.source_block),
            static_cast<cl_ulong>(step.dim_size));
        result_buffer->set_completion_event(reduction_is_done);
        source_buffer = result_buffer;
    }
    auto result = MultiArray::from_buffer(
        result_buffer,
        (_keep_dims ? result_shape_dims_kept : result_shape_dims_cut),
        _result_dtype);
    return result;
}

// Forward transformation for cases with only one input
MultiArrayRef Reduction::forward(const MultiArrayRef &value) const {
    std::vector<ReductionStep> reduction_steps;
    Shape result_shape_dims_cut;
    Shape result_shape_dims_kept;
    estimate_steps_and_dimensions(
        value->shape(), _dims_to_cut,
        reduction_steps, result_shape_dims_cut, result_shape_dims_kept);
    if (result_shape_dims_cut.rank() == 0) {
        return full_reduction(value);
    } else {
        return partial_reduction(
            value, reduction_steps,
            result_shape_dims_cut, result_shape_dims_kept);
    }
}

// Forward transformation for cases with two inputs (to be like)
MultiArrayRef Reduction::forward(const MultiArrayRef &value,
                                 const MultiArrayRef &to_be_like_value) const {
    if (value->shape() == to_be_like_value->shape()) {
        // Handles cases when the value already looks like the target
        // and we need nothing to do
        return value;
    }
    auto dims_to_cut = estimate_dims_to_cut(
        value->shape(), to_be_like_value->shape());
    if (dims_to_cut.empty()) {
        // Dimensions are already equivalent, like {5} and {1, 5}
        // so we can do simple reshaping
        return value->reshape(to_be_like_value->shape().dims());
    }
    std::vector<ReductionStep> reduction_steps;
    Shape result_shape_dims_cut;
    Shape result_shape_dims_kept;
    estimate_steps_and_dimensions(
        value->shape(), dims_to_cut,
        reduction_steps, result_shape_dims_cut, result_shape_dims_kept);
    if (result_shape_dims_cut.rank() == 0) {
        return full_reduction(value);
    } else {
        return partial_reduction(
            value, reduction_steps,
            result_shape_dims_cut, result_shape_dims_kept);
    }
}

MultiArrayRef Reduction::full_reduction(const MultiArrayRef &value) const {
    auto pool = value->buffer_unsafe()->pool();
    auto queue = pool->cl_queue();
    auto context = get_context_from_queue(queue);
    auto device = get_device_from_queue(queue);
    cl_uint max_compute_units;
    device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &max_compute_units);
    // This two-step reduction algorithm requires the work group to be
    // the power of two in size.
    cl_uint optimal_num_work_groups = (
        1U << static_cast<cl_uint>(std::log2(max_compute_units)));
    if (optimal_num_work_groups < 2) {
        throw std::runtime_error(
            "Incompatible OpenCL device that cannot "
            "be used for 2-step reduction.");
    }
    auto program = load_reduction_program(queue);
    auto kernel_name = get_kernel_name(false);
    auto kernel = cl::Kernel(program, kernel_name.c_str());
    const std::size_t step1_work_items = (
        WORK_GROUP_SIZE * optimal_num_work_groups);
    const std::size_t step1_scratchpad_size = (
        array_type_size(_result_dtype) * WORK_GROUP_SIZE);
    auto wait_for_events = make_event_list(
        {value->buffer_unsafe()->completion_event()});
    auto step1_buffer = pool->reserve_buffer(
        array_type_size(_result_dtype) * optimal_num_work_groups);
    step1_buffer->set_label(__func__, __LINE__);
    step1_buffer->add_dependencies({value->buffer_unsafe()});
    kernel.setArg(0, value->cl_buffer_unsafe());
    kernel.setArg(1, static_cast<cl_ulong>(value->buffer_offset()));
    kernel.setArg(2, step1_buffer->cl_buffer_unsafe());
    kernel.setArg(3, static_cast<cl_ulong>(step1_scratchpad_size), nullptr);
    kernel.setArg(4, static_cast<cl_ulong>(value->shape().size()));
    kernel.setArg(5, static_cast<cl_int>(CL_TRUE));
    cl::Event step_is_done;
    queue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange(step1_work_items),
        cl::NDRange(WORK_GROUP_SIZE),
        &wait_for_events,
        &step_is_done);
    // Full reduction always results in a scalar
    auto result = pool->make_array(Shape(), _result_dtype);
    result->add_dependencies({step1_buffer});
    kernel.setArg(0, step1_buffer->cl_buffer_unsafe());
    kernel.setArg(1, static_cast<cl_ulong>(0));
    kernel.setArg(2, result->cl_buffer_unsafe());
    kernel.setArg(
        3,
        array_type_size(_result_dtype) * (optimal_num_work_groups / 2),
        nullptr);
    kernel.setArg(4, static_cast<cl_ulong>(optimal_num_work_groups));
    kernel.setArg(5, static_cast<cl_int>(CL_FALSE));
    wait_for_events[0] = std::move(step_is_done);
    cl::Event step2_is_done;
    queue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange(optimal_num_work_groups / 2),
        cl::NDRange(optimal_num_work_groups / 2),
        &wait_for_events,
        &step2_is_done);
    result->set_completion_event(step2_is_done);
    return result;
}

const NodeRef Reduction::apply_chain_rule(const NodeRef &wrt_input,
                                          const NodeRef &d_target_wrt_this,
                                          const NodeRefList &all_inputs) const {

    if (all_inputs[0] == wrt_input) {
        return F<Multiply>(
            ReshapeLike::make(d_target_wrt_this, wrt_input, _dims_to_cut),
            partial_derivative(wrt_input));
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}

std::string Reduction::rh_name() const {
    return fmt::format(", {})", Shape::dims_to_string(_dims_to_cut));
}

const std::string Reduction::get_kernel_name(bool is_partial_reduction) const {
    auto op_name = kernel_op_name();
    if (is_partial_reduction) {
        return fmt::format("reduce_{}_{}", op_name,
                           array_type_name(_result_dtype));
    } else {
        return fmt::format("step_of_full_reduce_{}_{}", op_name,
                           array_type_name(_result_dtype));
    }
}

std::string Reduction::name() const {
    return fmt::format("reduce_{}_to_be_like", kernel_op_name());
}

std::string Reduction::repr_extra() const {
    return fmt::format("along_axis: {}",
                       Shape::dims_to_string(_dims_to_cut, false));
}

const NodeRef ReduceSum::partial_derivative(const NodeRef &input) const {
    return Constant::ones_like(input);
}

const NodeRef ReduceMean::partial_derivative(const NodeRef &input) const {
    // TODO: Replace with F<Repeat>
    return F<Multiply>(
        Constant::ones_like(input),
        F<Recip>(
            FU<ProductOfDims>(input, _dims_to_cut, dtype())));
}

const NodeRef softmax(const NodeRef &node, const ShapeDim axis) {
    auto mean = FU<ReduceMean>(node, std::vector<ShapeDim>({axis}), true);
    auto exp_node = FU<Exp>(node - mean);
    auto sum_node = FU<ReduceSum>(exp_node, std::vector<ShapeDim>({axis}), true);
    return exp_node / sum_node;
}

} // namespace
