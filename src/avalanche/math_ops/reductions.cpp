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

namespace avalanche {

constexpr char reducing_kernels_source[] = {
#include "avalanche/kernels/reductions.hex"
};

constexpr std::size_t work_group_size = 64;

template <typename Container>
inline void leave_unique_only(Container &container) {
    std::sort(container.begin(), container.end());
    auto last = std::unique(container.begin(), container.end());
    container.erase(last, container.end());
}

Reduction::Reduction(const NodeRef &input)
    : _result_shape_dims_cut{},
      _result_shape_dims_kept{},
      _result_dtype{input->dtype()},
      _kernel_name{},
      _reduction_steps{},
      _dims_to_cut{},
      _keep_dims{false}
{
}

Reduction::Reduction(const NodeRef &input,
                     std::vector<ShapeDim> reduce_axis,
                     const bool keep_dims)
    :_result_shape_dims_cut{},
     _result_shape_dims_kept{},
     _result_dtype{input->dtype()},
     _kernel_name{},
     _reduction_steps{},
     _dims_to_cut{reduce_axis},
     _keep_dims{keep_dims}
{
    const auto input_shape = input->shape();
    // Normalizing "negative" (relative) reduce_axis into absolute ones
    auto input_rank = input_shape.rank();
    for (auto &dim: reduce_axis) {
        if (dim < 0) {
            dim = static_cast<ShapeDim>(input_rank) + dim;
        }
        if (dim >= input_rank) {
            throw std::invalid_argument(
                "One of the reduce_axis doesn't exist");
        }
    }
    leave_unique_only(reduce_axis);
    if (reduce_axis.empty() || reduce_axis.size() == input->shape().rank()) {
        // It seems we need to cut all dimensions, completely (full reduction)
        _reduction_steps.clear();
        _dims_to_cut.clear();
        _result_shape_dims_cut = Shape();
    } else {
        // Preparing list of intermediate steps necessary to reduce
        // all requested dimensions (one by one)
        _reduction_steps.reserve(reduce_axis.size());
        auto result_dims = input_shape.dims();
        auto result_dims_preserved = input_shape.dims();
        for (auto dim = reduce_axis.rbegin();
             dim != reduce_axis.rend(); ++dim) {
            ReductionStep step;
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
            _reduction_steps.push_back(step);
            result_dims_preserved[*dim] = 1;
        }
        _result_shape_dims_cut = Shape(result_dims);
        _result_shape_dims_kept = Shape(result_dims_preserved);
    }
}

cl::Program load_reduction_program(cl::CommandQueue &queue) {
    auto context = get_context_from_queue(queue);
    return CodeCache::get_default().get_program(
        context,
        queue,
        "reductions",
        reducing_kernels_source,
        "");
}

MultiArrayRef Reduction::partial_reduction(const MultiArrayRef &value) const {
    if (_reduction_steps.empty()) {
        // If we pass empty array as list of dimensions to reduce,
        // then we need to do nothing.
        return value;
    }
    auto pool = value->buffer_unsafe()->pool();
    auto queue = pool->cl_queue();
    auto program = load_reduction_program(queue);
    using KernelType = cl::KernelFunctor<
        const cl::Buffer&, const cl::Buffer&,
        cl_ulong, cl_ulong, cl_ulong, cl_ulong>;
    KernelType kernel(program, cached_kernel_name());
    CLBufferRef result_buffer;
    CLBufferRef source_buffer = value->buffer_unsafe();
    std::vector<cl::Event> wait_for_events(1);
    for (auto &step: _reduction_steps) {
        wait_for_events[0] = source_buffer->completion_event();
        result_buffer = pool->reserve_buffer(
            step.result_size * array_type_size(_result_dtype));
        result_buffer->add_dependencies({source_buffer});
        const auto work_items = make_divisible_by(
            work_group_size, step.result_size);

        cl::Event reduction_is_done = kernel(
            cl::EnqueueArgs(queue,
                            wait_for_events,
                            cl::NDRange(work_items),
                            cl::NDRange(work_group_size)),
            source_buffer->cl_buffer_unsafe(),
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
        (_keep_dims ? _result_shape_dims_kept : _result_shape_dims_cut),
        _result_dtype);
    return result;
}

MultiArrayRef Reduction::forward(const MultiArrayRef &value) const {
    if (_result_shape_dims_cut.rank() == 0) {
        return full_reduction(value);
    } else {
        return partial_reduction(value);
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
    auto kernel = cl::Kernel(program, cached_kernel_name().c_str());
    const std::size_t step1_work_items = (
        work_group_size * optimal_num_work_groups);
    const std::size_t step1_scratchpad_size = (
        array_type_size(_result_dtype) * work_group_size);
    auto wait_for_events = make_event_list(
        {value->buffer_unsafe()->completion_event()});
    auto step1_buffer = pool->reserve_buffer(
        array_type_size(_result_dtype) * optimal_num_work_groups);
    step1_buffer->set_label(__func__, __LINE__);
    step1_buffer->add_dependencies({value->buffer_unsafe()});
    kernel.setArg(0, value->cl_buffer_unsafe());
    kernel.setArg(1, step1_buffer->cl_buffer_unsafe());
    kernel.setArg(2, step1_scratchpad_size, nullptr);
    kernel.setArg(3, static_cast<cl_ulong>(value->shape().size()));
    kernel.setArg(4, static_cast<cl_int>(CL_TRUE));
    cl::Event step_is_done;
    queue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange(step1_work_items),
        cl::NDRange(work_group_size),
        &wait_for_events,
        &step_is_done);
    auto result = pool->make_array(_result_shape_dims_cut, _result_dtype);
    result->add_dependencies({step1_buffer});
    kernel.setArg(0, step1_buffer->cl_buffer_unsafe());
    kernel.setArg(1, result->cl_buffer_unsafe());
    kernel.setArg(
        2,
        array_type_size(_result_dtype) * (optimal_num_work_groups / 2),
        nullptr);
    kernel.setArg(3, static_cast<cl_ulong>(optimal_num_work_groups));
    kernel.setArg(4, static_cast<cl_int>(CL_FALSE));
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
        // FIXME: Cleanup
        std::cout << "Reduction: " << d_target_wrt_this->to_string() << std::endl;
        return F<Multiply>(
            FU<Reshape>(d_target_wrt_this, _result_shape_dims_kept),
            partial_derivative(wrt_input));
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}

std::string Reduction::rh_name() const {
    return fmt::format(", {})", Shape::dims_to_string(_dims_to_cut));
}


const NodeRef ReduceSum::partial_derivative(const NodeRef &input) const {
    return Constant::ones_like(input);
}

const NodeRef ReduceMean::partial_derivative(const NodeRef &input) const {
    double dim_prod = (
        _reduction_steps.empty() ? input->shape().size() : 1.0);
    for (auto step: _reduction_steps) {
        dim_prod *= static_cast<float>(step.dim_size);
    }
    return FU<Scale>(Constant::ones_like(input), 1.0 / dim_prod);
}

const NodeRef softmax(const NodeRef &node, const ShapeDim axis) {
    auto mean = FU<ReduceMean>(node, std::vector<ShapeDim>({axis}), true);
    auto exp_node = FU<Exp>(node - mean);
    auto sum_node = FU<ReduceSum>(exp_node, std::vector<ShapeDim>({axis}), true);
    return exp_node / sum_node;
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

} // namespace
