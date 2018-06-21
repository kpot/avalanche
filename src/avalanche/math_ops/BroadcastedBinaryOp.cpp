#include <clblast.h>
#include <iostream>

#include "fmt/format.h"

#include "avalanche/math_ops/BroadcastedBinaryOp.h"
#include "avalanche/opencl_utils.h"
#include "avalanche/CodeCache.h"
#include "avalanche/casting.h"

namespace avalanche {

constexpr const std::size_t WORK_GROUP_SIZE = 64;


std::string broadcasing_kernel_name(const std::string &operation_name,
                                    ArrayType left_dtype,
                                    ArrayType righ_dtype,
                                    ArrayType output_dtype) {
    return fmt::format(
        "broadcasted_{operation_name}_{left_type}_{right_type}_{output_type}",
        fmt::arg("operation_name", operation_name),
        fmt::arg("left_type", cl_type_name_of_array(left_dtype)),
        fmt::arg("right_type", cl_type_name_of_array(righ_dtype)),
        fmt::arg("output_type", cl_type_name_of_array(output_dtype)));
}

std::string generate_broadcasting_kernel(const std::string &operation_name,
                                         ArrayType left_dtype,
                                         ArrayType right_dtype,
                                         ArrayType output_dtype,
                                         const std::string &operation_code,
                                         int work_group_size) {
    constexpr const char *kernel_template = R"clkernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size({work_group_size}, 1, 1)))
void {kernel_name}(
         __global {left_dtype} *source1,
         const ulong source1_offset,
         __global {right_dtype} *source2,
         const ulong source2_offset,
         __global {output_type} *output,
         __global ulong *size_mask1,
         __global ulong *size_mask2,
         __global ulong *result_sizes,
         const ulong result_size,
         const int rank) {{
    ulong index_to_parse = get_global_id(0);
    ulong source1_index = 0, source2_index = 0;
    for (int j = 0; j < rank - 1; ++j) {{
        ulong dim_coord = index_to_parse / result_sizes[j];
        source1_index += dim_coord * size_mask1[j];
        source2_index += dim_coord * size_mask2[j];
        index_to_parse = index_to_parse % result_sizes[j];
    }}
    source1_index += size_mask1[rank - 1] * index_to_parse;
    source2_index += size_mask2[rank - 1] * index_to_parse;
    if (get_global_id(0) < result_size) {{
        {left_dtype} a = source1[source1_offset + source1_index];
        {right_dtype} b = source2[source2_offset + source2_index];
        output[get_global_id(0)] = ({output_type})({operation_code});
    }}
}}
    )clkernel";

    return fmt::format(
        kernel_template,
        fmt::arg("kernel_name",
                 broadcasing_kernel_name(
                     operation_name, left_dtype, right_dtype, output_dtype)),
        fmt::arg("operation_code", operation_code),
        fmt::arg("work_group_size", work_group_size),
        fmt::arg("left_dtype", cl_type_name_of_array(left_dtype)),
        fmt::arg("right_dtype", cl_type_name_of_array(right_dtype)),
        fmt::arg("output_type", cl_type_name_of_array(output_dtype)));
}


std::size_t broadcast_size_masks(const Shape &shape1, const Shape &shape2,
                                 std::vector<cl_ulong> &size_mask1,
                                 std::vector<cl_ulong> &size_mask2,
                                 std::vector<cl_ulong> &result_sub_sizes) {
    std::array<avalanche::Shape, 3> aligned_shapes;
    if (shape1.is_scalar() && shape2.is_scalar()) {
        aligned_shapes = std::array<avalanche::Shape, 3>(
            {avalanche::Shape({1}),
             avalanche::Shape({1}),
             avalanche::Shape({1})});
    } else {
        aligned_shapes = avalanche::Shape::align_for_broadcasting(
            shape1, shape2);
    }
    // a scalar value can be interpreted as a vector the shape (1,)
    auto aligned_rank = aligned_shapes[0].rank();
    size_mask1.resize(aligned_rank);
    size_mask2.resize(aligned_rank);
    result_sub_sizes.resize(aligned_rank);
    std::size_t cumprod1 = 1, cumprod2 = 1, cumprod_res = 1;
    for (long i = aligned_rank - 1; i > 0; --i) {
        cumprod1 *= aligned_shapes[0].dims()[i];
        cumprod2 *= aligned_shapes[1].dims()[i];
        cumprod_res *= aligned_shapes[2].dims()[i];
        size_mask1[i - 1] = aligned_shapes[0].dims()[i - 1] == 1 ? 0 : cumprod1;
        size_mask2[i - 1] = aligned_shapes[1].dims()[i - 1] == 1 ? 0 : cumprod2;
        result_sub_sizes[i - 1] = cumprod_res;
    }
    size_mask1[aligned_rank - 1] = aligned_shapes[0].dim(-1) == 1 ? 0 : 1;
    size_mask2[aligned_rank - 1] = aligned_shapes[1].dim(-1) == 1 ? 0 : 1;
    return aligned_shapes[2].size();
}

BroadcastedBinaryOp::BroadcastedBinaryOp(const NodeRef &left,
                                         const NodeRef &right,
                                         const std::string &operation_name,
                                         const std::string &operation_cl_code,
                                         ArrayType output_dtype)
    :_result_dtype{output_dtype},
     _operation_name{operation_name},
     _kernel_name{
         broadcasing_kernel_name(
             operation_name, left->dtype(), right->dtype(), output_dtype)},
     _kernel_source{
         generate_broadcasting_kernel(
             operation_name, left->dtype(), right->dtype(), output_dtype,
             operation_cl_code, WORK_GROUP_SIZE)}
{
    // Here we calculate only a preliminary result shape for debugging
    // purposes. The real shape can be only evaluated in runtime (`forward`)
    Shape tmp_left_shape_aligned, tmp_right_shape_aligned;
    Shape::align_for_broadcasting(
        left->shape(), right->shape(),
        tmp_left_shape_aligned, tmp_right_shape_aligned, _result_shape);
}


MultiArrayRef avalanche::BroadcastedBinaryOp::forward(
        const MultiArrayRef &v1,
        const MultiArrayRef &v2) const {
    // Calculating sizes, alignments, differences
    Shape result_shape, aligned_shape_left, aligned_shape_right;
    std::vector<cl_ulong> left_size_mask, right_size_mask, result_sub_sizes;

    Shape::align_for_broadcasting(
        v1->shape(), v2->shape(),
        aligned_shape_left, aligned_shape_right, result_shape);
    broadcast_size_masks(
        v1->shape(), v2->shape(),
        left_size_mask, right_size_mask, result_sub_sizes);

    // At this point we assume that every aspect of both arrays have already
    // been checked by the constructor, so there's nothing to worry about
    // Also at this point both arrays v1 and v2 may still be calculating,
    // so meanwhile we prepare and upload some other data
    auto pool = v1->buffer_unsafe()->pool();
    auto queue = pool->cl_queue();
    auto left_mask_buffer = pool->reserve_buffer_for_vector(left_size_mask);
    left_mask_buffer->set_label(__func__, __LINE__);
    auto right_mask_buffer = pool->reserve_buffer_for_vector(right_size_mask);
    right_mask_buffer->set_label(__func__, __LINE__);
    auto result_sizes_buffer = pool->reserve_buffer_for_vector(result_sub_sizes);
    result_sizes_buffer->set_label(__func__, __LINE__);



    auto masks_are_ready = make_event_list(
        {left_mask_buffer->write_from_vector(left_size_mask, 0),
         right_mask_buffer->write_from_vector(right_size_mask, 0),
         result_sizes_buffer->write_from_vector(result_sub_sizes, 0)});
    auto data_are_ready = make_event_list(
        {v1->buffer_unsafe()->completion_event(),
         v2->buffer_unsafe()->completion_event()});
    std::copy(masks_are_ready.begin(), masks_are_ready.end(),
              std::back_inserter(data_are_ready));
    auto result = pool->make_array(result_shape, _result_dtype);
    result->set_label(_operation_name + " at " + __func__, __LINE__);
    // To keep the buffers alive until the computation is done we add them
    // as dependencies.
    result->add_dependencies(
        {left_mask_buffer, right_mask_buffer, result_sizes_buffer});
    result->add_dependencies({v1, v2});
    // The main job
    auto program = CodeCache::get_default().get_program(
        pool->cl_context(), queue,
        _kernel_name, _kernel_source, "");
    using Buf = const cl::Buffer&;
    cl::KernelFunctor<Buf, cl_ulong, Buf, cl_ulong, Buf, Buf, Buf, Buf, cl_ulong, cl_int>
        kernel_functor(program, _kernel_name);
    const auto result_size = result_shape.size();
    const auto work_items = make_divisible_by(WORK_GROUP_SIZE, result_size);
    cl::Event result_event = kernel_functor(
        cl::EnqueueArgs(queue,
                        data_are_ready,
                        cl::NDRange(work_items),
                        cl::NDRange(WORK_GROUP_SIZE)),
        v1->cl_buffer_unsafe(),
        static_cast<cl_ulong>(v1->buffer_offset()),
        v2->cl_buffer_unsafe(),
        static_cast<cl_ulong>(v2->buffer_offset()),
        result->cl_buffer_unsafe(),
        left_mask_buffer->cl_buffer_unsafe(),
        right_mask_buffer->cl_buffer_unsafe(),
        result_sizes_buffer->cl_buffer_unsafe(),
        static_cast<cl_ulong>(result_size),
        static_cast<cl_int>(left_size_mask.size()));
    // Without waiting we cannot guarantee that OpenCL will have enough
    // time to copy all the data into the buffers before the vectors are gone
    cl::WaitForEvents(masks_are_ready);
    // Let us know when everything is done by marking the resulting array
    // as "complete" (ready)
    result->set_completion_event(result_event);
    return result;
}

} // namespace
