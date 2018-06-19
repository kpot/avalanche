#include <fmt/format.h>

#include "avalanche/math_ops/ElemWiseBinaryOp.h"
#include "avalanche/opencl_utils.h"
#include "avalanche/CodeCache.h"

namespace avalanche {

constexpr const std::size_t WORK_GROUP_SIZE = 64;

std::string elemwise_binary_kernel_name(
    const std::string &operation_name,
    ArrayType left_dtype,
    ArrayType right_dtype,
    ArrayType output_dtype) {
    return fmt::format(
        "update_{operation_name}_{left_dtype}_{right_dtype}_{output_dtype}",
        fmt::arg("operation_name", operation_name),
        fmt::arg("left_dtype", cl_type_name_of_array(left_dtype)),
        fmt::arg("right_dtype", cl_type_name_of_array(right_dtype)),
        fmt::arg("output_dtype", cl_type_name_of_array(output_dtype))
    );
}

std::string elemwise_binary_kernel_code(
    const std::string &operation_name,
    ArrayType left_dtype,
    ArrayType right_dtype,
    ArrayType output_dtype,
    const std::string &operation_code,
    int work_group_size) {
    constexpr const char *kernel_template = R"clkernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size({work_group_size}, 1, 1)))
void {kernel_name}(
         __global {left_dtype} *left_source,
         const ulong left_offset,
         __global {right_dtype} *right_source,
         const ulong right_offset,
         __global {output_dtype} *output,
         const ulong data_size) {{
    if (get_global_id(0) < data_size) {{
        {left_dtype} a = left_source[left_offset + get_global_id(0)];
        {right_dtype} b = right_source[right_offset + get_global_id(0)];
        output[get_global_id(0)] = ({output_dtype})({operation_code});
    }}
}}
    )clkernel";

    return fmt::format(
        kernel_template,
        fmt::arg("kernel_name",
                 elemwise_binary_kernel_name(operation_name, left_dtype, right_dtype, output_dtype)),
        fmt::arg("operation_code", operation_code),
        fmt::arg("work_group_size", work_group_size),
        fmt::arg("left_dtype", cl_type_name_of_array(left_dtype)),
        fmt::arg("right_dtype", cl_type_name_of_array(right_dtype)),
        fmt::arg("output_dtype", cl_type_name_of_array(output_dtype)));
}

ElemWiseBinaryOp::ElemWiseBinaryOp(const NodeRef &left, const NodeRef &right,
                                   const std::string &operation_name,
                                   const std::string &operation_cl_code,
                                   ArrayType output_dtype)
:_result_shape{infer_elemwise_shape(left->shape(), right->shape())},
 _result_dtype{output_dtype},
 _operation_name{operation_name},
 _kernel_name{elemwise_binary_kernel_name(operation_name, left->dtype(), right->dtype(), output_dtype)},
 _kernel_source{elemwise_binary_kernel_code(operation_name, left->dtype(), right->dtype(), output_dtype, operation_cl_code, WORK_GROUP_SIZE)}
{
}

Shape ElemWiseBinaryOp::infer_elemwise_shape(const Shape &shape1,
                                             const Shape &shape2) {
    if (shape1.rank() != shape2.rank()) {
        throw std::invalid_argument(
            fmt::format(
                "Cannot infer shape of element-wise operation's result "
                "because shape {} and {} have different ranks",
                shape1.to_string(), shape2.to_string()));
    }
    std::vector<ShapeDim> result_dims(shape1.rank());
    for (ShapeDim dim_id = 0; dim_id < shape1.rank(); ++dim_id) {
        ShapeDim d1 = shape1.dim(dim_id), d2 = shape2.dim(dim_id);
        if (d1 == UnknownDim) {
            if (d2 == UnknownDim) {
                result_dims[dim_id] = UnknownDim;
            } else {
                result_dims[dim_id] = d2;
            }
        } else {
            if (d2 == UnknownDim) {
                result_dims[dim_id] = d1;
            } else {
                if (d1 != d2) {
                    throw std::invalid_argument(
                        fmt::format("Element-wise operation is impossible "
                                    "for arrays of shapes {} and {}",
                                    shape1.to_string(), shape2.to_string()));
                } else {
                    result_dims[dim_id] = d1;
                }
            }
        }
    }
    return Shape(result_dims);
}

MultiArrayRef ElemWiseBinaryOp::forward(const MultiArrayRef &v1,
                                        const MultiArrayRef &v2) const {
    if (v1->shape() != v2->shape()) {
        throw std::invalid_argument(
            fmt::format("Cannot perform element-wise operation {} on two "
                        "incompatible arrays with different shapes: {} and {}",
                        _operation_name, v1->shape().to_string(),
                        v2->shape().to_string()));
    }
    // At this point we assume that every aspect of both arrays have already
    // been checked by the constructor, so there's nothing to worry about
    // Also at this point both arrays v1 and v2 may still be calculating
    auto pool = v1->buffer_unsafe()->pool();
    auto queue = pool->cl_queue();
    auto data_are_ready = make_event_list(
        {v1->buffer_unsafe()->completion_event(),
         v2->buffer_unsafe()->completion_event()});
    auto result = pool->make_array(v1->shape(), _result_dtype);
    result->set_label(_operation_name + " at " + __func__, __LINE__);
    // To keep the buffers alive until the computation is done we add them
    // as dependencies.
    result->add_dependencies({v1, v2});
    // The main job
    auto program = CodeCache::get_default().get_program(
        pool->cl_context(), queue,
        _kernel_name, _kernel_source, "");
    using Buf = const cl::Buffer&;
    cl::KernelFunctor<Buf, cl_ulong, Buf, cl_ulong, Buf, cl_ulong>
        kernel_functor(program, _kernel_name);
    const auto result_size = v1->shape().size();
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
        static_cast<cl_ulong>(result_size));
    // Let us know when everything is done by marking the resulting array
    // as "complete" (ready)
    result->set_completion_event(result_event);
    return result;
}

} // namespace
