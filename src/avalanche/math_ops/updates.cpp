#include <fmt/format.h>

#include "avalanche/math_ops/updates.h"
#include "avalanche/CodeCache.h"
#include "avalanche/opencl_utils.h"


namespace avalanche {

constexpr const std::size_t WORK_GROUP_SIZE = 64;

std::string updating_kernel_name(
        const std::string &operation_name,
        ArrayType stype,
        ArrayType dtype) {
    return fmt::format(
        "update_{operation_name}_{source_type}_{destination_type}",
        fmt::arg("operation_name", operation_name),
        fmt::arg("source_type", cl_type_name_of_array(stype)),
        fmt::arg("destination_type", cl_type_name_of_array(dtype))
    );
}

std::string updating_kernel_source(
    const std::string &operation_name,
    ArrayType stype,
    ArrayType dtype,
    const std::string &operation_code,
    int work_group_size) {
    constexpr const char *kernel_template = R"clkernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size({work_group_size}, 1, 1)))
void {kernel_name}(
         __global {dtype} *target,
         const ulong target_offset,
         __global {stype} *update,
         const ulong update_offset,
         const ulong target_size) {{
    if (get_global_id(0) < target_size) {{
        target[target_offset + get_global_id(0)] {operation_code} update[update_offset + get_global_id(0)];
    }}
}}
    )clkernel";

    return fmt::format(
        kernel_template,
        fmt::arg("kernel_name",
                 updating_kernel_name(operation_name, stype, dtype)),
        fmt::arg("operation_code", operation_code),
        fmt::arg("work_group_size", work_group_size),
        fmt::arg("dtype", cl_type_name_of_array(dtype)),
        fmt::arg("stype", cl_type_name_of_array(stype)));
}

BaseUpdateOp::BaseUpdateOp(const NodeRef &variable, const NodeRef &update,
                   const std::string &operation_name,
                   const std::string &operation_cl_code)
:_result_shape{variable->shape()},
 _result_dtype{variable->dtype()},
 _operation_name{operation_name},
 _kernel_name{
    updating_kernel_name(operation_name, update->dtype(), variable->dtype())},
 _kernel_source{
    updating_kernel_source(operation_name, update->dtype(),
                           variable->dtype(), operation_cl_code, WORK_GROUP_SIZE)}
{
    if (variable->shape() != update->shape()) {
        throw std::invalid_argument(
            fmt::format("The shapes of the left ({}) and right ({}) arguments "
                        "must be identical",
                        variable->shape().to_string(),
                        update->shape().to_string()));
    }
}

MultiArrayRef
BaseUpdateOp::forward(const MultiArrayRef &v1, const MultiArrayRef &v2) const {
    auto pool = v1->buffer_unsafe()->pool();
    auto queue = pool->cl_queue();
    auto program = CodeCache::get_default().get_program(
        pool->cl_context(), queue,
        _kernel_name, _kernel_source, "");
    using Buf = const cl::Buffer&;
    cl::KernelFunctor<Buf, cl_ulong, Buf, cl_ulong, cl_ulong>
        kernel_functor(program, _kernel_name);

    const auto result_size = v1->shape().size();
    const auto work_items = make_divisible_by(WORK_GROUP_SIZE, result_size);

    auto data_are_ready = make_event_list(
        {v1->buffer_unsafe()->completion_event(),
         v2->buffer_unsafe()->completion_event()});
    v1->add_dependencies({v2});
    cl::Event result_event = kernel_functor(
        cl::EnqueueArgs(queue,
                        data_are_ready,
                        cl::NDRange(work_items),
                        cl::NDRange(WORK_GROUP_SIZE)),
        v1->cl_buffer_unsafe(),
        static_cast<cl_ulong>(v1->buffer_offset()),
        v2->cl_buffer_unsafe(),
        static_cast<cl_ulong>(v2->buffer_offset()),
        static_cast<cl_ulong>(result_size));
    v1->set_completion_event(result_event);
    return v1;
}


} // namespace
