#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define WORK_GROUP_SIZE 64

#define summarize(accumulator, x)  accumulator += x
#define product(accumulator, x)  accumulator *= x
#define choose_max(accumulator, x) accumulator = max(x, accumulator)
#define choose_min(accumulator, x) accumulator = min(x, accumulator)
#define choose_fmax(accumulator, x) accumulator = fmax(x, accumulator)
#define choose_fmin(accumulator, x) accumulator = fmin(x, accumulator)
#define choose_fmax64(accumulator, x) accumulator = (x > accumulator ? x : accumulator)
#define choose_fmin64(accumulator, x) accumulator = (x < accumulator ? x : accumulator)
#define leave_as_is(x, n, active) x
#define calc_mean(x, n, active) (active == 1 ? (x / n) : x)

/* Kernel template that reduces just one dimension. */
#define partial_reduction_kernel_template(OpName, DType, Type, Op, ResultOp, Initial) \
__kernel __attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1))) \
void reduce_##OpName##_##DType ( \
         __global Type *source, \
         const ulong source_offset, \
         __global Type *output, \
         const ulong result_size, \
         const ulong source_stride, \
         const ulong source_block, \
         const ulong dim_size) { \
    size_t output_index = get_global_id(0); \
    ulong source_start_index = (output_index / source_block) * source_stride + (output_index % source_block); \
    Type accumulator = (Type) Initial; \
    for (ulong i = 0; i < dim_size; ++i) { \
        Op(accumulator, source[source_offset + source_start_index]); \
        source_start_index += source_block; \
    } \
    if (output_index < result_size) { output[output_index] = ResultOp(accumulator, dim_size, 1); } \
}

#ifdef HALF_MAX
#define set_of_partial_reduction_kernels(OpName, Op, ResultOp, Initial) \
    partial_reduction_kernel_template(OpName, float16, half,   Op, ResultOp, Initial) \
    partial_reduction_kernel_template(OpName, float32, float,  Op, ResultOp, Initial) \
    partial_reduction_kernel_template(OpName, float64, double, Op, ResultOp, Initial) \
    partial_reduction_kernel_template(OpName, int8,    char,   Op, ResultOp, Initial) \
    partial_reduction_kernel_template(OpName, int16,   short,  Op, ResultOp, Initial) \
    partial_reduction_kernel_template(OpName, int32,   int,    Op, ResultOp, Initial) \
    partial_reduction_kernel_template(OpName, int64,   long,   Op, ResultOp, Initial)
#else
#define set_of_partial_reduction_kernels(OpName, Op, ResultOp, Initial) \
    partial_reduction_kernel_template(OpName, float32, float,  Op, ResultOp, Initial) \
    partial_reduction_kernel_template(OpName, float64, double, Op, ResultOp, Initial) \
    partial_reduction_kernel_template(OpName, int8,    char,   Op, ResultOp, Initial) \
    partial_reduction_kernel_template(OpName, int16,   short,  Op, ResultOp, Initial) \
    partial_reduction_kernel_template(OpName, int32,   int,    Op, ResultOp, Initial) \
    partial_reduction_kernel_template(OpName, int64,   long,   Op, ResultOp, Initial)
#endif

/* Two-stage reduction kernel. Merges all, ignoring dimensions.
   IMPORTANT: This kernel requires work group size to the power of 2.
   Here's a very rough reminder of how it works:

                    GRID_SIZE (2 x global_size)
     -------------------------+-------------+-----+-------------|
     assigned to workgroup 1  | workgroup 2 | ... | workgroup N |
     -------------------------+-------------+-----+-------------|
     block_size | block_size  |
     (WG size)  | (WG size)   |
     0 1 2 3 4    5 6 7 8 9 10|
     | | | | |   / / / / / /  |
     | | | | |  / / / / / /   |
     | | | | | / / / / / /    |
     + + + + +----------      |
     | | | | |                |
     Workgroup scratchpad     |
*/
#define full_reduction_template(OpName, DType, Type, Op, ResultOp, Initial) \
__kernel void step_of_full_reduce_##OpName##_##DType( \
    __global Type *source, \
    const ulong source_offset, \
    __global Type *output, \
    __local Type *scratch, \
    const ulong length, \
    const int is_first_step) \
{ \
    const ulong local_id = get_local_id(0); \
    ulong i = get_group_id(0) * (2 * get_local_size(0)) + get_local_id(0); \
    ulong grid_size = 2 * get_global_size(0); \
    scratch[local_id] = (Type) Initial; \
    while (i < length) { \
        Op(scratch[local_id], source[source_offset + i]); \
        if (i + get_local_size(0) < length) { \
            Op(scratch[local_id], source[source_offset + i + get_local_size(0)]); \
        } \
        i += grid_size; \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for(ulong offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) { \
        if (local_id < offset) { \
            Op(scratch[local_id], scratch[local_id + offset]); \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (local_id == 0) { \
        output[get_group_id(0)] = ResultOp(scratch[0], length, is_first_step); \
    } \
}

#ifdef HALF_MAX
#define set_of_full_reduction_kernels(OpName, Op, ResultOp, Initial) \
    full_reduction_template(OpName, float16, half,   Op, ResultOp, Initial) \
    full_reduction_template(OpName, float32, float,  Op, ResultOp, Initial) \
    full_reduction_template(OpName, float64, double, Op, ResultOp, Initial) \
    full_reduction_template(OpName, int8,    char,   Op, ResultOp, Initial) \
    full_reduction_template(OpName, int16,   short,  Op, ResultOp, Initial) \
    full_reduction_template(OpName, int32,   int,    Op, ResultOp, Initial) \
    full_reduction_template(OpName, int64,   long,   Op, ResultOp, Initial)
#else
#define set_of_full_reduction_kernels(OpName, Op, ResultOp, Initial) \
    full_reduction_template(OpName, float32, float,  Op, ResultOp, Initial) \
    full_reduction_template(OpName, float64, double, Op, ResultOp, Initial) \
    full_reduction_template(OpName, int8,    char,   Op, ResultOp, Initial) \
    full_reduction_template(OpName, int16,   short,  Op, ResultOp, Initial) \
    full_reduction_template(OpName, int32,   int,    Op, ResultOp, Initial) \
    full_reduction_template(OpName, int64,   long,   Op, ResultOp, Initial)
#endif

#ifdef HALF_MAX
#define set_of_maximizing_kernels(template_name) \
    template_name(max, float16, half,   choose_fmax, leave_as_is, -INFINITY) \
    template_name(max, float32, float,  choose_fmax, leave_as_is, -INFINITY) \
    template_name(max, float64, double, choose_fmax64, leave_as_is, -INFINITY) \
    template_name(max, int8,    char,   choose_max, leave_as_is, CHAR_MIN) \
    template_name(max, int16,   short,  choose_max, leave_as_is, SHRT_MIN) \
    template_name(max, int32,   int,    choose_max, leave_as_is, INT_MIN) \
    template_name(max, int64,   long,   choose_max, leave_as_is, LONG_MIN)
#else
#define set_of_maximizing_kernels(template_name) \
    template_name(max, float32, float,  choose_fmax, leave_as_is, -INFINITY) \
    template_name(max, float64, double, choose_fmax64, leave_as_is, -INFINITY) \
    template_name(max, int8,    char,   choose_max, leave_as_is, CHAR_MIN) \
    template_name(max, int16,   short,  choose_max, leave_as_is, SHRT_MIN) \
    template_name(max, int32,   int,    choose_max, leave_as_is, INT_MIN) \
    template_name(max, int64,   long,   choose_max, leave_as_is, LONG_MIN)
#endif

#ifdef HALF_MAX
#define set_of_minimizing_kernels(template_name) \
    template_name(min, float16, float,  choose_fmin, leave_as_is, INFINITY) \
    template_name(min, float32, float,  choose_fmin, leave_as_is, INFINITY) \
    template_name(min, float64, double, choose_fmin64, leave_as_is, INFINITY) \
    template_name(min, int8,    char,   choose_min, leave_as_is, CHAR_MAX) \
    template_name(min, int16,   short,  choose_min, leave_as_is, SHRT_MAX) \
    template_name(min, int32,   int,    choose_min, leave_as_is, INT_MAX) \
    template_name(min, int64,   long,   choose_min, leave_as_is, LONG_MAX)
#else
#define set_of_minimizing_kernels(template_name) \
    template_name(min, float32, float,  choose_fmin, leave_as_is, INFINITY) \
    template_name(min, float64, double, choose_fmin64, leave_as_is, INFINITY) \
    template_name(min, int8,    char,   choose_min, leave_as_is, CHAR_MAX) \
    template_name(min, int16,   short,  choose_min, leave_as_is, SHRT_MAX) \
    template_name(min, int32,   int,    choose_min, leave_as_is, INT_MAX) \
    template_name(min, int64,   long,   choose_min, leave_as_is, LONG_MAX)
#endif

/* ========================================================================= */

/* Partial sum reduction */
set_of_partial_reduction_kernels(sum, summarize, leave_as_is, 0)

/* Partial prod reduction */
set_of_partial_reduction_kernels(prod, product, leave_as_is, 1)

/* Partial mean reduction */
set_of_partial_reduction_kernels(mean, summarize, calc_mean, 0)

/* Partial max reduction */
set_of_maximizing_kernels(partial_reduction_kernel_template)

/* Partial min reduction */
set_of_minimizing_kernels(partial_reduction_kernel_template)

/* Full sum reduction */
set_of_full_reduction_kernels(sum, summarize, leave_as_is, 0)

/* Full prod reduction */
set_of_full_reduction_kernels(prod, product, leave_as_is, 1)

/* Full mean reduction */
set_of_full_reduction_kernels(mean, summarize, calc_mean, 0)

/* Full minimizing reduction */
set_of_minimizing_kernels(full_reduction_template)

/* Full maximizing reduction */
set_of_maximizing_kernels(full_reduction_template)


