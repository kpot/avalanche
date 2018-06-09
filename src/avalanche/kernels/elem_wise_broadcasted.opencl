#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define WORK_GROUP_SIZE 64

#define op_plus(a, b) a + b
#define op_minus(a, b) a - b
#define op_multiply(a, b) a * b
#define op_divide(a, b) a / b
#define op_divide_native(a, b) native_divide(a, b)

#define broadcasted_op(OpName, DType, Type, Op) \
__kernel __attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1))) \
void broadcasted_ ## OpName ## _ ## DType( \
         __global Type *source1, \
         __global Type *source2, \
         __global Type *output, \
         __global ulong *size_mask1, \
         __global ulong *size_mask2, \
         __global ulong *result_sizes, \
         const ulong result_size, \
         const int rank) { \
    ulong index_to_parse = get_global_id(0); \
    ulong source1_index = 0, source2_index = 0; \
    for (int j = 0; j < rank - 1; ++j) { \
        ulong dim_coord = index_to_parse / result_sizes[j]; \
        source1_index += dim_coord * size_mask1[j]; \
        source2_index += dim_coord * size_mask2[j]; \
        index_to_parse = index_to_parse % result_sizes[j]; \
    } \
    source1_index += size_mask1[rank - 1] * index_to_parse; \
    source2_index += size_mask2[rank - 1] * index_to_parse; \
    if (get_global_id(0) < result_size) { \
        output[get_global_id(0)] = Op(source1[source1_index], source2[source2_index]); \
    } \
}

//broadcasted_op(plus, float16, half, +)
#define set_of_broadcasted_ops(OpName, Op, NativeOp) \
    broadcasted_op(OpName, float32, float, NativeOp) \
    broadcasted_op(OpName, float64, double, Op) \
    broadcasted_op(OpName, int8, char, Op) \
    broadcasted_op(OpName, int16, short, Op) \
    broadcasted_op(OpName, int32, int, Op) \
    broadcasted_op(OpName, int64, long, Op)

set_of_broadcasted_ops(plus, op_plus, op_plus)
set_of_broadcasted_ops(minus, op_minus, op_minus)
set_of_broadcasted_ops(multiply, op_multiply, op_multiply)
set_of_broadcasted_ops(divide, op_divide, op_divide_native)
