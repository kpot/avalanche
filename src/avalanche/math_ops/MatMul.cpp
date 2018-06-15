#include "clblast.h"

#include "avalanche/base_ops_nodes.h"
#include "avalanche/opencl_utils.h"
#include "avalanche/math_ops/messages.h"
#include "avalanche/macroses.h"
#include "avalanche/casting.h"

#include "avalanche/math_ops/MatMul.h"

namespace avalanche {

MatMul::MatMul(const NodeRef &left, const NodeRef &right,
               bool transpose_left,
               bool transpose_right)
    :transpose_left{transpose_left},
     transpose_right{transpose_right},
     _result_shape({
                      transpose_left ? left->shape().dim(1) : left->shape().dim(0),
                      transpose_right ? right->shape().dim(0) : right->shape().dim(1)}),
     _result_dtype{left->dtype()}
{
    if (left->dtype() != right->dtype()) {
        throw std::invalid_argument(
            "You cannot multiply matrices of different types");
    }
    if (left->dtype() != ArrayType::float16 &&
        left->dtype() != ArrayType::float32 &&
        left->dtype() != ArrayType::float64) {
        throw std::invalid_argument(
            "MatMul supports only data types with floating point");
    }
    if (left->shape().rank() != 2) {
        throw std::invalid_argument(
            "Left operand must be a matrix (tensor rank 2)");
    }
    if (right->shape().rank() != 2) {
        throw std::invalid_argument(
            "Right operand must be a matrix (tensor rank 2)");
    }
    if ((transpose_left ? left->shape().dim(0) : left->shape().dim(1)) !=
        (transpose_right ? right->shape().dim(1) : right->shape().dim(0))) {
        throw std::invalid_argument(
            "Number of columns in the first matrix must be the same "
                "as the number of rows in the second. With respect "
                "to transpositions, if needed.");
    }
}


template <typename T>
inline clblast::StatusCode array_gemm(
    const MultiArrayRef &a, bool transpose_a,
    const MultiArrayRef &b, bool transpose_b,
    const MultiArrayRef &result,
    cl_command_queue *queue, cl_event *result_event)
{
    auto inputs_are_ready = make_event_list(
        {a->buffer_unsafe()->completion_event(),
         b->buffer_unsafe()->completion_event()});
    if (!inputs_are_ready.empty()) {
        cl::Event::waitForEvents(inputs_are_ready);
    }
    return clblast::Gemm<T>(
        clblast::Layout::kRowMajor,
        transpose_a ? clblast::Transpose::kYes : clblast::Transpose::kNo,
        transpose_b ? clblast::Transpose::kYes : clblast::Transpose::kNo,
        static_cast<const size_t>(a->shape().dim(transpose_a ? 1 : 0)),
        static_cast<const size_t>(b->shape().dim(transpose_b ? 0 : 1)),
        static_cast<const size_t>(a->shape().dim(transpose_a ? 0 : 1)),
        to_array_type<T>(1.0f),
        a->cl_buffer_unsafe()(), 0,
        static_cast<const size_t>(a->shape().dim(-1)),
        b->cl_buffer_unsafe()(), 0,
        static_cast<const size_t>(b->shape().dim(-1)),
        to_array_type<T>(0.0f),
        result->buffer_unsafe()->cl_buffer_unsafe()(), 0,
        static_cast<const size_t>(result->shape().dim(-1)),
        queue,
        result_event
    );
}

ARRAY_DTYPE_SWITCH_FLOAT_FUNCTION(gemm_switch, array_gemm, clblast::StatusCode,)

MultiArrayRef
MatMul::forward(const MultiArrayRef &v1, const MultiArrayRef &v2) const {
    if ((transpose_left ? v1->shape().dim(0) : v1->shape().dim(1)) !=
        (transpose_right ? v2->shape().dim(1) : v2->shape().dim(0))) {
        throw std::invalid_argument(
            "Number of columns in the first matrix must be the same "
            "as the number of rows in the second. With respect "
            "to transpositions, if needed.");
    }
    Shape result_shape(
        {transpose_left ? v1->shape().dim(1) : v1->shape().dim(0),
         transpose_right ? v2->shape().dim(0) : v2->shape().dim(1)});
    auto pool = v1->buffer_unsafe()->pool();
    auto queue = pool->cl_queue();
    auto result = pool->make_array(result_shape, _result_dtype);
    result->set_label(__func__, __LINE__);
    result->add_dependencies({v1, v2});
    cl_command_queue ll_queue = queue.get();
    cl_event result_event = nullptr;
    auto status = gemm_switch(
        _result_dtype, v1, transpose_left, v2, transpose_right, result,
        &ll_queue, &result_event);
    if (status != clblast::StatusCode::kSuccess) {
        throw std::runtime_error(
            std::string("OpenCL error reported failure: ") +
            get_opencl_error_string(static_cast<int>(status)));
    }
    result->set_completion_event(result_event);
    return result;
}

const NodeRef MatMul::apply_chain_rule(const NodeRef &wrt_input,
                                       const NodeRef &d_target_wrt_this,
                                       const NodeRefList &all_inputs) const {
    if (all_inputs[0] == wrt_input) {
        if (transpose_left) {
            return F<MatMul>(all_inputs[1], d_target_wrt_this,
                             transpose_right, true);
        } else {
            return F<MatMul>(d_target_wrt_this, all_inputs[1],
                             false, !transpose_right);
        }
    } else if (all_inputs[1] == wrt_input) {
        if (transpose_right) {
            return F<MatMul>(d_target_wrt_this, all_inputs[0],
                             true, transpose_left);
        } else {
            return F<MatMul>(all_inputs[0], d_target_wrt_this,
                             !transpose_left, false);
        }
    } else {
        throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
    }
}
} // namespace
