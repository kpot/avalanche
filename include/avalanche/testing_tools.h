#ifndef AVALANCHE_TESTING_TOOLS_H
#define AVALANCHE_TESTING_TOOLS_H

#include "catch.hpp"

#include "avalanche/terminal_nodes.h"
#include "avalanche/nodes.h"
#include "avalanche/Context.h"
#include "avalanche/Shape.h"
#include "avalanche/Executor.h"

namespace avalanche {

template <typename T>
void require_almost_equal(const std::vector<T> &expected,
                          std::vector<T> &result,
                          float epsilon,
                          float margin=5e-5) {
    REQUIRE(result.size() == expected.size());
    auto mismatches = std::mismatch(
        result.begin(), result.end(), expected.begin(),
        [&](T r, T e) { return r == Approx(e).epsilon(epsilon).margin(margin); });
    CAPTURE(result);
    CAPTURE(expected);
    CAPTURE(epsilon);
    REQUIRE(mismatches.first == result.end());
}

template <typename T>
void evaluate_and_check(const NodeRef &output,
                        const std::vector<T> &expected,
                        const Shape &expected_shape,
                        ContextRef context=nullptr) {
    using namespace avalanche;
    if (!context) {
        context = Context::make_for_device(0);
    }
    Executor executor(context, {output});
    auto results = executor.run();
    REQUIRE(results[0]->dtype() == dtype_of_static_type<T>);
    std::vector<T> cpu_copy;
    results[0]->fetch_data_into(cpu_copy);
    require_almost_equal(expected, cpu_copy, 1e-6);
    REQUIRE(results[0]->shape().dims() == expected_shape.dims());
}

template <typename T>
void verify_derivatives(avalanche::ContextRef context,
                        const avalanche::NodeRefList &variables,
                        const avalanche::NodeRef output,
                        const T epsilon) {
    std::vector<T> variable_state;
    std::vector<T> function_results;
    auto grad_table = avalanche::build_back_propagation_graph(output, variables);
    avalanche::Executor executor(context, {output});

    for (auto &var_node: variables) {
        auto grad_output = grad_table[var_node];
        REQUIRE(grad_output->shape().dims() == var_node->shape().dims());
        avalanche::Executor diff_executor(context, {grad_output});
        auto var_array = context->eval(var_node);
        var_array->fetch_data_into(variable_state);
        std::vector<T> finite_diff_derivative(variable_state.size());
        auto gradient_results = diff_executor.run();
        std::vector<T> gradient;
        gradient_results[0]->fetch_data_into(gradient);
        std::vector<T> orig_variable_state(variable_state);
        for (std::size_t var_idx = 0; var_idx < variable_state.size(); ++var_idx) {
            T old_value = variable_state[var_idx];
            T results[2];
            for (int i = 0; i < 2; ++i) {
                int sign = (2 * i - 1);
                variable_state[var_idx] = old_value + sign * epsilon / 2.0;
                var_array->write_from_vector(variable_state);
                auto run_results = executor.run();
                run_results[0]->fetch_data_into(function_results);
                results[i] = 0;
                for (auto v: function_results) { results[i] += v; }
            }
            finite_diff_derivative[var_idx] = (results[1] - results[0]) / epsilon;
            variable_state[var_idx] = old_value;
        }
        var_array->write_from_vector(orig_variable_state);
        REQUIRE(gradient_results[0]->shape().agrees_with(var_node->shape().dims()));
        require_almost_equal(finite_diff_derivative, gradient, 1e-2);
    }
}


/** Broadcasting algorithm written in C++, analog to the same algorithm
 * implemented as an OpenCL kernel */
template <typename T, typename D, typename Op>
std::vector<D> broadcasted_elemwise_op(const std::vector<T> &source1,
                                       const avalanche::Shape &shape1,
                                       const std::vector<T> &source2,
                                       const avalanche::Shape &shape2,
                                       Op op) {
    std::vector<cl_ulong> size_mask1, size_mask2, result_sub_sizes;
    auto result_size = avalanche::broadcast_size_masks(
        shape1, shape2, size_mask1, size_mask2, result_sub_sizes);
    std::vector<D> result(result_size);
    for (std::size_t i = 0; i < result.size(); ++i) {
        std::size_t index_to_parse = i;
        std::size_t source1_index = 0, source2_index = 0;
//        std::cout << "index " << i << " dims: ";
        auto size_mask1_iter = size_mask1.begin(),
            size_mask2_iter = size_mask2.begin();
        for (std::size_t j = 0; j < size_mask1.size() - 1; ++j) {
            std::size_t dim_coord = index_to_parse / result_sub_sizes[j];
//            std::cout << dim_coord << ", ";
            source1_index += dim_coord * (*size_mask1_iter++);
            source2_index += dim_coord * (*size_mask2_iter++);
            index_to_parse = index_to_parse % result_sub_sizes[j];
        }
//        std::cout << index_to_parse << "\n";
        source1_index += *size_mask1_iter * index_to_parse;
        source2_index += *size_mask2_iter * index_to_parse;
        result[i] = static_cast<D>(op(source1[source1_index], source2[source2_index]));
    }
    return result;
}


// Checks that element-wise broadcasted operation keeps working
// when we swap the arguments
template <typename T, typename NodeOp, typename D, typename Op>
void test_broadcasted_elemwise_op(const std::vector<T> &source1,
                                  const avalanche::Shape &shape1,
                                  const std::vector<T> &source2,
                                  const avalanche::Shape &shape2,
                                  Op op,
                                  const std::vector<D> &expected,
                                  bool is_symmetrical = true) {
    using namespace avalanche;
    auto straight = broadcasted_elemwise_op<T, D>(
        source1, shape1, source2, shape2, op);
    auto input1 = Constant::tensor(source1, shape1);
    auto input2 = Constant::tensor(source2, shape2);
    auto output = F<NodeOp>(input1, input2);
    Executor executor(Context::make_for_device(0), {output});
    auto gpu_results = executor.run();
    REQUIRE(gpu_results[0]->dtype() == dtype_of_static_type<D>);
    std::vector<D> cpu_copy;
    gpu_results[0]->fetch_data_into(cpu_copy);
    REQUIRE(straight == expected);
    REQUIRE(cpu_copy == expected);
    if (is_symmetrical) {
        auto reversed = broadcasted_elemwise_op<T, D>(
            source2, shape2, source1, shape1, op);
        REQUIRE(reversed == expected);
    }
};

} // namespace

#endif //AVALANCHE_TESTING_TOOLS_H
