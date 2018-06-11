#define CATCH_CONFIG_MAIN

#include <numeric>
#include <functional>
#include <algorithm>
#include <numeric>


#include "catch.hpp"

#include "avalanche/terminal_nodes.h"
#include "avalanche/nodes.h"
#include "avalanche/Shape.h"
#include "avalanche/Executor.h"


TEST_CASE("Checking broadcasting shapes") {
    using namespace avalanche;
    SECTION("the first shape is larger") {
        Shape shape1({2, 3, 1}), shape2({2});
        auto aligned_shapes = Shape::align_for_broadcasting(shape1, shape2);
        REQUIRE(aligned_shapes[0] == Shape({2, 3, 1}));
        REQUIRE(aligned_shapes[1] == Shape({1, 1, 2}));
        REQUIRE(aligned_shapes[2] == Shape({2, 3, 2}));
    }

    SECTION("the second shape is larger") {
        {
            Shape shape1({2}), shape2({2, 3, 1});
            auto aligned_shapes = Shape::align_for_broadcasting(shape1, shape2);
            REQUIRE(aligned_shapes[0] == Shape({1, 1, 2}));
            REQUIRE(aligned_shapes[1] == Shape({2, 3, 1}));
            REQUIRE(aligned_shapes[2] == Shape({2, 3, 2}));
        }
        {
            Shape shape1({3, 5}), shape2({2, 3, 5});
            auto aligned_shapes = Shape::align_for_broadcasting(shape1, shape2);
            REQUIRE(aligned_shapes[0] == Shape({1, 3, 5}));
            REQUIRE(aligned_shapes[1] == Shape({2, 3, 5}));
            REQUIRE(aligned_shapes[2] == Shape({2, 3, 5}));
        }
    }

    SECTION("The shapes must be incompatible") {
        {
            Shape shape1({3, 2}), shape2({2, 3, 5});
            CHECK_THROWS_WITH(
                Shape::align_for_broadcasting(shape1, shape2),
                Catch::Contains("Cannot align shapes"));
        }
        {

            CHECK_THROWS_WITH(
                Shape({2, 0, 5}),
                Catch::Contains("impossible shape"));
            Shape shape1({2, 8, 5}), shape2({2, 3, 5});
            CHECK_THROWS_WITH(
                Shape::align_for_broadcasting(shape1, shape2),
                Catch::Contains("Cannot align shapes"));
        }
    }
}




template <typename T, typename Op>
std::vector<T> broadcasted_elemwise_op(const std::vector<T> &source1,
                                       const avalanche::Shape &shape1,
                                       const std::vector<T> &source2,
                                       const avalanche::Shape &shape2,
                                       Op op) {
    std::vector<cl_ulong> size_mask1, size_mask2, result_sub_sizes;
    auto result_size = avalanche::broadcast_size_masks(
        shape1, shape2, size_mask1, size_mask2, result_sub_sizes);
    std::vector<T> result(result_size);
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
        result[i] = op(source1[source1_index], source2[source2_index]);
    }
    return result;
}


// Checks that element-wise broadcasted operation keeps working
// when we swap the arguments
template <typename T, typename NodeOp, typename Op>
void test_broadcasted_elemwise_op(
        const std::vector<T> &source1,
        const avalanche::Shape &shape1,
        const std::vector<T> &source2,
        const avalanche::Shape &shape2,
        Op op,
        const std::vector<T> &expected) {
    using namespace avalanche;
    auto straight = broadcasted_elemwise_op<T>(
        source1, shape1, source2, shape2, op);
    auto reversed = broadcasted_elemwise_op<T>(
        source2, shape2, source1, shape1, op);
    auto input1 = Constant::tensor(source1, shape1);
    auto input2 = Constant::tensor(source2, shape2);
    auto output = F<NodeOp>(input1, input2);
    Executor executor(Context::make_for_device(0), {output});
    auto gpu_results = executor.run();
    std::vector<T> cpu_copy;
    gpu_results[0]->fetch_data_into(cpu_copy);
    REQUIRE(straight == expected);
    REQUIRE(reversed == expected);
    REQUIRE(cpu_copy == expected);
};


TEST_CASE("Broadcasting element-wise ops") {
    using namespace avalanche;

    SECTION("Multiplying two scalars") {
        test_broadcasted_elemwise_op<float, Multiply>(
            {2}, Shape(),
            {8}, Shape(),
            std::multiplies<>(),
            // Expected
            {16});
    }
    SECTION("Summing two tensors the rank of 3 and 1") {
        test_broadcasted_elemwise_op<float, Plus>(
            {1, 2, 3, 4, 5, 6}, Shape({2, 3, 1}),
            {1, 2}, Shape({2}),
            std::plus<>(),
            // Expected
            {2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8});
    }

    SECTION("Summing tensors the shape of {2, 3, 1} and {1} (like a scalar)") {
        test_broadcasted_elemwise_op<float, Plus>(
            {1, 2, 3, 4, 5, 6}, Shape({2, 3, 1}),
            {1}, Shape({1}),
            std::plus<>(),
            // Expected
            {2, 3, 4, 5, 6, 7});
    }

    SECTION("Multiplying a matrix to a scalar value (empty shape)") {
        test_broadcasted_elemwise_op<float, Multiply>(
            {1, 2, 3, 4, 5, 6}, Shape({2, 3, 1}),
            {2}, Shape(),
            std::multiplies<>(),
            // Expected
            {2, 4, 6, 8, 10, 12});
    }
    SECTION("More complicated case") {
        test_broadcasted_elemwise_op<float, Plus>(
            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23}, Shape({4, 3, 1, 2}),
            {0, 1, 2, 3, 4, 5, 6, 7}, Shape({4, 1, 1, 2}),
            std::plus<>(),
            {0, 2, 2, 4, 4, 6, 8, 10, 10, 12, 12, 14, 16, 18, 18, 20, 20, 22,
             24, 26, 26, 28, 28, 30});
    }
    SECTION("All dimensions are 1") {
        test_broadcasted_elemwise_op<float, Multiply>(
            {2}, Shape({1, 1, 1, 1, 1}),
            {8}, Shape({1, 1, 1}),
            std::multiplies<>(),
            // Expected
            {16});
    }
    SECTION("All trailing dimensions are 1") {
        test_broadcasted_elemwise_op<float, Multiply>(
            {1, 2, 3, 4, 5}, Shape({5, 1, 1, 1}),
            {1, 2, 3, 4, 5}, Shape({5, 1, 1, 1}),
            std::multiplies<>(),
            // Expected
            {1, 4, 9, 16, 25});
    }
    SECTION("Previously failed on #1") {
        test_broadcasted_elemwise_op<float, Plus>(
            {0.0f, 3.0f, 6.0,
             1.0, 4.0, 7.0,
             2.0, 5.0, 8.0}, Shape({3, 3}),
            {0.5f, 0.0f, -0.5f}, Shape({3}),
            std::plus<>(),
            // Expected
            {0.5, 3.0, 5.5, 1.5, 4.0, 6.5, 2.5, 5.0, 7.5});
    }
}


TEST_CASE("Testing matrix multiplication") {
    using namespace avalanche;
    auto val1 = Constant::tensor<float>(
        {0.0f, 1.0f, 2.0f,
         3.0f, 4.0f, 5.0f,
         6.0f, 7.0f, 8.0f,
         9.0f, 10.0f, 11.0f},
        Shape({4, 3}));
    // The same matrix transposed
    auto val2 = Constant::tensor<float>(
        {0, 3, 6, 9,
         1, 4, 7, 10,
         2, 5, 8, 11},
        Shape({3, 4}));
    std::vector<float> expected(
        {5.0, 14.0, 23.0, 32.0,
         14.0, 50.0, 86.0, 122.0,
         23.0, 86.0, 149.0, 212.0,
         32.0, 122.0, 212.0, 302.0});

    SECTION("Multiplication without transposition") {
        auto output = F<MatMul>(val1, val2, false, false);
        Executor executor(Context::make_for_device(0), {output});
        auto results = executor.run();
        std::vector<float> cpu_copy;
        results[0]->fetch_data_into(cpu_copy);
        REQUIRE(cpu_copy == expected);
    }

    SECTION("Right value transposed by the MatMul itself") {
        auto output = F<MatMul>(val1, val1, false, true);
        Executor executor(Context::make_for_device(0), {output});
        auto results = executor.run();
        std::vector<float> cpu_copy;
        results[0]->fetch_data_into(cpu_copy);
        REQUIRE(cpu_copy == expected);
    }

    SECTION("Left value transposed by the MatMul") {
        auto output = F<MatMul>(val1, val1, true, false);
        Executor executor(Context::make_for_device(0), {output});
        auto results = executor.run();
        std::vector<float> cpu_copy;
        results[0]->fetch_data_into(cpu_copy);
        REQUIRE(
            cpu_copy ==
            std::vector<float>({126, 144, 162, 144, 166, 188, 162, 188, 214}));
    }

    SECTION("Both values transposed by the MatMul") {
        auto output = F<MatMul>(val1, val2, true, true);
        Executor executor(Context::make_for_device(0), {output});
        auto results = executor.run();
        std::vector<float> cpu_copy;
        results[0]->fetch_data_into(cpu_copy);
        REQUIRE(
            cpu_copy ==
            std::vector<float>({126, 144, 162, 144, 166, 188, 162, 188, 214}));
    }
}

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
void evaluate_and_check(const avalanche::NodeRef &output,
                        const std::vector<T> &expected,
                        const avalanche::Shape &expected_shape,
                        avalanche::ContextRef context=nullptr) {
    using namespace avalanche;
    if (!context) {
        context = Context::make_for_device(0);
    }
    Executor executor(context, {output});
    auto results = executor.run();
    std::vector<T> cpu_copy;
    results[0]->fetch_data_into(cpu_copy);
    require_almost_equal(expected, cpu_copy, 1e-6);
    REQUIRE(results[0]->shape() == expected_shape);
}


TEST_CASE("Testing scaling") {
    using namespace avalanche;
    auto val1 = Constant::tensor<float>(
        {0.0f, 1.0f, 2.0f,
         3.0f, 4.0f, 5.0f,
         6.0f, 7.0f, 8.0f,
         9.0f, 10.0f, 11.0f},
        Shape({4, 3}));
    auto output = FU<Scale>(val1, 2.5);
    evaluate_and_check<float>(
        output,
        {0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5},
        Shape({4, 3}));
}


TEST_CASE("Test various transformations") {
    using namespace avalanche;

    SECTION("Raising to a power") {
        auto val1 = Constant::tensor<float>(
            {0.0f, 1.0f, 2.0f,
             3.0f, 4.0f, 5.0f,
             6.0f, 7.0f, 8.0f,
             9.0f, 10.0f, 11.0f},
            Shape({4, 3}));
        auto output = FU<SPower>(val1, 0.5, 2);
        evaluate_and_check<float>(
            output,
            {0.0, 0.5, 2.0, 4.5, 8.0, 12.5, 18.0, 24.5, 32.0, 40.5, 50.0, 60.5},
            Shape({4, 3}));
    }

    SECTION("Natural logarithm") {
        auto val1 = Constant::tensor<float>(
            {1.0f, 2.0f, 3.0f},
            Shape({3}));
        auto output = FU<Log>(val1);
        evaluate_and_check<float>(
            output,
            {0.0f, 0.69314718f, 1.09861229f},
            Shape({3}));
    }

    SECTION("Reciprocal numbers") {
        auto val1 = Constant::tensor<float>(
            {1.0f, 2.0f, 10.0f},
            Shape({3}));
        auto output = FU<Recip>(val1);
        evaluate_and_check<float>(
            output,
            {1.0, 0.5, 0.1},
            Shape({3}));
    }

    SECTION("Sigmoid") {
        auto val1 = Constant::tensor<float>(
            {-5.0f, 0.0f, 5.0f},
            Shape({3}));
        auto output = FU<Sigmoid>(val1);
        evaluate_and_check<float>(
            output,
            {0.006692851f, 0.5f, 0.993307149f},
            Shape({3}));
    }

    SECTION("Tanh") {
        auto val1 = Constant::tensor<float>(
            {-5.0f, 0.0f, 5.0f},
            Shape({3}));
        auto output = FU<Tanh>(val1);
        evaluate_and_check<float>(
            output,
            {-0.9999092042625951f, 0.0f, 0.9999092042625951f},
            Shape({3}));
    }

    SECTION("ReduceSum") {
        auto val1 = Constant::tensor<float>(
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
            Shape({2, 3, 2}));
//        // Reducing empty list of dimensions. Should not change anything
//        auto output0 = FU<ReduceSum>(val1, std::vector<ShapeDim>({}));
//        evaluate_and_check<float>(
//            output0,
//            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
//            Shape({2, 3, 2}));
        // Reducing one negative dimension (last)
        auto output1 = FU<ReduceSum>(val1, std::vector<ShapeDim>({-1}));
        evaluate_and_check<float>(
            output1,
            {1, 5, 9, 13, 17, 21},
            Shape({2, 3}));
        // Reducing middle dimension only
        auto output2 = FU<ReduceSum>(val1, std::vector<ShapeDim>({1}));
        evaluate_and_check<float>(
            output2,
            {6, 9, 24, 27},
            Shape({2, 2}));
        // Reducing first dimension only
        auto output3 = FU<ReduceSum>(val1, std::vector<ShapeDim>({0}));
        evaluate_and_check<float>(
            output3,
            {6, 8, 10, 12, 14, 16},
            Shape({3, 2}));
        // Reducing a combination of dimension
        auto output4 = FU<ReduceSum>(val1, std::vector<ShapeDim>({0, 2}));
        evaluate_and_check<float>(
            output4,
            {14, 22, 30},
            Shape({3}));
        // Full reduction on a small value
        auto output5 = FU<ReduceSum>(val1);
        evaluate_and_check<float>(
            output5,
            {66},
            Shape());
        // Reduction of all dimensions should have the same effect
        auto output5_1 = FU<ReduceSum>(val1, std::vector<ShapeDim>({1, 2, 0}));
        evaluate_and_check<float>(
            output5_1,
            {66},
            Shape());
        // Full reduction of a long vector (longer than get_global_size(0))
        std::vector<float> big_tensor(10001);
        std::iota(big_tensor.begin(), big_tensor.end(), 0);
        auto big_value = Constant::tensor<float>(
            big_tensor, Shape({static_cast<ShapeDim>(big_tensor.size())}));
        auto output6 = FU<ReduceSum>(big_value);
        evaluate_and_check<float>(
            output6,
            {50005000.0},
            Shape());
    }

    SECTION("ReduceMean") {
        // Partial reduction over last dimension
        auto val = Constant::tensor<float>(
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
            Shape({2, 3, 2}));
        auto output1 = FU<ReduceMean>(val, std::vector<ShapeDim>({-1}));
        evaluate_and_check<float>(
            output1,
            {0.5, 2.5, 4.5, 6.5, 8.5, 10.5},
            Shape({2, 3}));
        // partial reduction over first dimension
        auto output2 = FU<ReduceMean>(val, std::vector<ShapeDim>({0}));
        evaluate_and_check<float>(
            output2,
            {3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
            Shape({3, 2}));
        // Full reduction
        auto output3 = FU<ReduceMean>(val);
        evaluate_and_check<float>(output3, {5.5}, Shape());
    }
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
        std::cout << grad_output->to_string() << std::endl;
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
        REQUIRE(gradient_results[0]->shape().dims() == var_node->shape().dims());
        require_almost_equal(finite_diff_derivative, gradient, 1e-2);
    }
}

TEST_CASE("Checking automatic derivations (backprop)") {
    using namespace avalanche;

    SECTION("Sigmoid") {
        auto input = Variable::make("input1", {3}, ArrayType::float32);
        auto output = FU<Sigmoid>(input);
        auto context = Context::make_for_device(0);
        context->init<float>(input, {0.0f, 1.0f, 2.0}, input->shape());
        verify_derivatives<float>(context, {input}, output, 1e-2);
    }

    SECTION("MatMul") {
        auto input1 = Variable::make("input1", {3, 3}, ArrayType::float32);
        auto input2 = Variable::make("input2", {3, 3}, ArrayType::float32);
        auto output = F<MatMul>(input1, input2);
        auto context = Context::make_for_device(0);
        context->init<float>(
            input1,
            {0.0f, 1.0f, 2.0,
             3.0, 4.0, 5.0,
             6.0, 7.0, 8.0},
            input1->shape());
        context->init<float>(
            input2,
            {0.0f, 3.0f, 6.0,
             1.0, 4.0, 7.0,
             2.0, 5.0, 8.0},
            input1->shape());
        verify_derivatives<float>(context, {input1, input2}, output, 1e-2);
    }

    SECTION("Broadcasted plus") {
        auto weights = Variable::make("weights", {3, 3} , ArrayType::float32);
        auto biases = Variable::make("biases", {3}, ArrayType::float32);
        auto output = F<Plus>(weights, biases);
        auto context = Context::make_for_device(0);
        context->init<float>(
            weights,
            {0.0f, 3.0f, 6.0,
             1.0, 4.0, 7.0,
             2.0, 5.0, 8.0},
            weights->shape());
        context->init<float>(
            biases,
            {0.5, 0.0, -0.5},
            biases->shape());
        evaluate_and_check<float>(
            output,
            {0.5, 3.0, 5.5, 1.5, 4.0, 6.5, 2.5, 5.0, 7.5},
            Shape({3, 3}),
            context);
        verify_derivatives<float>(context, {weights, biases}, output, 1e-2);
    }

    SECTION("Broadcasted minus") {
        auto weights = Variable::make("weights", {3, 3} , ArrayType::float32);
        auto biases = Variable::make("biases", {3}, ArrayType::float32);
        auto output = F<Minus>(weights, biases);
        auto context = Context::make_for_device(0);
        context->init<float>(
            weights,
            {0.0f, 3.0f, 6.0,
             1.0, 4.0, 7.0,
             2.0, 5.0, 8.0},
            weights->shape());
        context->init<float>(
            biases,
            {0.5, 0.0, -0.5},
            biases->shape());
        evaluate_and_check<float>(
            output,
            {-0.5, 3.0, 6.5, 0.5, 4.0, 7.5, 1.5, 5.0, 8.5},
            Shape({3, 3}),
            context);
        verify_derivatives<float>(context, {weights, biases}, output, 1e-2);
    }

    SECTION("Fully-connected feed-forward layer with sigmoid activation") {
        auto inputs = Variable::make("inputs", {3, 3}, ArrayType::float32);
        auto weights = Variable::make("weights", {3, 3}, ArrayType::float32);
        auto biases = Variable::make("biases", {3}, ArrayType::float32);
        auto output = FU<Sigmoid>(F<Plus>(F<MatMul>(inputs, weights), biases));
        auto context = Context::make_for_device(0);
        context->init<float>(
            inputs,
            {0.0f, 1.0f, 2.0,
             3.0, 4.0, 5.0,
             6.0, 7.0, 8.0},
            inputs->shape());
        context->init<float>(
            weights,
            {0.0f, 3.0f, 6.0,
             1.0, 4.0, 7.0,
             2.0, 5.0, 8.0},
            weights->shape());
        context->init<float>(
            biases,
            {0.5f, 0.0f, -0.5},
            biases->shape());
        verify_derivatives<float>(context, {inputs, weights, biases}, output, 0.05);
    }

    SECTION("Fully-connected feed-forward layer with tanh activation") {
        auto inputs = Variable::make("inputs", {3, 3}, ArrayType::float32);
        auto weights = Variable::make("weights", {3, 3}, ArrayType::float32);
        auto biases = Variable::make("biases", {3}, ArrayType::float32);
        auto output = FU<Tanh>(F<Plus>(F<MatMul>(inputs, weights), biases));
        auto context = Context::make_for_device(0);
        context->init<float>(
            inputs,
            {0.0f, 1.0f, 2.0,
             3.0, 4.0, 5.0,
             6.0, 7.0, 8.0},
            inputs->shape());
        context->init<float>(
            weights,
            {0.0f, 3.0f, 6.0,
             1.0, 4.0, 7.0,
             2.0, 5.0, 8.0},
            weights->shape());
        context->init<float>(
            biases,
            {0.5f, 0.0f, -0.5},
            biases->shape());
        verify_derivatives<float>(context, {inputs, weights, biases}, output, 0.05);
    }

    SECTION("Fully-connected feed-forward layer with ReLU activation") {
        auto inputs = Variable::make("inputs", {3, 3}, ArrayType::float32);
        auto weights = Variable::make("weights", {3, 3}, ArrayType::float32);
        auto biases = Variable::make("biases", {3}, ArrayType::float32);
        auto output = FU<ReLU>(F<Plus>(F<MatMul>(inputs, weights), biases));
        auto context = Context::make_for_device(0);
        context->init<float>(
            inputs,
            {0.0f, 1.0f, 2.0,
             3.0, 4.0, 5.0,
             6.0, 7.0, 8.0},
            inputs->shape());
        context->init<float>(
            weights,
            {0.0f, 3.0f, 6.0,
             1.0, 4.0, 7.0,
             2.0, 5.0, 8.0},
            weights->shape());
        context->init<float>(
            biases,
            {0.5f, 0.0f, -0.5},
            biases->shape());
        verify_derivatives<float>(context, {inputs, weights, biases}, output, 0.05);
    }

    SECTION("Full reduce mean") {
        auto inputs = Variable::make("inputs", {3, 3}, ArrayType::float32);
        auto weights = Variable::make("weights", {3, 3}, ArrayType::float32);
        auto output = FU<Scale>(FU<ReduceMean>(F<Minus>(inputs, weights)), 2);
        auto context = Context::make_for_device(0);
        context->init<float>(
            inputs,
            {0.0f, 1.0f, 2.0,
             3.0, 4.0, 5.0,
             6.0, 7.0, 8.0},
            inputs->shape());
        context->init<float>(
            weights,
            {0.0f, 3.0f, 6.0,
             1.0, 4.0, 7.0,
             2.0, 5.0, 8.0},
            weights->shape());
        verify_derivatives<float>(context, {inputs, weights}, output, 0.05);
    }

    SECTION("Full reduce sum") {
        auto inputs = Variable::make("inputs", {3, 3}, ArrayType::float32);
        auto weights = Variable::make("weights", {3, 3}, ArrayType::float32);
        auto output = FU<Scale>(FU<ReduceSum>(F<Minus>(inputs, weights)), 2);
        auto context = Context::make_for_device(0);
        context->init<float>(
            inputs,
            {0.0f, 1.0f, 2.0,
             3.0, 4.0, 5.0,
             6.0, 7.0, 8.0},
            inputs->shape());
        context->init<float>(
            weights,
            {0.0f, 3.0f, 6.0,
             1.0, 4.0, 7.0,
             2.0, 5.0, 8.0},
            weights->shape());
        verify_derivatives<float>(context, {inputs, weights}, output, 0.05);
    }

    SECTION("Reshape") {
        auto data = Variable::make("data", {1, 3}, ArrayType::float32);
        auto output = FU<Reshape>(F<Exp>(data), Shape({3, 1}));
        auto context = Context::make_for_device(0);
        context->init<float>(
            data,
            {0.0f, 1.0f, 2.0},
            data->shape());
        verify_derivatives<float>(context, {data}, output, 0.05);
    }

    SECTION("Derivatives of ReduceSum") {
        auto data = Variable::make("data", {3, 1}, ArrayType::float32);
        auto output = FU<ReduceSum>(data, std::vector<ShapeDim>({-1}));
        auto context = Context::make_for_device(0);
        context->init<float>(
            data,
            {0.0f, 1.0f, 2.0},
            data->shape());
        verify_derivatives<float>(context, {data}, output, 0.05);
    }

    SECTION("Derivatives of Softmax #1") {
        auto data = Variable::make("data", {3, 1}, ArrayType::float32);
        auto output = softmax(data, -1);
        auto context = Context::make_for_device(0);
        context->init<float>(
            data,
            {0.0f, 1.0f, 2.0},
            data->shape());
        verify_derivatives<float>(context, {data}, output, 0.05);
    }

    SECTION("Derivatives of Softmax #2") {
        auto data = Variable::make("data", {3}, ArrayType::float32);
        auto output = softmax(data, -1);
        auto context = Context::make_for_device(0);
        context->init<float>(
            data,
            {0.0f, 1.0f, 2.0},
            data->shape());
        verify_derivatives<float>(context, {data}, output, 0.05);
    }

    SECTION("Derivatives of broadcasted subtraction #2") {
        auto data1 = Variable::make("data1", {3, 2}, ArrayType::float32);
        auto data2 = Variable::make("data2", {3, 1}, ArrayType::float32);
        auto output = data1 - data2;
        auto context = Context::make_for_device(0);
        context->init<float>(
            data1,
            {0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f},
            data1->shape());
        context->init<float>(
            data2,
            {1.0f, 2.0f, 3.0f},
            data2->shape());
        verify_derivatives<float>(context, {data1, data2}, output, 0.05);
    }

    SECTION("Derivatives of broadcasted addition #2") {
        auto data1 = Variable::make("data1", {3, 2}, ArrayType::float32);
        auto data2 = Variable::make("data2", {3, 1}, ArrayType::float32);
        auto output = data1 + data2;
        auto context = Context::make_for_device(0);
        context->init<float>(
            data1,
            {0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f},
            data1->shape());
        context->init<float>(
            data2,
            {1.0f, 2.0f, 3.0f},
            data2->shape());
        verify_derivatives<float>(context, {data1, data2}, output, 0.05);
    }

    SECTION("Derivatives of broadcasted division") {
        auto data1 = Variable::make("data1", {3, 2}, ArrayType::float32);
        auto data2 = Variable::make("data2", {3, 1}, ArrayType::float32);
        auto output = data1 / data2;
        auto context = Context::make_for_device(0);
        context->init<float>(
            data1,
            {0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f},
            data1->shape());
        context->init<float>(
            data2,
            {1.0f, 2.0f, 3.0f},
            data2->shape());
        verify_derivatives<float>(context, {data1, data2}, output, 0.05);
    }

    SECTION("Derivatives of broadcasted multiplication") {
        auto data1 = Variable::make("data1", {3, 2}, ArrayType::float32);
        auto data2 = Variable::make("data2", {3, 1}, ArrayType::float32);
        auto output = data1 * data2;
        auto context = Context::make_for_device(0);
        context->init<float>(
            data1,
            {0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f},
            data1->shape());
        context->init<float>(
            data2,
            {1.0f, 2.0f, 3.0f},
            data2->shape());
        verify_derivatives<float>(context, {data1, data2}, output, 0.05);
    }

    SECTION("Derivatives of MLP with softmax output") {
        auto inputs = Variable::make("inputs", {3, 3}, ArrayType::float32);
        auto weights1 = Variable::make("weights1", {3, 3}, ArrayType::float32);
        auto biases1 = Variable::make("biases1", {3}, ArrayType::float32);
        auto weights2 = Variable::make("weights2", {3, 3}, ArrayType::float32);
        auto biases2 = Variable::make("biases2", {3}, ArrayType::float32);
        auto output1 = FU<Sigmoid>(F<Plus>(F<MatMul>(inputs, weights1), biases1));
        auto output2 = softmax(F<Plus>(F<MatMul>(output1, weights2), biases2));
        auto context = Context::make_for_device(0);
        context->init<float>(
            inputs,
            {0.0f, 1.0f, 2.0,
             3.0, 4.0, 5.0,
             6.0, 7.0, 8.0},
            inputs->shape());
        context->init<float>(
            weights1,
            {0.0f, 3.0f, 6.0,
             1.0, 4.0, 7.0,
             2.0, 5.0, 8.0},
            weights1->shape());
        context->init<float>(
            biases1,
            {0.5f, 0.0f, -0.5},
            biases2->shape());
        context->init<float>(
            weights2,
            {0.0f, 3.0f, 6.0,
             1.0, 4.0, 7.0,
             2.0, 5.0, 8.0},
            weights2->shape());
        context->init<float>(
            biases2,
            {0.5f, 0.0f, -0.5},
            biases2->shape());
        verify_derivatives<float>(
            context,
            {inputs, weights1, biases1, weights2, biases2},
            output2, 0.05);
    }
}
