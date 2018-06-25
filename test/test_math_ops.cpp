#define CATCH_CONFIG_MAIN

#include <numeric>
#include <functional>
#include <algorithm>
#include <numeric>

#include "avalanche/testing_tools.h"

using namespace avalanche;

TEST_CASE("Broadcasting element-wise ops") {

    SECTION("Multiplying two scalars") {
        test_broadcasted_elemwise_op<float, Multiply, float>(
            {2}, Shape(),
            {8}, Shape(),
            std::multiplies<>(),
            // Expected
            {16});
    }
    SECTION("Summing two tensors the rank of 3 and 1") {
        test_broadcasted_elemwise_op<float, Plus, float>(
            {1, 2, 3, 4, 5, 6}, Shape({2, 3, 1}),
            {1, 2}, Shape({2}),
            std::plus<>(),
            // Expected
            {2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8});
    }

    SECTION("Summing tensors the shape of {2, 3, 1} and {1} (like a scalar)") {
        test_broadcasted_elemwise_op<float, Plus, float>(
            {1, 2, 3, 4, 5, 6}, Shape({2, 3, 1}),
            {1}, Shape({1}),
            std::plus<>(),
            // Expected
            {2, 3, 4, 5, 6, 7});
    }

    SECTION("Multiplying a matrix to a scalar value (empty shape)") {
        test_broadcasted_elemwise_op<float, Multiply, float>(
            {1, 2, 3, 4, 5, 6}, Shape({2, 3, 1}),
            {2}, Shape(),
            std::multiplies<>(),
            // Expected
            {2, 4, 6, 8, 10, 12});
    }
    SECTION("More complicated case") {
        test_broadcasted_elemwise_op<float, Plus, float>(
            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23}, Shape({4, 3, 1, 2}),
            {0, 1, 2, 3, 4, 5, 6, 7}, Shape({4, 1, 1, 2}),
            std::plus<>(),
            {0, 2, 2, 4, 4, 6, 8, 10, 10, 12, 12, 14, 16, 18, 18, 20, 20, 22,
             24, 26, 26, 28, 28, 30});
    }
    SECTION("All dimensions are 1") {
        test_broadcasted_elemwise_op<float, Multiply, float>(
            {2}, Shape({1, 1, 1, 1, 1}),
            {8}, Shape({1, 1, 1}),
            std::multiplies<>(),
            // Expected
            {16});
    }
    SECTION("All trailing dimensions are 1") {
        test_broadcasted_elemwise_op<float, Multiply, float>(
            {1, 2, 3, 4, 5}, Shape({5, 1, 1, 1}),
            {1, 2, 3, 4, 5}, Shape({5, 1, 1, 1}),
            std::multiplies<>(),
            // Expected
            {1, 4, 9, 16, 25});
    }
    SECTION("Previously failed on #1") {
        test_broadcasted_elemwise_op<float, Plus, float>(
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



TEST_CASE("Testing scaling") {
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

    SECTION("ReduceSum to be like some other node") {
        auto data1 = Variable::make("data1", {3, 2}, ArrayType::float32);
        auto data2 = Variable::make("data2", {3, 1}, ArrayType::float32);
        auto output = F<ReduceSum>(data1, F<NoBackProp>(data2), true);
        REQUIRE(output->shape() == Shape({3, 1}));
        auto context = Context::make_for_device(0);
        context->init<float>(
            data1,
            {0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f},
            data1->shape());
        context->init<float>(
            data2,
            {1.0f, 2.0f, 3.0f},
            data2->shape());
        Executor executor(context, {output});
        auto results = executor.run();
        REQUIRE(results[0]->shape() == Shape({3, 1}));
        std::vector<float> fetched;
        results[0]->fetch_data_into(fetched);
        REQUIRE(fetched == std::vector<float>({1.0f, 3.0f, 5.0f}));
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

    SECTION("Casting") {
        auto inputs = Variable::make("inputs", {3, 3}, ArrayType::float32);
        auto output = FU<Cast>(inputs, ArrayType::int32);
        auto context = Context::make_for_device(0);
        context->init<float>(
            inputs,
            {0.0f, 1.0f, 2.0,
             3.0, 4.0, 5.0,
             6.0, 7.0, 8.0},
            inputs->shape());
        REQUIRE(output->dtype() == ArrayType::int32);
        evaluate_and_check<std::int32_t>(
            output, {0, 1, 2, 3, 4, 5, 6, 7, 8}, inputs->shape(), context);
    }

    SECTION("ProductOfDims") {
        auto inputs = Variable::make("inputs", {3, 3}, ArrayType::float32);
        auto context = Context::make_for_device(0);
        context->init<float>(
            inputs,
            {0.0f, 1.0f, 2.0,
             3.0, 4.0, 5.0,
             6.0, 7.0, 8.0},
            inputs->shape());
        auto output = FU<ProductOfDims>(inputs, std::vector<ShapeDim>(), ArrayType::float32);
        evaluate_and_check<float>(output, {9.0f}, Shape(), context);
    }
}


TEST_CASE("Checking automatic derivations (backprop)") {

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

    SECTION("Elemwise plus") {
        auto weights = Variable::make("weights", {3, 3} , ArrayType::float32);
        auto biases = Variable::make("biases", {3, 3}, ArrayType::float32);
        auto output = F<ElemWisePlus>(weights, biases);
        auto context = Context::make_for_device(0);
        context->init<float>(
            weights,
            {0.0f, 3.0f, 6.0,
             1.0, 4.0, 7.0,
             2.0, 5.0, 8.0},
            weights->shape());
        context->init<float>(
            biases,
            {0.5, 0.0, -0.5,
             0.5, 0.0, -0.5,
             0.5, 0.0, -0.5},
            biases->shape());
        evaluate_and_check<float>(
            output,
            {0.5, 3.0, 5.5, 1.5, 4.0, 6.5, 2.5, 5.0, 7.5},
            Shape({3, 3}),
            context);
        verify_derivatives<float>(context, {weights, biases}, output, 1e-2);
    }

    SECTION("Elemwise multiply") {
        auto weights = Variable::make("weights", {3, 3} , ArrayType::float32);
        auto biases = Variable::make("biases", {3, 3}, ArrayType::float32);
        auto output = F<ElemWiseMultiply>(weights, biases);
        auto context = Context::make_for_device(0);
        context->init<float>(
            weights,
            {0.0f, 3.0f, 6.0,
             1.0, 4.0, 7.0,
             2.0, 5.0, 8.0},
            weights->shape());
        context->init<float>(
            biases,
            {0.5, 0.0, -0.5,
             0.5, 0.0, -0.5,
             0.5, 0.0, -0.5},
            biases->shape());
        evaluate_and_check<float>(
            output,
            {0.0, 0.0, -3, 0.5, 0.0, -3.5, 1.0, 0.0, -4.0},
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

    SECTION("Raising to a scalar power") {
        auto val1 = Variable::make(
            "value", {4, 3}, ArrayType::float32);
        auto context = Context::make_for_device(0);
        context->init<float>(
            val1,
            {0.0f, 1.0f, 2.0f,
             3.0f, 4.0f, 5.0f,
             6.0f, 7.0f, 8.0f,
             9.0f, 10.0f, 11.0f},
            val1->shape());
        auto output = FU<SPower>(val1, 0.5, 2);
        evaluate_and_check<float>(
            output,
            {0.0, 0.5, 2.0, 4.5, 8.0, 12.5, 18.0, 24.5, 32.0, 40.5, 50.0, 60.5},
            Shape({4, 3}),
            context);
        verify_derivatives<float>(context, {val1}, output, 1e-2);
    }

    SECTION("Raising to a power (x ^ y) with broadcasting") {
        auto val1 = Variable::make(
            "value", {4, 3}, ArrayType::float32);
        auto val2 = Variable::make("power", {}, ArrayType::float32);
        auto context = Context::make_for_device(0);
        context->init<float>(
            val1,
            {0.5f, 1.0f, 2.0f,
             3.0f, 1.0f, 2.0f,
             3.0f, 1.0f, 2.0f,
             3.0f, 1.0f, 2.0f},
            val1->shape());
        context->init<float>(val2, {2}, val2->shape());
        auto output = F<Power>(val1, val2);
        evaluate_and_check<float>(
            output,
            {0.25, 1, 4, 9, 1, 4, 9, 1, 4, 9, 1, 4},
            Shape({4, 3}),
            context);
        verify_derivatives<float>(context, {val1, val2}, output, 1e-2);
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

    SECTION("Broadcasted subtraction of nodes of unknown shape") {
        auto var1 = Variable::make("input1", {UnknownDim, UnknownDim}, ArrayType::float32);
        auto var2 = Variable::make("input2", {UnknownDim, 3}, ArrayType::float32);
        auto output = F<Minus>(var1, var2);
        REQUIRE(output->shape() == Shape({UnknownDim, 3}));
        auto context = Context::make_for_device(0);
        context->init<float>(
            var1,
            {1,  2,  3,
             4,  5,  6,
             7,  8,  9,
             10, 11, 12},
            Shape({4, 3}));
        context->init<float>(
            var2,
            {1,  2,  3},
            Shape({1, 3}));
        verify_derivatives<float>(context, {var1, var2}, output, 0.05);
    }

    SECTION("Derivatives of broadcasted multiplication of equivalient shapes") {
        auto var1 = Variable::make("var1", {1, 5}, ArrayType::float32);
        auto var2 = Variable::make("var2", {5}, ArrayType::float32);
        auto output = F<Multiply>(var1, var2);
        REQUIRE(output->shape() == Shape({1, 5}));
        auto context = Context::make_for_device(0);
        context->init<float>(var1, {0.0f, 1.0f, 2.0, 3.0, 4.0});
        context->init<float>(var2, {1.0f, 2.0f, 3.0, 4.0, 5.0});
        evaluate_and_check<float>(output, {0, 2, 6, 12, 20}, {1, 5}, context);
        verify_derivatives<float>(context, {var1, var2}, output, 0.05);
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


TEST_CASE("Checking comparisons") {

    SECTION("Checking inequality") {
        test_broadcasted_elemwise_op<float, NotEqual, std::int8_t>(
            {0, 2, 0, 4, 5, 6}, Shape({2, 3, 1}),
            {0}, Shape(),
            std::not_equal_to<>(),
            // Expected
            {0, 1, 0, 1, 1, 1});
    }

    SECTION("Checking equality") {
        test_broadcasted_elemwise_op<float, Equal, std::int8_t>(
            {0, 2, 0, 4, 5, 6}, Shape({2, 3, 1}),
            {0}, Shape(),
            std::equal_to<>(),
            // Expected
            {1, 0, 1, 0, 0, 0});
    }

    SECTION("Checking greater") {
        test_broadcasted_elemwise_op<float, Greater, std::int8_t>(
            {-1, 2, 0, 4, 5, 6}, Shape({2, 3, 1}),
            {0}, Shape(),
            std::greater<>(),
            // Expected
            {0, 1, 0, 1, 1, 1},
            false);
    }

    SECTION("Checking greater_equal") {
        test_broadcasted_elemwise_op<float, GreaterEqual, std::int8_t>(
            {-1, 2, 0, 4, 5, 6}, Shape({2, 3, 1}),
            {0}, Shape(),
            std::greater_equal<>(),
            // Expected
            {0, 1, 1, 1, 1, 1},
            false);
    }

    SECTION("Checking less") {
        test_broadcasted_elemwise_op<float, Less, std::int8_t>(
            {-1, 2, 0, 4, 5, 6}, Shape({2, 3, 1}),
            {0}, Shape(),
            std::less<>(),
            // Expected
            {1, 0, 0, 0, 0, 0},
            false);
    }

    SECTION("Checking less_equal") {
        test_broadcasted_elemwise_op<float, LessEqual, std::int8_t>(
            {-1, 2, 0, 4, 5, 6}, Shape({2, 3, 1}),
            {0}, Shape(),
            std::less_equal<>(),
            // Expected
            {1, 0, 1, 0, 0, 0},
            false);
    }
}


TEST_CASE("In-place update operations") {
    SECTION("update_add") {
        auto weights = Constant::tensor<float>({0, 1, 2, 3, 4, 5}, {2, 3});
        auto updates = Constant::tensor<float>({0, 1, 2, 3, 4, 5}, {2, 3});
        auto context = Context::make_for_device(0);
        auto update_op = F<UpdateAdd>(weights, updates);
        INFO("Checking how UpdateAdd outputs incremented value");
        evaluate_and_check<float>(update_op, {0, 2, 4, 6, 8, 10}, {2, 3},
                                  context);
        INFO("Checking that the value remains in the original variable");
        evaluate_and_check<float>(weights, {0, 2, 4, 6, 8, 10}, {2, 3},
                                  context);
        INFO("Second update should increment the value further");
        evaluate_and_check<float>(update_op, {0, 3, 6, 9, 12, 15}, {2, 3},
                                  context);
        INFO("Again, the incremented value should remain in the variable");
        evaluate_and_check<float>(weights, {0, 3, 6, 9, 12, 15}, {2, 3},
                                  context);
    }

    SECTION("update_sub") {
        auto weights = Constant::tensor<float>({0, 1, 2, 3, 4, 5}, {2, 3});
        auto updates = Constant::tensor<float>({0, 1, 2, 3, 4, 5}, {2, 3});
        auto context = Context::make_for_device(0);
        auto update_op = F<UpdateSub>(weights, updates);
        evaluate_and_check<float>(update_op, {0, 0, 0, 0, 0, 0}, {2, 3},
                                  context);
        evaluate_and_check<float>(weights, {0, 0, 0, 0, 0, 0}, {2, 3},
                                  context);
        evaluate_and_check<float>(update_op, {0, -1, -2, -3, -4, -5}, {2, 3},
                                  context);
        evaluate_and_check<float>(weights, {0, -1, -2, -3, -4, -5}, {2, 3},
                                  context);
    }
}


TEST_CASE("Choosing common type for operation") {
    REQUIRE(choose_common_array_type(ArrayType::float32, ArrayType::float64)
            == ArrayType::float64);
    REQUIRE(choose_common_array_type(ArrayType::float32, ArrayType::int32)
            == ArrayType::float32);
    REQUIRE(choose_common_array_type(ArrayType::int8, ArrayType::int64)
            == ArrayType::int64);
    REQUIRE(choose_common_array_type(ArrayType::int16, ArrayType::float16)
            == ArrayType::float16);
}



TEST_CASE("Broadcast operations on mixed types") {
    SECTION("Mixed Plus") {
        auto value1 = Constant::tensor<float>({0, 1, 2, 3}, Shape({4}));
        auto value2 = Constant::tensor<int>({0, 1, 2, 3}, Shape({4}));
        auto output = F<Plus>(value1, value2);
        REQUIRE(output->dtype() == ArrayType::float32);
        evaluate_and_check<float>(output, {0, 2, 4, 6}, Shape({4}));
    }
}


TEST_CASE("Loss functions") {
    SECTION("Test of binary cross-entropy") {
        auto nn_outputs = Variable::make("outputs", {5} , ArrayType::float32);
        auto labels = Variable::make("labels", {5}, ArrayType::float32);
        auto output = F<BinaryCrossEntropy>(nn_outputs, labels);
        auto context = Context::make_for_device(0);
        context->init<float>(
            nn_outputs,
            {0.62671396f, 0.53361464f, 0.5972166f, 0.68630082f, 0.33517416f},
            nn_outputs->shape());
        context->init<float>(
            labels,
            {1.0f, 0.0f, 1.0f, 0.0f, 0.0f},
            labels->shape());
        evaluate_and_check<float>(
            output,
            {0.46726505f, 0.76274304f, 0.51547541f, 1.15932078f, 0.40823016f},
            labels->shape(),
            context);
        verify_derivatives<float>(context, {nn_outputs, labels}, output, 1e-2);
    }

}
