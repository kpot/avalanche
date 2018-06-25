#define CATCH_CONFIG_MAIN

#include "avalanche/testing_tools.h"

using namespace avalanche;

TEST_CASE("Checking broadcasting shapes") {
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

    SECTION("Not fully defined shapes #1") {
        Shape shape1({-1, 5}), shape2({2, 3, 5});
        auto aligned_shapes = Shape::align_for_broadcasting(shape1, shape2);
        REQUIRE(aligned_shapes[0] == Shape({1, -1, 5}));
        REQUIRE(aligned_shapes[1] == Shape({2, 3, 5}));
        REQUIRE(aligned_shapes[2].agrees_with(Shape({2, -1, 5})));
    }

    SECTION("Not fully defined shapes #2") {
        Shape shape1({-1, -1}), shape2({2, 3, 5});
        auto aligned_shapes = Shape::align_for_broadcasting(shape1, shape2);
        REQUIRE(aligned_shapes[0] == Shape({1, -1, -1}));
        REQUIRE(aligned_shapes[1] == Shape({2, 3, 5}));
        REQUIRE(aligned_shapes[2].agrees_with(Shape({2, -1, -1})));
    }

    SECTION("Not fully defined shapes #3") {
        Shape shape1({-1, 5}), shape2({2, -1, -1});
        auto aligned_shapes = Shape::align_for_broadcasting(shape1, shape2);
        REQUIRE(aligned_shapes[0].dims() == Shape({1, -1, 5}).dims());
        REQUIRE(aligned_shapes[1] == Shape({2, -1, -1}));
        REQUIRE(aligned_shapes[2].agrees_with(Shape({2, -1, -1})));
    }

    SECTION("Not fully defined shapes #4") {
        Shape shape1({2, 5}), shape2({2, -1, 1});
        auto aligned_shapes = Shape::align_for_broadcasting(shape1, shape2);
        REQUIRE(aligned_shapes[0].dims() == Shape({1, 2, 5}).dims());
        REQUIRE(aligned_shapes[1] == Shape({2, -1, 1}));
        REQUIRE(aligned_shapes[2].agrees_with(Shape({2, -1, 5})));
    }
}


TEST_CASE("Inferring shapes for element-wise operations") {
    REQUIRE(
        ElemWiseBinaryOp::infer_elemwise_shape(
            {1, 2, 3}, {UnknownDim, 2, UnknownDim})
        == Shape({1, 2, 3}));
    REQUIRE(
        ElemWiseBinaryOp::infer_elemwise_shape(
            {UnknownDim, 2, UnknownDim}, {5, 2, UnknownDim})
        == Shape({5, 2, UnknownDim}));
    REQUIRE_THROWS_WITH(
        [&]() {
            ElemWiseBinaryOp::infer_elemwise_shape(
                {UnknownDim, 8, UnknownDim}, {5, 2, UnknownDim});
        }(),
        Catch::Contains("operation"));
}

TEST_CASE("Concatenation") {
    SECTION("1-D case") {
        auto value1 = Constant::tensor<float>({0, 1, 2, 3}, Shape({4}));
        auto value2 = Constant::tensor<float>({5, 6, 7, 8, 9}, Shape({5}));
        auto output = Concatenate::make({value1, value2}, -1);
        REQUIRE(output->dtype() == ArrayType::float32);
        evaluate_and_check<float>(
            output, {0, 1, 2, 3, 5, 6, 7, 8, 9}, Shape({9}));
    }

    SECTION("2-D case, concatenate over the last axis") {
        auto value1 = Constant::tensor<float>(
            {0, 1, 2, 3, 4, 5}, Shape({2, 3}));
        auto value2 = Constant::tensor<float>(
            {6, 7, 8, 9}, Shape({2, 2}));
        auto output = Concatenate::make({value1, value2}, -1);
        REQUIRE(output->dtype() == ArrayType::float32);
        evaluate_and_check<float>(
            output, {0, 1, 2, 6, 7, 3, 4, 5, 8, 9}, Shape({2, 5}));
    }

    SECTION("2-D case, concatenate over the first axis") {
        auto value1 = Constant::tensor<float>(
            {0, 1, 2, 3, 4, 5, 6, 7, 8}, Shape({3, 3}));
        auto value2 = Constant::tensor<float>(
            {9, 10, 11, 12, 13, 14}, Shape({2, 3}));
        auto output = Concatenate::make({value1, value2}, 0);
        REQUIRE(output->dtype() == ArrayType::float32);
        evaluate_and_check<float>(
            output, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
            Shape({5, 3}));
    }

    SECTION("3-D case, concatenate over the middle axis") {
        auto value1 = Constant::tensor<float>(
            {0, 1, 2, 3, 4, 5, 6, 7},Shape({2, 2, 2}));
        auto value2 = Constant::tensor<float>(
            {9, 10, 11, 12}, Shape({2, 1, 2}));
        auto output = Concatenate::make({value1, value2}, 1);
        REQUIRE(output->dtype() == ArrayType::float32);
        evaluate_and_check<float>(
            output, {0, 1, 2, 3, 9, 10, 4, 5, 6, 7, 11, 12}, Shape({2, 3, 2}));
    }

    SECTION("Back-propagation through concatenation") {
        auto inputs = Variable::make("inputs", {3, 3}, ArrayType::float32);
        auto weights = Variable::make("weights", {3, 3}, ArrayType::float32);
        auto output = Concatenate::make({inputs, weights});
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
}



TEST_CASE("Slicing") {
    SECTION("Continuous slicing within the first dimension, to check offsets") {
        // This type of slicing should in theory reuse the same buffer,
        // creating several MultiArray instances with different offsets
        // and shapes, all sharing this original buffer.
        auto value = Constant::tensor<float>(
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            Shape({4, 2, 2}));
        auto output1 = FU<SliceAxis>(value, 0, 1, 3);
        auto output2 = FU<SliceAxis>(output1, 0, 1, 2);
        REQUIRE(output1->dtype() == ArrayType::float32);
        REQUIRE(output1->shape() == Shape({3, 2, 2}));
        REQUIRE(output2->dtype() == ArrayType::float32);
        REQUIRE(output2->shape() == Shape({2, 2, 2}));
        Executor executor(Context::make_for_device(0), {output1, output2, value});
        auto results = executor.run();
        REQUIRE(results[0]->buffer_offset() == 4);
        REQUIRE(results[0]->shape() == Shape({3, 2, 2}));
        REQUIRE(results[1]->buffer_offset() == 8);
        REQUIRE(results[1]->shape() == Shape({2, 2, 2}));
        // both outputs must share the same buffer
        REQUIRE(results[0]->buffer_unsafe() == results[1]->buffer_unsafe());
        // ... and it must be the same buffer returned by the original value
        REQUIRE(results[0]->buffer_unsafe() == results[2]->buffer_unsafe());
        // Checking the exact content
        evaluate_and_check<float>(
            output1, {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            Shape({3, 2, 2}));
        evaluate_and_check<float>(
            output2, {8, 9, 10, 11, 12, 13, 14, 15},
            Shape({2, 2, 2}));
    }


    SECTION("Strided slicing within the dimensions after the first") {
        // This type of slicing should in theory reuse the same buffer,
        // creating several MultiArray instances with different offsets
        // and shapes, all sharing this original buffer.
        auto value = Constant::tensor<float>(
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            Shape({2, 4, 2}));
        auto output = FU<SliceAxis>(value, 1, 1, 3);
        REQUIRE(output->dtype() == ArrayType::float32);
        REQUIRE(output->shape() == Shape({2, 3, 2}));
        Executor executor(Context::make_for_device(0), {output, value});
        auto results = executor.run();
        REQUIRE(results[0]->buffer_offset() == 0);
        REQUIRE(results[0]->shape() == Shape({2, 3, 2}));
        // the output buffer shouldn't be the same as the input buffer
        REQUIRE(results[0]->buffer_unsafe() != results[1]->buffer_unsafe());
        // Checking the exact content
        evaluate_and_check<float>(
            output, {2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15},
            Shape({2, 3, 2}));
    }

    SECTION("Strided slicing within a continuous slice") {
        // This test verifies the accuracy of a strided slice when the source
        // buffer already uses some offset
        auto value = Constant::tensor<float>(
            {0, 1,    2, 3,     4, 5,     6, 7,
             8, 9,    10, 11,   12, 13,   14, 15,
             16, 17,  18, 19,   20, 21,   22, 23},
            Shape({3, 4, 2}));
        auto output_with_offset = FU<SliceAxis>(value, 0, 1, 2);
        auto output = FU<SliceAxis>(output_with_offset, 1, 1, 3);
        REQUIRE(output_with_offset->shape() == Shape({2, 4, 2}));
        REQUIRE(output->shape() == Shape({2, 3, 2}));
        Executor executor(Context::make_for_device(0), {output_with_offset, output});
        auto results = executor.run();
        // First result should be just the same buffer as value, with an offset
        REQUIRE(results[0]->buffer_offset() == 8);
        REQUIRE(results[0]->shape() == Shape({2, 4, 2}));
        // Second must be a new buffer
        REQUIRE(results[1]->buffer_unsafe() != results[0]->buffer_unsafe());
        REQUIRE(results[1]->buffer_offset() == 0);
        REQUIRE(results[1]->shape().dims() == Shape({2, 3, 2}).dims());
        // Checking the exact content
        evaluate_and_check<float>(
            output_with_offset,
            {8, 9,    10, 11,   12, 13,   14, 15,
             16, 17,  18, 19,   20, 21,   22, 23},
            Shape({2, 4, 2}));
        evaluate_and_check<float>(
            output,
            {10, 11,   12, 13,   14, 15,
             18, 19,   20, 21,   22, 23},
            Shape({2, 3, 2}));
    }


    SECTION("Slicing with dropping dimensions == 1") {
        // This test verifies the accuracy of a strided slice when the source
        // buffer already uses some offset
        auto value = Constant::tensor<float>(
            {0, 1,    2, 3,     4, 5,     6, 7,
             8, 9,    10, 11,   12, 13,   14, 15,
             16, 17,  18, 19,   20, 21,   22, 23},
            Shape({3, 4, 2}));
        auto output = FU<SliceAxis>(value, 1, 1, 1, false);
        REQUIRE(output->shape() == Shape({3, 2}));
        Executor executor(Context::make_for_device(0), {output});
        auto results = executor.run();
        // First result should be just the same buffer as value, with an offset
        REQUIRE(results[0]->buffer_offset() == 0);
        REQUIRE(results[0]->shape() == Shape({3, 2}));
        // Checking the exact content
        evaluate_and_check<float>(
            output,
            {2, 3,
             10, 11,
             18, 19},
            Shape({3, 2}));
    }

    SECTION("Back-propagation through slicing") {
        auto value = Variable::make("inputs", {3, 4, 2}, ArrayType::float32);
        auto context = Context::make_for_device(0);
        context->init<float>(
            value,
            {0, 1,    2, 3,     4, 5,     6, 7,
             8, 9,    10, 11,   12, 13,   14, 15,
             16, 17,  18, 19,   20, 21,   22, 23},
            Shape({3, 4, 2}));
        auto output_with_offset = FU<SliceAxis>(value, 0, 1, 2);
        verify_derivatives<float>(context, {value}, output_with_offset, 0.05);
        auto output = FU<SliceAxis>(output_with_offset, 1, 1, 3);
        verify_derivatives<float>(context, {value}, output, 0.05);
    }


    SECTION("Back-propagation through slicing with dropped dimensions") {
        auto value = Variable::make("inputs", {3, 4, 2}, ArrayType::float32);
        auto context = Context::make_for_device(0);
        context->init<float>(
            value,
            {0, 1,    2, 3,     4, 5,     6, 7,
             8, 9,    10, 11,   12, 13,   14, 15,
             16, 17,  18, 19,   20, 21,   22, 23},
            Shape({3, 4, 2}));
        auto output = FU<SliceAxis>(value, 1, 1, 1, false);
        REQUIRE(output->shape().dims() == Shape({3, 2}).dims());
        INFO("Checking forward propagation");
        evaluate_and_check<float>(
            output,
            {2, 3,
             10, 11,
             18, 19},
            Shape({3, 2}),
            context);
        INFO("Checking backward propagation");
        verify_derivatives<float>(context, {value}, output, 0.05);
    }
}


TEST_CASE("Tiling") {
    SECTION("Tiling forward") {
        INFO("Tiling across across one dimension")
        auto value = Constant::tensor<float>(
            {0, 1, 2, 3, 4, 5},
            Shape({2, 3}));
        auto output = FU<Tile>(value, std::vector<ShapeDim>({1, 2}));
        evaluate_and_check<float>(
            output, {0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5}, Shape({2, 6}));

        INFO("Tiling across across two dimensions")
        auto output2 = FU<Tile>(value, std::vector<ShapeDim>({3, 2}));
        evaluate_and_check<float>(
            output2,
            {
                0, 1, 2, 0, 1, 2,
                3, 4, 5, 3, 4, 5,
                0, 1, 2, 0, 1, 2,
                3, 4, 5, 3, 4, 5,
                0, 1, 2, 0, 1, 2,
                3, 4, 5, 3, 4, 5,
            },
            Shape({6, 6}));


        INFO("Tiling across across three dimensions")
        auto output3 = FU<Tile>(
            FU<Reshape>(value, Shape({1, 2, 3})),
            std::vector<ShapeDim>({2, 3, 2}));
        evaluate_and_check<float>(
            output3,
            {
                0, 1, 2, 0, 1, 2,
                3, 4, 5, 3, 4, 5,
                0, 1, 2, 0, 1, 2,
                3, 4, 5, 3, 4, 5,
                0, 1, 2, 0, 1, 2,
                3, 4, 5, 3, 4, 5,

                0, 1, 2, 0, 1, 2,
                3, 4, 5, 3, 4, 5,
                0, 1, 2, 0, 1, 2,
                3, 4, 5, 3, 4, 5,
                0, 1, 2, 0, 1, 2,
                3, 4, 5, 3, 4, 5,
            },
            Shape({2, 6, 6}));
    }

    SECTION("Backward tiling") {
        auto value = Constant::tensor<float>(
            {0, 1, 2, 0, 1, 2,
             3, 4, 5, 3, 4, 5},
            Shape({2, 6}));
        auto output = FU<Tile>(value, std::vector<ShapeDim>({1, 2}), false);
        evaluate_and_check<float>(
            output, {0, 2, 4, 6, 8, 10}, Shape({2, 3}));
    }

    SECTION("Back-propagation through tiling") {
        auto value = Variable::make("inputs", {2, 3}, ArrayType::float32);
        auto context = Context::make_for_device(0);
        context->init<float>(
            value,
            {0, 1, 2, 3, 4, 5},
            Shape({2, 3}));
        auto output = FU<Tile>(value, std::vector<ShapeDim>({1, 2}));
        verify_derivatives<float>(context, {value}, output, 0.05);
    }
}


TEST_CASE("Expanding and squeezing dimensions") {
    SECTION("ExpandDims") {
        auto value = Variable::make("value", {2, 6}, ArrayType::float32);
        auto output1 = FU<ExpandDims>(value, 1);
        auto output2 = FU<ExpandDims>(value, -1);
        REQUIRE(output1->shape() == Shape({2, 1, 6}));
        REQUIRE(output2->shape().dims() == Shape({2, 6, 1}).dims());
        auto context = Context::make_for_device(0);
        context->init<float>(
            value,
            {0, 1, 2, 0, 1, 2,
             3, 4, 5, 3, 4, 5},
            Shape({2, 6}));
        evaluate_and_check<float>(
            output1,
            {0, 1, 2, 0, 1, 2,
             3, 4, 5, 3, 4, 5},
            Shape({2, 1, 6}),
            context);
        evaluate_and_check<float>(
            output2,
            {0, 1, 2, 0, 1, 2,
             3, 4, 5, 3, 4, 5},
            Shape({2, 6, 1}),
            context);
        verify_derivatives<float>(context, {value}, output1, 0.05);
        verify_derivatives<float>(context, {value}, output2, 0.05);
    }

    SECTION("Squeeze") {
        auto value = Variable::make("value", {2, 1, 6}, ArrayType::float32);
        auto output = FU<Squeeze>(value, 1);
        REQUIRE(output->shape() == Shape({2, 6}));
        auto context = Context::make_for_device(0);
        context->init<float>(
            value,
            {0, 1, 2, 0, 1, 2,
             3, 4, 5, 3, 4, 5},
            Shape({2, 1, 6}));
        evaluate_and_check<float>(
            output,
            {0, 1, 2, 0, 1, 2,
             3, 4, 5, 3, 4, 5},
            Shape({2, 6}),
            context);
        verify_derivatives<float>(context, {value}, output, 0.05);
    }
}

TEST_CASE("Test stacking nodes") {
    auto val1 = Constant::ones(Shape({4, 3}), ArrayType::float32);
    auto val2 = Constant::zeros(Shape({4, 3}), ArrayType::float32);
    auto output = stack_nodes({val1, val2}, 1);
    evaluate_and_check<float>(
        output,
        {1, 1, 1, 0, 0, 0,
         1, 1, 1, 0, 0, 0,
         1, 1, 1, 0, 0, 0,
         1, 1, 1, 0, 0, 0},
        Shape({4, 2, 3}));
}

TEST_CASE("Reshaping one node to be like another one") {
    auto value1 = Variable::make("value1", {2, 3}, ArrayType::float32);
    auto value2 = Variable::make("value2", {3, 2}, ArrayType::float32);
    auto context = Context::make_for_device(0);
    context->init<float>(
        value1,
        {0, 1, 2, 3, 4, 5},
        Shape({2, 3}));
    context->init<float>(
        value2,
        {10, 11, 12, 13, 14, 15},
        Shape({3, 2}));
    auto output = ReshapeLike::make(value1, value2);
    evaluate_and_check<float>(
        output,
        {0, 1, 2, 3, 4, 5},
        Shape({3, 2}),
        context);
    verify_derivatives<float>(context, {value1, value2}, output, 0.05);
}
