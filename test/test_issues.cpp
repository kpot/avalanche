#define CATCH_CONFIG_MAIN

#include <vector>

#include "catch.hpp"

#include "avalanche/Shape.h"
#include "avalanche/terminal_nodes.h"
#include "avalanche/Context.h"
#include "avalanche/Executor.h"
#include "avalanche/nodes.h"

namespace av = avalanche;

template <typename T, typename NodeOp>
void test_broadcasted_elemwise_op(const std::vector<T> &source1,
                                  const av::Shape &shape1,
                                  const std::vector<T> &source2,
                                  const av::Shape &shape2,
                                  const std::vector<T> &expected) {
    auto input1 = av::Constant::tensor(source1, shape1);
    auto input2 = av::Constant::tensor(source2, shape2);
    auto output = av::F<NodeOp>(input1, input2);
    av::Executor executor(av::Context::make_for_device(0), {output});
    auto evaluated_output = executor.run();
    std::vector<T> cpu_copy;
    evaluated_output[0]->fetch_data_into(cpu_copy);
    REQUIRE(cpu_copy == expected);
};

TEST_CASE("Broadcasted plus") {
    auto weights = av::Variable::make("weights", {3, 3} , av::ArrayType::float32);
    auto biases = av::Variable::make("biases", {3}, av::ArrayType::float32);
    auto output = av::F<av::Plus>(weights, biases);
    auto context = av::Context::make_for_device(0);
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
    auto grad_table = av::build_back_propagation_graph(output, {weights, biases});
    av::Executor executor(context, {grad_table[weights]});
    executor.run();
}
