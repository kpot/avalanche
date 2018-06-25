#define CATCH_CONFIG_MAIN

#include <iostream>
#include <chrono>
#include <thread>

#include "catch.hpp"
#include "avalanche/terminal_nodes.h"
#include "avalanche/nodes.h"
#include "avalanche/Executor.h"
#include "avalanche/testing_tools.h"

using namespace avalanche;

template <typename T>
bool approximately_equal(const std::vector<T> &a, const std::vector<T> &b) {
    REQUIRE(a.size() == b.size());
    for (auto ac = a.begin(), bc = b.begin(); ac != a.end(); ++ac, ++bc) {
        if (*ac != Approx(*bc).epsilon(0.01)) {
            REQUIRE(a == b);
            return false;
        }
    }
    return true;
}


TEST_CASE("Construction of the tree") {
    using namespace std::chrono_literals;
    auto context = Context::make_for_device(0);
    auto val1 = Constant::scalar(2.0f);
    auto val2 = Constant::fill(Shape({4, 4}), ArrayType::float32, 10);
    // First we check the inputs
    Executor value_executor(context, {val1, val2});
    auto value_results = value_executor.run();
    std::vector<float> value_copy;
    value_results[0]->fetch_data_into(value_copy);
    REQUIRE(value_copy == std::vector<float>({2.0f}));
    value_results[1]->fetch_data_into(value_copy);
    REQUIRE(value_copy == std::vector<float>(
        { 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f,
          10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f }));

    auto val3 = val1 + val2;
    std::vector<float> expected(val2->shape().size(), 10.0f + 2.0f);
    Executor executor(context, {val3});
    for (int i = 0; i < 3; ++i) {
        auto results = executor.run();
        std::vector<float> cpu_copy;
        results[0]->fetch_data_into(cpu_copy);
        REQUIRE(results[0]->shape() == val2->shape());
        REQUIRE(cpu_copy.size() == val2->shape().size());
        REQUIRE(cpu_copy == expected);

        REQUIRE_THROWS_WITH(
            [&]() {
                std::vector<double> cpu_copy2;
                results[0]->fetch_data_into(cpu_copy2);
            }(),
            Catch::Contains("incompatible"));
    }
}

TEST_CASE("Variable initializers") {
    using namespace avalanche;
    SECTION("Initialization with some lazy initializer") {
        std::vector<float> initial_value({1.0, 2.0, 3.0});
        auto initializer = value_initializer(initial_value, Shape({1, 3}));
        auto var1 = Variable::make("variable", {1, 3}, ArrayType::float32,
                                   initializer);
        auto context = Context::make_for_device(0);
        Executor executor(context, {var1});
        auto results = executor.run();
        std::vector<float> cpu_copy;
        results[0]->fetch_data_into(cpu_copy);
        REQUIRE(cpu_copy == initial_value);
    }

    SECTION("Initialization from some other node") {
        auto rng = UniformRandom::make(Shape({1, 3}), -1, 1, ArrayType::float32, 0);
        auto var1 = Variable::make_from_node("variable", rng);
        REQUIRE(var1->shape() == rng->shape());
        REQUIRE(var1->dtype() == rng->dtype());
        auto context = Context::make_for_device(0);
        Executor executor(context, {var1});
        auto results = executor.run();
        std::vector<float> cpu_copy;
        results[0]->fetch_data_into(cpu_copy);
        REQUIRE(cpu_copy.size() == rng->shape().size());
    }
}


TEST_CASE("Checking the agreements between shapes") {
    Shape required_shape({1, 2, -1, -1, 5});
    REQUIRE_FALSE(required_shape.is_complete());
    REQUIRE_FALSE(required_shape.is_scalar());
    REQUIRE(required_shape.to_string() == "Shape(1, 2, ?, ?, 5)");

    Shape candidate1({1, 2});
    REQUIRE(candidate1.is_complete());
    REQUIRE_FALSE(candidate1.is_scalar());
    REQUIRE_FALSE(candidate1.agrees_with(required_shape));


    Shape candidate2;
    REQUIRE(candidate2.is_complete());
    REQUIRE(candidate2.is_scalar());
    REQUIRE_FALSE(candidate2.agrees_with(required_shape));

    Shape candidate3({1, 2, 18, 9, 5});
    REQUIRE(candidate3.is_complete());
    REQUIRE_FALSE(candidate3.is_scalar());
    REQUIRE(candidate3.agrees_with(required_shape));
}


TEST_CASE("Initializing variables with incompletely defined shapes") {
    auto var1 = Variable::make("check", {-1, 5}, ArrayType::float32, Initializer{});
    auto context = Context::make_for_device(0);
    std::vector<float> data(10);
    context->init(var1, data, Shape({2, 5}));
    REQUIRE_FALSE(var1->shape().is_complete());
    MultiArrayRef value;
    REQUIRE(context->get(var1->id, value));
    REQUIRE(value->shape().is_complete());
    REQUIRE(value->shape().agrees_with(var1->shape()));
}

TEST_CASE("Conditional evaluation") {
    // One of two variables should get incremented at a time depending
    // on a state of a third boolean variable, determining which one
    // of the two it will be
    auto var1 = Variable::make("var1", {}, ArrayType::float32);
    auto var2 = Variable::make("var2", {}, ArrayType::float32);
    auto condition = Variable::make("condition", {}, ArrayType::int8);
    auto context = Context::make_for_device(0);
    context->init<float>(var1, {0});
    context->init<float>(var2, {0});
    auto one = Constant::scalar<float>(1);
    auto update1 = F<UpdateAdd>(var1, one);
    auto update2 = F<UpdateAdd>(var2, one);
    auto output = Cond::make(condition, update1, update2);
    INFO("Check chat only var2 should be incremented, because condition == 0");
    context->init<std::int8_t>(condition, {0});
    evaluate_and_check<float>(output, {1}, Shape(), context);
    evaluate_and_check<float>(var1, {0}, Shape(), context);
    evaluate_and_check<float>(var2, {1}, Shape(), context);
    INFO("Now only var1 should be incremented, because condition == 1");
    context->init<std::int8_t>(condition, {1});
    evaluate_and_check<float>(output, {1}, Shape(), context);
    evaluate_and_check<float>(var1, {1}, Shape(), context);
    evaluate_and_check<float>(var2, {1}, Shape(), context);
    INFO("Again only var1 should be incremented, "
         "because condition is still == 1");
    evaluate_and_check<float>(output, {2}, Shape(), context);
    evaluate_and_check<float>(var1, {2}, Shape(), context);
    evaluate_and_check<float>(var2, {1}, Shape(), context);
}
