#define CATCH_CONFIG_MAIN

#include <iostream>
#include <chrono>
#include <thread>

#include "catch.hpp"
#include "avalanche/terminal_nodes.h"
#include "avalanche/nodes.h"
#include "avalanche/Executor.h"

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
    using namespace avalanche;
    auto val1 = Constant::scalar(2.0f);
    auto val2 = Constant::fill(Shape({4, 4}), ArrayType::float32, 10);
    auto val3 = val1 + val2;
    std::vector<float> expected(val2->shape().size(), 10.0f + 2.0f);
    Executor executor(Context::make_for_device(0), {val3});
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
