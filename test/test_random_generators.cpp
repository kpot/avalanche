#define CATCH_CONFIG_MAIN

#include <cstdint>

#include "catch.hpp"

#include "avalanche/terminal_nodes.h"
#include "avalanche/nodes.h"
#include "avalanche/Shape.h"
#include "avalanche/Executor.h"
#include "avalanche/Context.h"
#include "avalanche/random_nodes.h"


TEST_CASE("Checking uniform random generator on float numbers") {
    using namespace avalanche;
    float min_value = -1, max_value = 1;
    auto node = UniformRandom::make({3, 3}, min_value, max_value, ArrayType::float32, 0);
    auto context = Context::make_for_device(0);
    Executor executor(context, {node});
    auto outputs = executor.run();
    std::vector<float> first_run_data;
    outputs[0]->fetch_data_into(first_run_data);
    REQUIRE(outputs[0]->shape() == node->shape());
    REQUIRE(first_run_data.size() == node->shape().size());
    for (float item: first_run_data) {
        REQUIRE(item != 0);
        REQUIRE(item >= min_value);
        REQUIRE(item <= max_value);
    }
    // Second run
    outputs = executor.run();
    std::vector<float> second_run_data;
    outputs[0]->fetch_data_into(second_run_data);
    REQUIRE(first_run_data.size() == second_run_data.size());
    for (std::size_t i = 0; i < first_run_data.size(); ++i) {
        auto item = second_run_data[0];
        REQUIRE(item != first_run_data[i]);
        REQUIRE(item != 0);
        REQUIRE(item >= min_value);
        REQUIRE(item < max_value);
    }
}


TEST_CASE("Checking uniform random generator on integers") {
    using namespace avalanche;
    double min_value = -100, max_value = 100;
    auto node = UniformRandom::make({10000}, min_value, max_value,
                                    ArrayType::int32, 0);
    auto context = Context::make_for_device(0);
    Executor executor(context, {node});
    auto outputs = executor.run();
    std::vector<std::int32_t> first_run_data;
    outputs[0]->fetch_data_into(first_run_data);
    REQUIRE(outputs[0]->shape() == node->shape());
    REQUIRE(first_run_data.size() == node->shape().size());
    int positive = 0, negative = 0, exact_min = 0;
    for (std::int32_t item: first_run_data) {
        REQUIRE(item >= min_value);
        REQUIRE(item < max_value);
        if (item < 0) negative++;
        if (item >= 0) positive++;
        if (item == min_value) exact_min++;
    }
    // Such tests can fail in theory due to the randomness of the sample
    // although in practice this should never happen
    REQUIRE(negative > 0);
    REQUIRE(positive > 0);
    REQUIRE(exact_min > 0);
}
