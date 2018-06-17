#define CATCH_CONFIG_MAIN

#include <vector>
#include <numeric>
#include <cstdint>

#include <fmt/format.h>

#include "avalanche/testing_tools.h"


TEST_CASE("Broadcast operations on mixed types") {
    using namespace avalanche;

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
}
