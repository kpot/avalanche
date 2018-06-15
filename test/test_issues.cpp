#define CATCH_CONFIG_MAIN

#include <vector>
#include <numeric>

#include "avalanche/testing_tools.h"


TEST_CASE("Failed tests") {
    using namespace avalanche;

    SECTION("Full reduce mean") {
        auto inputs = Variable::make("inputs", {3, 3}, ArrayType::float32);
        auto output = FU<ReduceMean>(inputs);
        auto context = Context::make_for_device(0);
        context->init<float>(
            inputs,
            {0.0f, 1.0f, 2.0,
             3.0, 4.0, 5.0,
             6.0, 7.0, 8.0},
            inputs->shape());
        verify_derivatives<float>(context, {inputs}, output, 0.05);
    }


}
