#define CATCH_CONFIG_MAIN

#include <vector>
#include <numeric>
#include <cstdint>

#include <fmt/format.h>

#include "avalanche/testing_tools.h"


TEST_CASE("Broadcast operations on mixed types") {
    SECTION("Mixed Plus") {
        using namespace avalanche;
        auto value1 = Constant::tensor<float>({0, 1, 2, 3}, Shape({4}));
        auto value2 = Constant::tensor<int>({0, 1, 2, 3}, Shape({4}));
        auto output = F<Plus>(value1, value2);
        REQUIRE(output->dtype() == ArrayType::float32);
        evaluate_and_check<float>(output, {0, 2, 4, 6}, Shape({4}));
    }
}
