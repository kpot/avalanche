#define CATCH_CONFIG_MAIN

#include <vector>
#include <numeric>
#include <cstdint>

#include <fmt/format.h>

#include "avalanche/testing_tools.h"


TEST_CASE("In-place update operations") {
    using namespace avalanche;

    SECTION("update_add") {
        auto weights = Constant::tensor<float>({0, 1, 2, 3, 4, 5}, {2, 3});
        auto updates = Constant::tensor<float>({0, 1, 2, 3, 4, 5}, {2, 3});
        auto context = Context::make_for_device(0);
        auto update_op = F<UpdateAdd>(weights, updates);
        evaluate_and_check<float>(update_op, {0, 2, 4, 6, 8, 10}, {2, 3},
                                  context);
        evaluate_and_check<float>(weights, {0, 2, 4, 6, 8, 10}, {2, 3},
                                  context);
        evaluate_and_check<float>(update_op, {0, 3, 6, 9, 12, 15}, {2, 3},
                                  context);
        evaluate_and_check<float>(weights, {0, 3, 6, 9, 12, 15}, {2, 3},
                                  context);
    }

     SECTION("update_add") {
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
