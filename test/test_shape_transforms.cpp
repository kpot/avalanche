#define CATCH_CONFIG_MAIN

#include "avalanche/testing_tools.h"

TEST_CASE("Checking broadcasting shapes") {
    using namespace avalanche;
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
        REQUIRE(aligned_shapes[2] == Shape({2, -1, 5}));
    }

    SECTION("Not fully defined shapes #2") {
        Shape shape1({-1, -1}), shape2({2, 3, 5});
        auto aligned_shapes = Shape::align_for_broadcasting(shape1, shape2);
        REQUIRE(aligned_shapes[0] == Shape({1, -1, -1}));
        REQUIRE(aligned_shapes[1] == Shape({2, 3, 5}));
        REQUIRE(aligned_shapes[2] == Shape({2, -1, -1}));
    }

    SECTION("Not fully defined shapes #3") {
        Shape shape1({-1, 5}), shape2({2, -1, -1});
        auto aligned_shapes = Shape::align_for_broadcasting(shape1, shape2);
        REQUIRE(aligned_shapes[0].dims() == Shape({1, -1, 5}).dims());
        REQUIRE(aligned_shapes[1] == Shape({2, -1, -1}));
        REQUIRE(aligned_shapes[2] == Shape({2, -1, -1}));
    }

    SECTION("Not fully defined shapes #4") {
        Shape shape1({2, 5}), shape2({2, -1, 1});
        auto aligned_shapes = Shape::align_for_broadcasting(shape1, shape2);
        REQUIRE(aligned_shapes[0].dims() == Shape({1, 2, 5}).dims());
        REQUIRE(aligned_shapes[1] == Shape({2, -1, 1}));
        REQUIRE(aligned_shapes[2] == Shape({2, -1, 5}));
    }
}
