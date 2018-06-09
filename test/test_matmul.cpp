#define CATCH_CONFIG_MAIN

#include <numeric>
#include <functional>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>


#include "catch.hpp"

#include "avalanche/terminal_nodes.h"
#include "avalanche/nodes.h"
#include "avalanche/Shape.h"
#include "avalanche/Executor.h"

TEST_CASE("Testing matrix multiplication") {
    using namespace avalanche;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    long long matrix_side = 2048;
    auto full_size = matrix_side * matrix_side;
    std::vector<float> data_vector(full_size);
    for (int n = 0; n < full_size; ++n) {
        data_vector[n] = dis(gen);
    }
    auto val1 = Constant::tensor<float>(
        data_vector,
        Shape({matrix_side, matrix_side}));
    auto val2 = Constant::tensor<float>(
        data_vector,
        Shape({matrix_side, matrix_side}));

    auto output = F<MatMul>(val1, val2, false, false);
//    auto output = F<Multiply>(val1, val2);
    Executor executor(Context::make_for_device(0), {output});
    auto start_time = std::chrono::steady_clock::now();
    for (int timeit = 0; timeit < 10; ++timeit) {
        auto results = executor.run();
        std::vector<float> cpu_copy;
        results[0]->fetch_data_into(cpu_copy);
    }
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Time elapsed" << diff.count() << " s\n";
}
