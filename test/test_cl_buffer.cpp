#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include "avalanche/CLMemoryManager.h"
#include "avalanche/CLBufferPool.h"
#include "avalanche/CLBuffer.h"


TEST_CASE("Filling/fetching CLBuffer using vectors") {
    using namespace avalanche;
    const std::size_t data_length = 1024;
    std::vector<std::size_t> data(data_length);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = i;
    }
    auto pool = CLMemoryManager::get_default()->buffer_pool(0);
    auto buffer = pool->reserve_buffer_for_vector(data);
    auto &event = buffer->write_from_vector(data);
    REQUIRE(event.get() != nullptr);
//    buffer->wait_until_ready();
    REQUIRE(buffer->byte_size() == sizeof(std::size_t) * data.size());
    std::vector<std::size_t> result;
    std::vector<cl::Event> wait_for_events({event});
    buffer->read_into_vector(result, &wait_for_events);
    REQUIRE(result.size() == data.size());
    buffer->wait_until_ready();
    for (std::size_t i = 0; i < result.size(); ++i) {
        REQUIRE(result[i] == i);
    }
}

// TODO: Check how dependency works
