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
    auto &event = buffer->write_from_vector(data, 0);
    REQUIRE(event.get() != nullptr);
//    buffer->wait_until_ready();
    REQUIRE(buffer->byte_size() == sizeof(std::size_t) * data.size());
    std::vector<std::size_t> result;
    std::vector<cl::Event> wait_for_events({event});
    auto reading_is_done = buffer->read_into_vector(result, 0, data_length,
                                                    &wait_for_events);
    reading_is_done.wait();
    REQUIRE(result.size() == data.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        REQUIRE(result[i] == i);
    }
}
