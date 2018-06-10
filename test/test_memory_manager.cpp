#define CATCH_CONFIG_MAIN

#include <memory>

#include "catch.hpp"


#include "avalanche/MultiArray.h"
#include "avalanche/BaseNode.h"
#include "avalanche/Context.h"
#include "avalanche/ExecutionCache.h"


TEST_CASE("Test memory manager") {
    auto manager = avalanche::CLMemoryManager();
    manager.init_for_all_gpus();
    REQUIRE(avalanche::CLBufferPool::find_mem_slot(1024) == 10);
    REQUIRE(avalanche::CLBufferPool::find_mem_slot(1025) == 11);
    REQUIRE(avalanche::CLBufferPool::find_mem_slot(2048) == 11);
    REQUIRE(manager.num_devices() > 0);
    auto pool = manager.buffer_pool(0);
    REQUIRE(pool->num_available_blocks() == 0);
    {
        // We reserve memory in large blocks the size of power of 2,
        // to make the reuse of it easier
        auto buffer = pool->reserve_buffer(1024);
        REQUIRE(pool->num_available_blocks() == 0);
        REQUIRE(buffer->byte_size() == 1024);
        REQUIRE(buffer->capacity() == 1024);
    }
    REQUIRE(pool->num_available_blocks() == 1);
    {
        // A bit larger buffer and the slot/capacity will be differen
        auto buffer2 = manager.buffer_pool(0)->reserve_buffer(1025);
        REQUIRE(pool->num_available_blocks() == 1);
        REQUIRE(buffer2->byte_size() == 1025);
        REQUIRE(buffer2->capacity() == 2048);
    }
    std::cout << "now re-check\n";
    REQUIRE(pool->num_available_blocks() == 2);
    // Now if we reserve a new buffer of a size smaller than the first one,
    // at the same time disposing of the first buffer, the first buffer
    // should actually be reused
    {
        auto buffer = pool->reserve_buffer(1000);
        REQUIRE(buffer->byte_size() == 1000);
        REQUIRE(buffer->capacity() == 1024);
    }
    REQUIRE(pool->num_available_blocks() == 2);
}

TEST_CASE("Testing Shape class") {
    avalanche::Shape shape({1, 3, 5});
    REQUIRE(shape.rank() == 3);
    REQUIRE(shape.dim(1) == 3);
    REQUIRE(shape.size() == 15);
    REQUIRE(shape.dim(-1) == 5);
    REQUIRE(shape.dim(-2) == 3);
    avalanche::Shape scalar;
    REQUIRE(scalar.rank() == 0);
    REQUIRE(scalar.size() == 1);
}


TEST_CASE("Checking MultiArray") {
    auto dtype = avalanche::ArrayType::float32;
    auto array = avalanche::MultiArray::make(0, {1, 2, 3}, dtype);
    auto array2 = avalanche::MultiArray::make(0, avalanche::Shape({1, 2, 3}),
                                              dtype);
    auto reshaped_array = array->reshape({2, 3});
    REQUIRE(reshaped_array->size() == array->size());
    REQUIRE(reshaped_array->shape() == avalanche::Shape({2, 3}));
    REQUIRE(array->reshape({-1})->shape() == avalanche::Shape({6}));
    REQUIRE(array->reshape({2, -1})->shape() == avalanche::Shape({2, 3}));
    REQUIRE(array->reshape({1, -1, 1, 2})->shape()
            == avalanche::Shape({1, 3, 1, 2}));
    REQUIRE(array->reshape({1, -1, 1, 2})->shape()
            != avalanche::Shape({4, 3, 1, 2}));
}

TEST_CASE("Checking context storage") {
    auto dtype = avalanche::ArrayType::float32;
    avalanche::NodeId node1_id = 0, node2_id = 10;
    auto context = avalanche::Context::make_for_device(0);
    auto array_ref = avalanche::MultiArray::make(0, {1, 2, 3}, dtype);
    context->init(node1_id, array_ref);
    // Checks that the cache is properly initialized
    avalanche::MultiArrayRef context_array;
    REQUIRE(context->get(node1_id, context_array));
    REQUIRE(context_array == array_ref);
}

TEST_CASE("Checking ExecutionCache") {
    auto dtype = avalanche::ArrayType::float32;
    avalanche::NodeId node1_id = 0, node2_id = 10;
    avalanche::ExecutionCache cache(0);
    cache.set_node_params(node1_id, 2, 0); // fake descendants to activate cache
    auto array_ref = avalanche::MultiArray::make(0, {1, 2, 3}, dtype);
    cache.put(node1_id, array_ref);
    avalanche::CachedItem cached;
    REQUIRE(cache.get_info(node1_id, cached));
    REQUIRE(cached.num_descendants == 2);
    REQUIRE(cached.reuse_counter == 1);
    REQUIRE(cached.data == array_ref);
    // Setting a node's params shouldn't change the cached value
    cache.set_node_params(node1_id, 15, 10);
    REQUIRE(cache.get_info(node1_id, cached));
    REQUIRE(cached.num_descendants == 15);
    REQUIRE(cached.reuse_counter == 10);
    REQUIRE(cached.data == array_ref);
    // And if we add new for a not previously cached node,
    // the array must remain empty
    cache.set_node_params(node2_id, 3, 1);
    REQUIRE(cache.size() == 2);
    REQUIRE(cache.get_info(node2_id, cached));
    REQUIRE(cached.num_descendants == 3);
    REQUIRE(cached.reuse_counter == 1);
    REQUIRE(cached.data == nullptr);
    // But we can cache an array in the same position, incrementing the counter
    // though not changing the num_descendants
    cache.put(node2_id, array_ref);
    REQUIRE(cache.size() == 2);
    REQUIRE(cache.get_info(node2_id, cached));
    REQUIRE(cached.num_descendants == 3);
    REQUIRE(cached.reuse_counter == 2);  // num_descendants - 1
    REQUIRE(cached.data == array_ref);
    // Now we check how cache counting and memory releasing works
    avalanche::MultiArrayRef fetched_ref;
    REQUIRE(fetched_ref == nullptr);
    REQUIRE(cache.get(node2_id, fetched_ref));
    REQUIRE(fetched_ref == array_ref);
    REQUIRE(cache.get_info(node2_id, cached));
    REQUIRE(cached.num_descendants == 3);
    REQUIRE(cached.reuse_counter == 1);  // decreased by one
    REQUIRE(cached.data == array_ref);
    // ... now we do that again, to bring the counter to zero
    REQUIRE(cache.get(node2_id, fetched_ref));
    REQUIRE(fetched_ref == array_ref);
    REQUIRE(cache.get_info(node2_id, cached));
    REQUIRE(cached.num_descendants == 3);
    REQUIRE(cached.reuse_counter == 0);  // decreased by one
    REQUIRE(cached.data == nullptr);  // here we release the data
    // ... any further attempts to get the cached value should fail
    REQUIRE_FALSE(cache.get(node2_id, fetched_ref));
    REQUIRE(fetched_ref == nullptr);
    REQUIRE(cache.get_info(node2_id, cached));
    REQUIRE(cached.num_descendants == 3);
    REQUIRE(cached.reuse_counter == 0);  // decreased by one
    REQUIRE(cached.data == nullptr);  // here we release the data
    // Now we check how well we're able to clear the counters
    cache.set_node_params(node1_id, 3, 1);
    cache.set_node_params(node2_id, 3, 1);
    cache.zero_reuse_counters();
    REQUIRE(cache.get_info(node1_id, cached));
    REQUIRE(cached.reuse_counter == 0);
    REQUIRE(cache.get_info(node2_id, cached));
    REQUIRE(cached.reuse_counter == 0);
}