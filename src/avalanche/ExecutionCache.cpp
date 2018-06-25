#include "avalanche/ExecutionCache.h"


namespace avalanche {

ExecutionCache::ExecutionCache(BufferPoolRef buffer_pool)
    :CachedItemsMap(),
     _buffer_pool{buffer_pool}{}

ExecutionCache::ExecutionCache(DeviceIndex device_idx)
    :ExecutionCache(
    CLMemoryManager::get_default()->buffer_pool(device_idx)) {}

void ExecutionCache::put(const NodeId node_id, const MultiArrayRef &array) {
    check_multi_array_compatibility(array);
    auto cached = find(node_id);

    // Caching cannot be used for a node for which we don't know the full"
    // list of dependent nodes.
    // Also there's no point in caching values for nodes having only one
    // other node depending from it. Such value can just be used
    // immediately and discarded afterwards.
    if (cached != this->end() && cached->second.num_descendants > 1) {
        cached->second.data = array;
        // Here we assume that one of the descendants has already received
        // newly calculated value, so there's no need to cache it
        // one more time
        cached->second.reuse_counter = cached->second.num_descendants - 1;
    }
}

bool ExecutionCache::get(NodeId node_id, MultiArrayRef &result) {
    auto cached = find(node_id);
    if (cached != this->end() && cached->second.reuse_counter > 0) {
        result = cached->second.data;
        --cached->second.reuse_counter;
        if (cached->second.reuse_counter == 0) {
            cached->second.data.reset();
        }
        at(node_id) = cached->second;
        return true;
    }
    result = nullptr;
    return false;
}

void ExecutionCache::set_node_params(
    NodeId node_id, std::size_t num_descendants, std::size_t reuse_counter) {

    auto cached = find(node_id);
    if (cached != this->end()) {
        cached->second.reuse_counter = reuse_counter;
        cached->second.num_descendants = num_descendants;
        at(node_id) = cached->second;
    } else {
        CachedItem value {
            .data = nullptr,
            .num_descendants = num_descendants,
            .reuse_counter = reuse_counter
        };
        insert({node_id, std::move(value)});
    }
}

void ExecutionCache::zero_reuse_counters() {
    for (auto &item: *this) {
        item.second.reuse_counter = 0;
    }
}

bool ExecutionCache::get_from_cache_no_counter(
        NodeId node_id, MultiArrayRef &result) const {
    auto cached = find(node_id);
    if (cached != this->end() && cached->second.data) {
        result = cached->second.data;
        return true;
    }
    result = nullptr;
    return false;
}

void ExecutionCache::check_multi_array_compatibility(const MultiArrayRef &array) {
    if (array->buffer_unsafe()->pool() != _buffer_pool) {
        throw std::invalid_argument(
            "MultiArray and the Context cannot be linked to different "
            "devices or contexts");
    }
}

bool ExecutionCache::get_info(NodeId node_id, CachedItem &info) const {
    auto cached = find(node_id);
    if (cached != this->end()) {
        info = cached->second;
        return true;
    }
    return false;
}

bool ExecutionCache::is_cached(const NodeId node_id) const {
    auto cached = find(node_id);
    return cached != this->end() && cached->second.data;
}


} // namespace
