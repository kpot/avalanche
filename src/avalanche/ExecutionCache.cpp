#include "avalanche/ExecutionCache.h"

#include <fmt/format.h>


namespace avalanche {

ExecutionCache::ExecutionCache(BufferPoolRef buffer_pool)
    :CachedItemsMap(),
     _buffer_pool{buffer_pool}{}

ExecutionCache::ExecutionCache(DeviceIndex device_idx)
    :ExecutionCache(
    CLMemoryManager::get_default()->buffer_pool(device_idx)) {}

void ExecutionCache::put(const NodeId node_id,
                         const MultiArrayRef &array,
                         bool call_from_eval) {
    check_multi_array_compatibility(array);
    auto cached = find(node_id);

    // Caching cannot be used for a node for which we don't know the full"
    // list of dependent nodes.
    // Also if call_from_eval == true there's no point in caching values
    // for nodes having only one other node depending from it.
    // Such value can just be used immediately and discarded afterwards.
    if (cached != this->end()
            && (!call_from_eval || cached->second.num_descendants > 1)) {
        cached->second.data = array;

        // if call_from_eval == true we assume it'a a typical case when a node
        // just evaluated itself and stores its value, returning it right after
        // to a consumer node.
        // Which means one of the descendants has already received
        // the value, so there's no need to cache it for one more call to ::get
        cached->second.reuse_counter += (
            // We use increment here because reuse_couter can be made negative
            // if one of the consumers (like cond) will not be evaluating
            // the node
            cached->second.num_descendants - (call_from_eval ? 1 : 0));
        cached->second.was_cached_during_the_run = true;
    }
}

bool ExecutionCache::decrease_counter(const NodeId node_id) {
    auto cached = find(node_id);

    // Caching cannot be used for a node for which we don't know the full"
    // list of dependent nodes.
    // Also if call_from_eval == true there's no point in caching values
    // for nodes having only one other node depending from it.
    // Such value can just be used immediately and discarded afterwards.
    if (cached != this->end()) {
        cached->second.reuse_counter--;
        if (cached->second.reuse_counter == 0) {
            cached->second.data.reset();
        }
        return true;
    }
    return false;
}

bool ExecutionCache::get(const NodeId node_id, MultiArrayRef &result) {
    auto cached = find(node_id);
    if (cached != this->end()) {
        if (cached->second.reuse_counter > 0) {
            result = cached->second.data;
            --cached->second.reuse_counter;
            if (cached->second.reuse_counter == 0) {
                cached->second.data.reset();
            }
            at(node_id) = cached->second;
            return true;
        }
        if (cached->second.was_cached_during_the_run) {
            throw std::runtime_error(
                fmt::format(
                    "Node {} cache has expired (used by all consumers) but "
                    "something is trying to evaluate it again. "
                    "There must be some not declared dependencies within "
                    "the computational tree that must be made explicit.",
                    node_id));
        }
    }
    result = nullptr;
    return false;
}

void ExecutionCache::set_node_params(
        NodeId node_id, std::size_t num_descendants, int reuse_counter,
        const std::vector<NodeRef> &expected_consumers) {
    auto cached = find(node_id);
    if (cached != this->end()) {
        cached->second.reuse_counter = reuse_counter;
        cached->second.num_descendants = num_descendants;
        at(node_id) = cached->second;
    } else {
        std::vector<NodeId> consumer_ids;
        std::transform(expected_consumers.begin(),
                       expected_consumers.end(),
                       std::back_inserter(consumer_ids),
                       [](const NodeRef &node) { return node->id; });
        CachedItem value {
            .data = nullptr,
            .num_descendants = num_descendants,
            .reuse_counter = reuse_counter,
            .expected_consumers = consumer_ids,
            .was_cached_during_the_run = false
        };
        insert({node_id, std::move(value)});
    }
}

void ExecutionCache::zero_reuse_counters() {
    for (auto &item: *this) {
        item.second.reuse_counter = 0;
        item.second.data.reset();
        item.second.was_cached_during_the_run = false;
    }
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

void ExecutionCache::put_all(const NodeValueMap &nodes_and_values) {
    for (const auto &item: nodes_and_values) {
        put(item.first->id, item.second, false);
    }
}


} // namespace
