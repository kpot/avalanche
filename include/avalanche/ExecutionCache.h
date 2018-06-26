#ifndef AVALANCHE_EXECUTIONCACHE_H
#define AVALANCHE_EXECUTIONCACHE_H

#include <map>

#include "avalanche/BaseNode.h"

namespace avalanche {

struct CachedItem {
    MultiArrayRef data;
    std::size_t num_descendants;
    int reuse_counter;
    std::vector<NodeId> expected_consumers; // Useful for debugging
    bool was_cached_during_the_run;         // Useful for debugging
};


using CachedItemsMap = std::map<NodeId, CachedItem>;
using NodeValueMap = std::map<NodeRef, MultiArrayRef>;

class ExecutionCache : private CachedItemsMap {
public:
    explicit ExecutionCache(DeviceIndex device_idx);
    explicit ExecutionCache(BufferPoolRef buffer_pool);
    bool is_cached(const NodeId node_id) const;
    void zero_reuse_counters();
    void put(const NodeId node_id, const MultiArrayRef &array,
             bool call_from_eval = true);
    void put_all(const NodeValueMap &nodes_and_values);
    bool decrease_counter(const NodeId node_id);
    void set_node_params(NodeId node_id,
                         std::size_t num_descendants,
                         int reuse_counter,
                         const std::vector<NodeRef> &expected_consumers = {});
    bool get(const NodeId node_id, MultiArrayRef &result);
    // For testing purposes
    bool get_info(NodeId node_id, CachedItem &info) const;
    // For testing
    std::size_t size() { return CachedItemsMap::size(); }


private:
    BufferPoolRef _buffer_pool;

    void check_multi_array_compatibility(const MultiArrayRef &array);
};

} // namespace


#endif //AVALANCHE_EXECUTIONCACHE_H
