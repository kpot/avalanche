#ifndef AVALANCHE_EXECUTIONCACHE_H
#define AVALANCHE_EXECUTIONCACHE_H

#include <map>

#include "avalanche/BaseNode.h"

namespace avalanche {

struct CachedItem {
    MultiArrayRef data;
    std::size_t num_descendants;
    std::size_t reuse_counter;
};


using CachedItemsMap = std::map<NodeId, CachedItem>;

class ExecutionCache : private CachedItemsMap {
public:
    explicit ExecutionCache(DeviceIndex device_idx);
    explicit ExecutionCache(BufferPoolRef buffer_pool);
    bool get_from_cache_no_counter(NodeId node_id, MultiArrayRef &result) const;
    void zero_reuse_counters();
    void put(const NodeId node_id, const MultiArrayRef &array);
    void set_node_params(NodeId node_id,
                         std::size_t num_descendants,
                         std::size_t reuse_counter);
    bool get(NodeId node_id, MultiArrayRef &result);
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
