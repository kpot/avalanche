#ifndef AVALANCHE_EXECUTOR_H
#define AVALANCHE_EXECUTOR_H

#include <vector>

#include "avalanche/Context.h"
#include "avalanche/ExecutionCache.h"
#include "avalanche/BaseNode.h"
#include "avalanche/backprop.h"

namespace avalanche {

class Executor {
public:
    Executor(const ContextRef &context,
             const NodeRefList &result_nodes)
    :_context{context},
     _cache{context->device_pool()},
     _result_nodes{result_nodes}
    {
        if (!context) {
            throw std::invalid_argument("No context!");
        }
        auto consumer_map = build_consumers_map(result_nodes);
        for (auto &item: consumer_map) {
            _cache.set_node_params(item.first->id, item.second.size(), 0);
        }
    }

    std::vector<MultiArrayRef> run() {
        std::vector<MultiArrayRef> results;
        // just in case we zero the counters. Although in theory
        // if the previous run had not failed, all counters must already be 0
        _cache.zero_reuse_counters();
        for (auto &target: _result_nodes) {
            auto result = target->eval(*_context, _cache);
            results.emplace_back(std::move(result));
        }
        _context->device_pool()->cl_queue().flush();
        for (auto &result: results) {
            result->wait_until_ready();
        }
        return results;
    }

    const NodeRefList& result_nodes() { return _result_nodes; }


private:
    ContextRef _context;
    ExecutionCache _cache;
    const NodeRefList _result_nodes;
};

}


#endif //AVALANCHE_EXECUTOR_H
