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
        :Executor(context, result_nodes, {}) {}

    Executor(const ContextRef &context,
             const NodeRefList &result_nodes,
             const NodeRefList &updates)
    :_context{context},
     _cache{context->device_pool()},
     _result_nodes{result_nodes},
     _update_nodes{updates}
    {
        if (!context) {
            throw std::invalid_argument("No context!");
        }
        NodeRefList full_node_list;
        std::copy(_result_nodes.begin(), _result_nodes.end(), std::back_inserter(full_node_list));
        std::copy(_update_nodes.begin(), _update_nodes.end(), std::back_inserter(full_node_list));
        auto consumer_map = build_consumers_map(full_node_list);
        for (auto &item: consumer_map) {
            _cache.set_node_params(item.first->id, item.second.size(), 0);
        }
    }

    std::vector<MultiArrayRef> run() {
        std::vector<MultiArrayRef> results;
        std::vector<MultiArrayRef> update_results;
        // just in case we zero the counters. Although in theory
        // if the previous run had not failed, all counters must already be 0
        _cache.zero_reuse_counters();
        for (auto &target: _result_nodes) {
            auto result = target->eval(*_context, _cache);
            results.emplace_back(std::move(result));
        }
        for (auto &target: _update_nodes) {
            auto result = target->eval(*_context, _cache);
            update_results.emplace_back(std::move(result));
        }
        _context->device_pool()->cl_queue().flush();
        for (auto &result: results) {
            result->wait_until_ready();
        }
        for (auto &result: update_results) {
            result->wait_until_ready();
        }
        return results;
    }

    const NodeRefList& result_nodes() { return _result_nodes; }


private:
    ContextRef _context;
    ExecutionCache _cache;
    const NodeRefList _result_nodes;
    const NodeRefList _update_nodes;
};

}


#endif //AVALANCHE_EXECUTOR_H
