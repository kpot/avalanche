#include <numeric>
#include <algorithm>

#include "avalanche/BaseNode.h"
#include "avalanche/terminal_nodes.h"
#include "avalanche/nodes.h"

#include "avalanche/backprop.h"

namespace avalanche {

template <typename Container>
inline void leave_unique_only(Container &container) {
    std::sort(container.begin(), container.end());
    auto last = std::unique(container.begin(), container.end());
    container.erase(last, container.end());
}

NodeRefList find_involved_graph(NodeRefList start_nodes) {
    NodeRefList result;
    NodeRefList next_wave;
    leave_unique_only(start_nodes);

    std::map<const NodeRef, NodeRefList> consumers;
    while (!start_nodes.empty()) {
        std::copy(start_nodes.begin(), start_nodes.end(),
                  std::back_inserter(result));
        for (const auto &node: start_nodes) {
            auto inputs = node->inputs();
            std::copy(inputs.begin(), inputs.end(),
                      std::back_inserter(next_wave));
        }
        leave_unique_only(next_wave);
        start_nodes = std::move(next_wave);
    }
    leave_unique_only(result);
    return result;
}


ConsumerMap build_consumers_map(const NodeRefList &targets) {
    auto involved_graph = find_involved_graph(targets);
    ConsumerMap consumers;
    for (auto &node: involved_graph) {
        for (auto &input: node->inputs()) {
            consumers[input].push_back(node);
        }
    }
    return consumers;
}

const NodeRef back_propagate_node(
        const NodeRef &variable,
        const NodeRef &target,
        GradTable &grad_table,
        ConsumerMap &consumers) {
    auto cached = grad_table.find(variable);
    if (cached != grad_table.end()) {
        return cached->second;
    }
    NodeRefList chunks;
    auto &var_consumers = consumers[variable];
    chunks.reserve(var_consumers.size());
    for (const auto &consumer: var_consumers) {
        if (consumer->use_in_back_propagation()) {
            auto d_target_wrt_consumer = back_propagate_node(
                consumer, target, grad_table, consumers);
            auto d_chunk = consumer->apply_chain_rule(
                variable, d_target_wrt_consumer, consumer->inputs());
            chunks.push_back(d_chunk);
        }
    }
    NodeRef result;
    if (chunks.empty()) {
        result = Constant::zeros_like(variable);
    } else {
        result = chunks[0];
        for (std::size_t i = 1; i < chunks.size(); ++i) {
            result = F<ElemWisePlus>(result, chunks[i]);
        }
    }
    grad_table.insert({variable, result});
    return result;
}

/**
 * Constructs computational graph for calculating derivatives of the target
 * node with respect to every node listed in the second argument.
 * @param target target node for which we calculate the derivatives
 * @param with_respect_to calculate derivatives with respect to these variables
 * @return a mapping between nodes listed in `with_respect_to` and the output
 *  nodes of respecting derivatives, as well as any other derivatives that
 *  were necessary for calculating these ones.
 */
GradTable build_back_propagation_graph(const NodeRef &target,
                                       const NodeRefList &with_respect_to) {
    GradTable grad_table;
    grad_table.insert({target, Constant::ones_like(target)});

    auto consumers = build_consumers_map({target});
    for (const auto &variable: with_respect_to) {
        back_propagate_node(variable, target, grad_table, consumers);
    }
    return grad_table;
}

/**
 * Acts similarly to keras.backend.gradient, returning list of gradients
 * for every variable listed in `with_respect_to`.
 * See `build_back_propagation_graph` for more info.
 */
NodeRefList build_gradients(const NodeRef &target,
                            const NodeRefList &with_respect_to) {
    NodeRefList result;
    auto grad_table = build_back_propagation_graph(target, with_respect_to);
    for (const NodeRef &var: with_respect_to) {
        result.push_back(grad_table[var]);
    }
    return result;
}

} // namespace
