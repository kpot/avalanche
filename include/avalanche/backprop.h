#ifndef AVALANCHE_BACKPROP_H
#define AVALANCHE_BACKPROP_H

namespace avalanche {

using ConsumerMap = std::map<const NodeRef, NodeRefList>;
using GradTable = std::map<const NodeRef, const NodeRef>;

ConsumerMap build_consumers_map(const NodeRefList &targets);

/**
 * Constructs new computational graph calculating gradients of the target
 * function with respect to some given variables, using backward-mode automatic
 * differentiation.
 *
 * @param target - a node representing function we calculate gradients for.
 * @param with_respect_to - a list of variables for which we calculate the
 *      gradient.
 * @return
 */
GradTable build_back_propagation_graph(
    const NodeRef &target,
    const NodeRefList &with_respect_to);


/**
 * Acts similarly to keras.backend.gradient, returning list of gradients
 * for every variable listed in `with_respect_to`.
 * See `build_back_propagation_graph` for more info.
 */
NodeRefList build_gradients(const NodeRef &target,
                            const NodeRefList &with_respect_to);

} // namespace

#endif //AVALANCHE_BACKPROP_H
