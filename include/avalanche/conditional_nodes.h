#ifndef AVALANCHE_CONDITIONAL_NODES_H
#define AVALANCHE_CONDITIONAL_NODES_H

#include "avalanche/BaseNode.h"
#include "avalanche/math_ops/ElemWiseBinaryOp.h"

namespace avalanche {

using CondExpression = std::function<NodeRef()>;

class Cond : public BaseNode {
public:
    MultiArrayRef eval(Context &context, ExecutionCache &cache) const override;

    std::string to_string() const override;

    NodeRefList inputs() const override;

    std::string repr() const override {
        return format_repr("Cond", "", "");
    }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const override ;

    bool use_in_back_propagation() const override { return true; }

    static NodeRef make(const NodeRef &condition,
                        CondExpression true_fn,
                        CondExpression false_fn);
    static NodeRef make(const NodeRef &condition,
                        const NodeRef &true_node,
                        const NodeRef &false_node);
    static NodeRef make(const NodeRef &condition,
                        const NodeRef &true_node,
                        CondExpression false_fn);
    static NodeRef make(const NodeRef &condition,
                        CondExpression true_fn,
                        const NodeRef &false_node);

private:
    const NodeRef _cond_node;
    const NodeRef _true_node;
    const NodeRef _false_node;

    Cond(const NodeRef &condition,
         const NodeRef &true_node,
         const NodeRef &false_node);
};

} // namespace

#endif //AVALANCHE_CONDITIONAL_NODES_H
