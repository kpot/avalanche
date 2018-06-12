#ifndef AVALANCHE_RANDOM_NODES_H
#define AVALANCHE_RANDOM_NODES_H

#include <vector>
#include <string>
#include <cstdint>

#include "avalanche/BaseNode.h"

namespace avalanche {

class UniformRandom : public BaseNode {
public:
    MultiArrayRef eval(Context &context, ExecutionCache &cache) const override;

    const NodeRef
    apply_chain_rule(const NodeRef &wrt_input, const NodeRef &d_target_wrt_this,
                     const NodeRefList &all_inputs) const override {
        return d_target_wrt_this;
    }

    std::string to_string() const override;

    std::string repr() const override;

    NodeRefList inputs() const override { return NodeRefList(); }

    /**
     *
     * @param shape
     * @param min_value
     * @param max_value
     * @param dtype
     * @param seed : if zero, std::random_device will be used
     *    to initialize the base offset
     * @return
     */
    static NodeRef make(const Shape &shape, double min_value,
                        double max_value, ArrayType dtype,
                        std::uint64_t seed) {
        if (min_value > max_value) {
            throw std::invalid_argument(
                "Min random value should not be larger than max random value");
        }
        return std::static_pointer_cast<BaseNode>(
            std::shared_ptr<UniformRandom>(
                new UniformRandom(shape, min_value, max_value, dtype, seed)));
    }

private:
    double _min_value;
    double _max_value;
    std::uint64_t _seed;
    mutable std::string _generate_uniform_kernel_name;

    const std::string& cached_generate_uniform_kernel_name() const;
    MultiArrayRef generate_uniform_random(const MultiArrayRef &seeds) const;

    explicit UniformRandom(const Shape &shape, double min_value,
                           double max_value, ArrayType dtype,
                           std::uint64_t seed);
};

} // namespace

#endif //AVALANCHE_RANDOM_NODES_H
