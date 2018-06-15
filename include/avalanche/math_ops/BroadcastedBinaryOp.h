#ifndef AVALANCHE_BROADCASTEDBINARYOP_H
#define AVALANCHE_BROADCASTEDBINARYOP_H

#include <vector>

#include "avalanche/Shape.h"
#include "avalanche/MultiArray.h"
#include "avalanche/BaseNode.h"

namespace avalanche {

/**
 * Calculates three special arrays necessary for all broadcasted operations like
 * addition, multiplication, etc., performed on tensors.
 * @param shape1 - the shape of the first tensor
 * @param shape2 - the shape of the second tensor
 * @param size_mask1 - result for the first shape
 * @param size_mask2 - result for the second shape
 * @param result_sub_sizes - sub-sizes of all dimensions of the resulting tensor
 * @returns How big the result is in items items (not bytes). Good for testing.
 */
std::size_t broadcast_size_masks(const Shape &shape1, const Shape &shape2,
                                 std::vector<cl_ulong> &size_mask1,
                                 std::vector<cl_ulong> &size_mask2,
                                 std::vector<cl_ulong> &result_sub_sizes);


cl::Event call_broadcasted_kernel(
    cl::CommandQueue queue,
    const char *op_name,
    ArrayType dtype,
    const Shape &result_shape,
    const cl::Buffer &left_value,
    const cl::Buffer &right_value,
    const cl::Buffer &result_buffer,
    const cl::Buffer &left_mask_buffer,
    const cl::Buffer &right_mask_buffer,
    const cl::Buffer &result_sizes_buffer,
    const std::vector<cl::Event> &wait_for_events);


struct BroadcastedBinaryOp {
    Shape _result_shape;
    ArrayType _result_dtype;
    mutable std::string _kernel_name;

    Shape shape() const {
        return _result_shape;
    }

    ArrayType dtype() const {
        return _result_dtype;
    }

    virtual const char* kernel_op_name() const = 0;

    const std::string& cached_kernel_name() const {
        if (_kernel_name.empty()) {
            _kernel_name = (
                std::string("broadcasted_") + kernel_op_name() + "_" +
                array_type_name(_result_dtype));
        }
        return _kernel_name;
    }

    BroadcastedBinaryOp(const NodeRef &left, const NodeRef &right);

    MultiArrayRef forward(const MultiArrayRef &v1,
                          const MultiArrayRef &v2) const;

    bool use_in_back_propagation() const { return true; };
};

} // namespace

#endif //AVALANCHE_BROADCASTEDBINARYOP_H
