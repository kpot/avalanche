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


/**
 * Base class for many operations involving broadcasting, which allows
 * to operate on arrays of different shapes without the need to transform
 * them. This class uses a general kernel doing the job, which every
 * subclass can tweak as necessary.
 * More about broadcasting can be found here:
 * https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
 */
class BroadcastedBinaryOp {
public:
    Shape shape() const {
        return _result_shape;
    }

    ArrayType dtype() const {
        return _result_dtype;
    }

    BroadcastedBinaryOp(const NodeRef &left,
                        const NodeRef &right,
                        const std::string &operation_name,
                        const std::string &operation_cl_code,
                        ArrayType output_dtype);

    MultiArrayRef forward(const MultiArrayRef &v1,
                          const MultiArrayRef &v2) const;

    virtual bool use_in_back_propagation() const { return true; };


private:
    Shape _result_shape;
    ArrayType _result_dtype;
    std::string _operation_name;
    std::string _kernel_name;
    std::string _kernel_source;
};

} // namespace

#endif //AVALANCHE_BROADCASTEDBINARYOP_H
