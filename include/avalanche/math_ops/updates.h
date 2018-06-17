#ifndef AVALANCHE_UPDATES_H
#define AVALANCHE_UPDATES_H

#include "avalanche/base_ops_nodes.h"

namespace avalanche {

class BaseUpdateOp {
public:
    BaseUpdateOp(const NodeRef &variable,
             const NodeRef &update,
             const std::string &operation_name,
             const std::string &operation_cl_code);

    Shape shape() const { return _result_shape; }

    ArrayType dtype() const { return _result_dtype; }

    MultiArrayRef forward(const MultiArrayRef &v1,
                          const MultiArrayRef &v2) const;

    virtual bool use_in_back_propagation() const { return false; };

    const NodeRef apply_chain_rule(
            const NodeRef &wrt_input,
            const NodeRef &d_target_wrt_this,
            const NodeRefList &all_inputs) const {
        return d_target_wrt_this;
    }

    std::string name() const { return _operation_name; }

private:
    Shape _result_shape;
    ArrayType _result_dtype;
    std::string _operation_name;
    std::string _kernel_name;
    std::string _kernel_source;
};

class Update: public BaseUpdateOp {
public:
    Update(const NodeRef &left, const NodeRef &right)
        :BaseUpdateOp(left, right, "update", "=") {}
};

class UpdateAdd: public BaseUpdateOp {
public:
    UpdateAdd(const NodeRef &left, const NodeRef &right)
        :BaseUpdateOp(left, right, "update_add", "+=") {}
};

class UpdateSub: public BaseUpdateOp {
public:
    UpdateSub(const NodeRef &left, const NodeRef &right)
        :BaseUpdateOp(left, right, "update_sub", "-=") {}
};

} // namespace

#endif //AVALANCHE_UPDATES_H
