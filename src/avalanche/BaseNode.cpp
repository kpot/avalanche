#include <sstream>

#include "avalanche/BaseNode.h"

std::string avalanche::BaseNode::format_repr(const std::string &operation,
                                             const std::string &name,
                                             const std::string &extra) const {
    std::ostringstream res;
    res << "<" << operation << " | id: " << std::to_string(id)
        << ", shape: " << Shape::dims_to_string(shape().dims())
        << ", type: " << array_type_name(dtype());
    if (!name.empty()) {
        res << ", name: " << name;
    }
    if (!extra.empty()) {
        res << ", " << extra;
    }
    res << ">";
    return res.str();
}

std::string avalanche::BaseNode::tree_repr() {
    std::ostringstream out;
    _tree_repr_body(0, out);
    return out.str();
}

void avalanche::BaseNode::_tree_repr_body(int depth, std::ostringstream &out) {
    for (int i = 0; i < 3 * depth; ++i) out << " ";
    if (depth > 0) out << "|-";
    out << repr() << "\n";
    for (const auto &node: inputs()) {
        node->_tree_repr_body(depth + 1, out);
    }
}
