//
// Created by Kirill on 29/01/18.
//

#include <sstream>

#include "avalanche/BaseNode.h"

std::string avalanche::BaseNode::format_repr(const std::string &operation,
                                             const std::string &name) const {
    std::ostringstream res;
    res << "<" << operation << " | id: " << std::to_string(id)
        << ", shape: " << Shape::dims_to_string(shape().dims())
        << ", type: " << array_type_name(dtype());
    if (!name.empty()) {
        res << ", name: " << name;
    }
    res << ">";
    return res.str();
}
