#include <numeric>
#include <functional>
#include <algorithm>
#include <sstream>

#include "avalanche/Shape.h"

namespace avalanche {

std::size_t avalanche::Shape::size() const {
    if (std::any_of(_dims.begin(), _dims.end(),
                    [](ShapeDim dim) { return dim <= 0; })) {
        throw std::out_of_range(
            "Cannot calculate the size of a shape having dimensions <= 0");
    }
    std::size_t result = 1;
    for (auto dim: _dims) { result *= dim; }
    return result;
}

Shape Shape::reshape(const std::vector<ShapeDim> &dims) {
    std::size_t new_part_size = 1,
        undefined_dim_index = 0,
        i = 0;
    bool has_undefined_dim = false;
    for (auto dim: dims) {
        if (dim < -1) {
            throw std::invalid_argument(
                "The new shape has an invalid dimension");
        } else if (dim == -1) {
            if (has_undefined_dim) {
                throw std::invalid_argument(
                    "Only one of the dimensions can be equal -1");
            }
            undefined_dim_index = i;
            has_undefined_dim = true;
        } else {
            new_part_size *= dim;
        }
        ++i;
    }
    auto current_size = size();
    Shape result_shape(dims);
    if (!has_undefined_dim) {
        if (current_size != new_part_size) {
            throw std::invalid_argument(
                "The new shape doesn't match the original size of the array");
        }
    } else {
        if (current_size % new_part_size > 0) {
            throw std::invalid_argument(
                "The new shape doesn't match the original size of the array");
        }
        result_shape._dims[undefined_dim_index] = static_cast<ShapeDim >(
            current_size / new_part_size);
    }
    return result_shape;
}

ShapeDim Shape::dim(ShapeDim index) const {
    index = (index < 0) ? static_cast<ShapeDim>(_dims.size()) + index : index;
    return _dims.at(static_cast<std::size_t>(index));
}

ShapeDim Shape::operator[](ShapeDim index) const {
    index = (index < 0) ? static_cast<ShapeDim>(_dims.size()) + index : index;
    return _dims[static_cast<std::size_t>(index)];
}

std::string Shape::to_string() {
    std::ostringstream result;
    result << "Shape(";
    for (std::size_t i = 0; i < _dims.size(); ++i) {
        result << _dims[i] << (i == _dims.size() - 1 ? "" : ", ");
    }
    result << ")";
    return result.str();
}

void expand_shape_dims(std::vector<ShapeDim> &smaller,
                       std::vector<ShapeDim> &larger) {
    std::size_t smaller_orig_size = smaller.size(), larger_size = larger.size();
    smaller.resize(larger_size);
    std::move_backward(
        smaller.begin(),
        smaller.begin() + smaller_orig_size,
        smaller.begin() + smaller.size());
    fill_n(smaller.begin(), larger_size - smaller_orig_size, 1);
}


void Shape::align_for_broadcasting(const Shape &shape1, const Shape &shape2,
                                   Shape &shape1_aligned, Shape &shape2_aligned,
                                   Shape &result_shape) {
    shape1_aligned = shape1;
    shape2_aligned = shape2;
    {
        std::size_t size1 = shape1._dims.size(), size2 = shape2._dims.size();
        if (size1 < size2) {
            expand_shape_dims(shape1_aligned._dims, shape2_aligned._dims);
        } else if (size1 > size2) {
            expand_shape_dims(shape2_aligned._dims, shape1_aligned._dims);
        }
    }
    result_shape._dims.resize(shape1_aligned._dims.size());
    for (auto s1 = shape1_aligned._dims.begin(),
             s2 = shape2_aligned._dims.begin(),
             res = result_shape._dims.begin();
         s1 != shape1_aligned._dims.end();
         ++s1, ++s2, ++res) {
        if (!(*s1 == 1 || *s2 == 1) && *s1 != *s2) {
            throw std::invalid_argument("Incompatible shapes");
        }
        *res = std::max(*s1, *s2);
    }
};

std::array<Shape, 3>
Shape::align_for_broadcasting(const Shape &shape1, const Shape &shape2) {
    std::array<Shape, 3> result;
    align_for_broadcasting(shape1, shape2, result[0], result[1], result[2]);
    return result;
}

void validate_dims(const std::vector<ShapeDim> &dims) {
    for (auto dim: dims) {
        if (dim < -1 || dim == 0) {
            throw std::invalid_argument("An impossible shape!");
        }
    }
}

//Shape::Shape(std::vector<ShapeDim> dims) : _dims{std::move(dims)} {
//    validate_dims(_dims);
//}

Shape::Shape(const std::vector<ShapeDim> &dims) :_dims(dims) {
    validate_dims(_dims);
}


} // namespace
