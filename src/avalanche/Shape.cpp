#include <numeric>
#include <functional>
#include <algorithm>
#include <sstream>

#include <fmt/format.h>

#include "avalanche/Shape.h"

namespace avalanche {

std::size_t avalanche::Shape::size() const {
    if (std::any_of(_dims.begin(), _dims.end(),
                    [](ShapeDim dim) { return dim <= 0; })) {
        throw std::out_of_range(
            "Cannot calculate the size of a shape having dimensions "
            "of 0 size or undefined");
    }
    std::size_t result = 1;
    for (auto dim: _dims) { result *= dim; }
    return result;
}

Shape Shape::reshape(const std::vector<ShapeDim> &dims) const {
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
                fmt::format("The new shape {} doesn't match "
                            "the original shape of the array {}",
                            result_shape.to_string(), to_string()));
        }
    } else {
        if (current_size % new_part_size > 0) {
            throw std::invalid_argument(
                fmt::format("The new shape {} doesn't match "
                            "the original shape of the array {}",
                            result_shape.to_string(), to_string()));
        }
        result_shape._dims[undefined_dim_index] = static_cast<ShapeDim >(
            current_size / new_part_size);
    }
    return result_shape;
}

std::size_t Shape::dim_real_index(ShapeDim dim) const {
    return static_cast<std::size_t>(
        (dim < 0) ? static_cast<ShapeDim>(_dims.size()) + dim : dim);
}

ShapeDim Shape::dim(ShapeDim index) const {
    return _dims.at(dim_real_index(index));
}

ShapeDim Shape::operator[](ShapeDim index) const {
    return _dims[dim_real_index(index)];
}

std::string Shape::to_string() const {
    std::ostringstream result;
    result << "Shape(";
    for (std::size_t i = 0; i < _dims.size(); ++i) {
        if (_dims[i] < 0) {
            result << "?";
        } else {
            result << _dims[i];
        }
        result << (i == _dims.size() - 1 ? "" : ", ");
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
                                   // Outputs
                                   Shape &shape1_aligned, Shape &shape2_aligned,
                                   Shape &result_shape) {
    shape1_aligned = shape1;
    shape2_aligned = shape2;
    {
        std::size_t size1 = shape1.rank(), size2 = shape2.rank();
        if (size1 < size2) {
            expand_shape_dims(shape1_aligned._dims, shape2_aligned._dims);
        } else if (size1 > size2) {
            expand_shape_dims(shape2_aligned._dims, shape1_aligned._dims);
        }
    }
    result_shape._dims.resize(shape1_aligned.rank());
    for (auto s1 = shape1_aligned._dims.begin(),
             s2 = shape2_aligned._dims.begin(),
             res = result_shape._dims.begin();
         s1 != shape1_aligned._dims.end();
         ++s1, ++s2, ++res) {
        if (!(*s1 == 1 || *s2 == 1) && *s1 != *s2 &&
                *s1 != UnknownDim && *s2 != UnknownDim) {
            throw std::invalid_argument(
                fmt::format("Cannot align shapes {} and {} for broadcasting",
                            shape1.to_string(), shape2.to_string()));
        }

        if (*s1 == UnknownDim) {
            *res = (*s2 == UnknownDim) ? UnknownDim : *s2;
        } else {
            *res = (*s2 == UnknownDim) ? *s1 : std::max(*s1, *s2);
        }
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

Shape::Shape(std::initializer_list<ShapeDim> dims) :_dims(dims) {
    validate_dims(_dims);
}

std::string Shape::dims_to_string(const std::vector<ShapeDim> &dims,
                                  bool convert_unknown) {
    std::ostringstream result;
    result << "{";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == UnknownDim && convert_unknown) {
            result << "?";
        } else {
            result << dims[i];
        }
        result << (i == dims.size() - 1 ? "" : ", ");
    }
    result << "}";
    return result.str();
}

bool Shape::is_complete() const {
    return std::find(_dims.begin(), _dims.end(), UnknownDim) == _dims.end();
}

bool Shape::agrees_with(const Shape &needed) const {
    return agrees_with(needed.dims());
}

bool Shape::agrees_with(const std::vector<ShapeDim> &needed) const {
    if (rank() != needed.size()) return false;
    for (std::size_t i = 0; i < _dims.size(); ++i) {
        if (_dims[i] != needed[i] && needed[i] != UnknownDim) {
            return false;
        }
    }
    return true;
}

template <typename Container>
inline void leave_unique_only(Container &container) {
    std::sort(container.begin(), container.end());
    auto last = std::unique(container.begin(), container.end());
    container.erase(last, container.end());
}

std::vector<ShapeDim> Shape::normalize_dims(const std::vector<ShapeDim> &dims) const {
    auto input_rank = rank();
    std::vector<ShapeDim> result = dims;
    for (auto &dim: result) {
        auto orig_dim = dim;
        if (dim < 0) {
            dim = static_cast<ShapeDim>(input_rank) + dim;
        }
        if (dim >= input_rank || dim < 0) {
            throw std::invalid_argument(
                fmt::format("One of the axis ({}) doesn't exist", orig_dim));
        }
    }
    leave_unique_only(result);
    return result;
}

std::vector<ShapeDim>
Shape::dims_difference(const Shape &aligned_shape, const Shape &result_shape) {
    std::vector<ShapeDim> result;
    if (aligned_shape.rank() != result_shape.rank()) {
        throw std::invalid_argument("The shapes must be aligned!");
    }
    for (ShapeDim i = 0; i < aligned_shape.rank(); ++i) {
        if (aligned_shape.dims()[i] != result_shape.dims()[i]) {
            result.push_back(i);
        }
    }
    return result;
}

void Shape::normalize_range(const ShapeDim axis, const Range &range,
                            ShapeDim &real_axis, Range &real_range) const {
    real_axis = static_cast<ShapeDim>(dim_real_index(axis));
    if (real_axis >= rank() || real_axis < 0) {
        throw std::invalid_argument(
            fmt::format(
                "Cannot slice along given axis {} because it "
                "exceeds the shape ({})",
                real_axis, to_string()));
    }
    auto axis_size = _dims[real_axis];
    if (axis_size == UnknownDim) {
        real_range = range;
    } else {
        ShapeDim range_start = (range.start < 0) ? axis_size + range.start
                                                 : range.start;
        ShapeDim range_end = (range.end < 0) ? axis_size + range.end
                                             : range.end;
        if (range_start > range_end) {
            throw std::invalid_argument(
                fmt::format(
                    "Range start index {} ({}) cannot be large than "
                    "the end index {} ({})",
                    range.start, range_start, range.end, range_end));
        }
        if (range_start < 0 || range_end < 0 ||
            range_start >= axis_size || range_end >= axis_size) {
            throw std::invalid_argument(
                fmt::format(
                    "Range {}:{} is out of the boundaries "
                    "of axis {} (the size of {})",
                    range_start, range_end, real_axis, axis_size));
        }
        real_range = {range_start, range_end};
    }
}

ShapeDim Shape::dims_product(const std::vector<ShapeDim> dims,
                             ShapeDim start, ShapeDim end) {
    ShapeDim result = 1;
    for (ShapeDim i = start; i <= end; ++i) {
        result *= dims[i];
    }
    return result;
}


} // namespace
