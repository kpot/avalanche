#ifndef AVALANCHE_ARRAYTYPE_H
#define AVALANCHE_ARRAYTYPE_H

#include <cstring>

#include "CL_cust/cl2.hpp"

#include "avalanche/casting.h"

namespace avalanche {

enum struct ArrayType {
    // Do not change the order here.
    // There are some places depending on it (like ArrayTypeCLNames)
        float16 = 0, float32 = 1, float64 = 2,
    int8 = 3, int16 = 4, int32 = 5, int64 = 6
};
const char ArrayTypeCLNames[][7] = {
    // Follows the same order as types from ArrayType to simplify lookups
    "half", "float", "double", "char", "short", "int", "long"};
const char ArrayTypeNumpyNames[][7] = {
    // Follows the same order as types from ArrayType to simplify lookups
    "f2", "f4", "f8", "i1", "i2", "i4", "i8"};
const char ArrayTypeVerboseNames[][9] = {
    "float16", "float32", "float64", "int8", "int16", "int32", "int64"};
const std::size_t ArrayTypeSizes[] = {
    sizeof(cl_half), sizeof(cl_float), sizeof(cl_double), 1, 2, 4, 8};

template<class T> constexpr ArrayType dtype_of_static_type = ArrayType::int8;
template<> constexpr ArrayType dtype_of_static_type<cl_short> = ArrayType::int16;
template<> constexpr ArrayType dtype_of_static_type<cl_int> = ArrayType::int32;
template<> constexpr ArrayType dtype_of_static_type<cl_long> = ArrayType::int64;
template<> constexpr ArrayType dtype_of_static_type<cl_half> = ArrayType::float16;
template<> constexpr ArrayType dtype_of_static_type<float> = ArrayType::float32;
template<> constexpr ArrayType dtype_of_static_type<double> = ArrayType::float64;

template<class T> constexpr char cl_type_name_of_static_type[] = "char";
template<> constexpr char cl_type_name_of_static_type<cl_short>[] = "short";
template<> constexpr char cl_type_name_of_static_type<cl_int>[] = "int";
template<> constexpr char cl_type_name_of_static_type<cl_long>[] = "long";
template<> constexpr char cl_type_name_of_static_type<cl_half>[] = "half";
template<> constexpr char cl_type_name_of_static_type<float>[] = "float";
template<> constexpr char cl_type_name_of_static_type<double>[] = "double";

inline const char *cl_type_name_of_array(ArrayType t) {
    if (static_cast<int>(t) >=
        (sizeof(ArrayTypeCLNames) / sizeof(ArrayTypeCLNames[0]))) {
        return "unsupported";
    }
    return ArrayTypeCLNames[static_cast<int>(t)];
}

inline const char *numpy_type_name_of_array(ArrayType t) {
    if (static_cast<int>(t) >=
        (sizeof(ArrayTypeNumpyNames) / sizeof(ArrayTypeNumpyNames[0]))) {
        return "unsupported";
    }
    return ArrayTypeNumpyNames[static_cast<int>(t)];
}

inline const char *array_type_name(ArrayType t) {
    if (static_cast<int>(t) >=
        (sizeof(ArrayTypeVerboseNames) / sizeof(ArrayTypeVerboseNames[0]))) {
        return "unsupported";
    }
    return ArrayTypeVerboseNames[static_cast<int>(t)];
}

inline std::size_t array_type_size(ArrayType t) {
    if (static_cast<int>(t) >=
        (sizeof(ArrayTypeVerboseNames) / sizeof(ArrayTypeVerboseNames[0]))) {
        return 0;
    }
    return ArrayTypeSizes[static_cast<int>(t)];
}

inline bool is_floating_array_type(ArrayType t) {
    return (t == ArrayType::float32 || t == ArrayType::float16 ||
            t == ArrayType::float64);
}


/**
 * similar to `to_array_type`, only more general
 * Converts a `value` to a type represented by `dtype` and copies the result
 * into a "one size fit all" uint64 value.
 * Useful when you need to pass something to an OpenCL kernel.
 */
template <typename T>
inline std::uint64_t cast_to_value_of_array_type(ArrayType dtype, T value) {
    std::uint64_t result = 0;
    switch (dtype) {
        case ArrayType::float16: {
            cl_half tmp_half = float_to_half(static_cast<float>(value));
            std::memcpy(&result, &tmp_half, sizeof(tmp_half));
            break;
        }
        case ArrayType::float32: {
            auto tmp_float = static_cast<float>(value);
            std::memcpy(&result, &tmp_float, sizeof(tmp_float));
            break;
        }
        case ArrayType::float64: {
            auto tmp_double = static_cast<double>(value);
            std::memcpy(&result, &tmp_double, sizeof(tmp_double));
            break;
        }
        case ArrayType::int8: {
            auto tmp_int8 = static_cast<std::int8_t>(value);
            std::memcpy(&result, &tmp_int8, sizeof(tmp_int8));
            break;
        }
        case ArrayType::int16: {
            auto tmp_int16 = static_cast<std::int16_t>(value);
            std::memcpy(&result, &tmp_int16, sizeof(tmp_int16));
            break;
        }
        case ArrayType::int32: {
            auto tmp_int32 = static_cast<std::int32_t>(value);
            std::memcpy(&result, &tmp_int32, sizeof(tmp_int32));
            break;
        }
        case ArrayType::int64: {
            auto tmp_int64 = static_cast<std::int64_t>(value);
            std::memcpy(&result, &tmp_int64, sizeof(tmp_int64));
            break;
        }
    }
    return result;
}


/**
 * Mimics the rules C++ uses for operations on mixed types, such
 * as multiplication of an integer and a float numbers.
 * Returns a common type to which all arguments should be converted.
 **/
inline ArrayType choose_common_array_type(ArrayType dtype1, ArrayType dtype2) {
    bool a_is_floating = is_floating_array_type(dtype1);
    bool b_is_floating = is_floating_array_type(dtype2);
    if (a_is_floating) {
        if (b_is_floating) {
            return static_cast<ArrayType>(
                std::max(static_cast<int>(dtype1),
                         static_cast<int>(dtype2)));
        } else {
            return dtype1;
        }
    } else {
        if (b_is_floating) {
            return dtype2;
        } else {
            return static_cast<ArrayType>(
                std::max(static_cast<int>(dtype1),
                         static_cast<int>(dtype2)));
        }
    }
}

} // namespace

#endif //AVALANCHE_ARRAYTYPE_H
