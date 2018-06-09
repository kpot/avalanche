//
// Created by Kirill on 18/02/18.
//

#ifndef AVALANCHE_ARRAYTYPE_H
#define AVALANCHE_ARRAYTYPE_H

#include "CL_cust/cl2.hpp"

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
const std::size_t ArrayTypeSizes[] = {2, 4, 8, 1, 2, 4, 8};

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

} // namespace

#endif //AVALANCHE_ARRAYTYPE_H
