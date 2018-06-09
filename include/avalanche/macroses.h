#ifndef AVALANCHE_MATH_OPS_MACROSES_H
#define AVALANCHE_MATH_OPS_MACROSES_H

#define ARRAY_DTYPE_SWITCH_FUNCTION(NAME, NESTED_FUNC, RESULT_TYPE, QUALIFIER) \
template <class... Args> \
inline RESULT_TYPE NAME(ArrayType dtype, Args&&... args) QUALIFIER { \
    switch (dtype) { \
        case ArrayType::float32: \
            return NESTED_FUNC<cl_float>(std::forward<Args>(args)...); \
        case ArrayType::float16: \
            return NESTED_FUNC<cl_half>(std::forward<Args>(args)...); \
        case ArrayType::float64: \
            return NESTED_FUNC<cl_double >(std::forward<Args>(args)...); \
        case ArrayType::int8: \
            return NESTED_FUNC<cl_char>(std::forward<Args>(args)...); \
        case ArrayType::int16: \
            return NESTED_FUNC<cl_short>(std::forward<Args>(args)...); \
        case ArrayType::int32: \
            return NESTED_FUNC<cl_int>(std::forward<Args>(args)...); \
        case ArrayType::int64: \
            return NESTED_FUNC<cl_long>(std::forward<Args>(args)...); \
    } \
};

#define ARRAY_DTYPE_SWITCH_FLOAT_FUNCTION(NAME, NESTED_FUNC, RESULT_TYPE, QUALIFIER) \
template <class... Args> \
inline RESULT_TYPE NAME(ArrayType dtype, Args&&... args) QUALIFIER { \
    switch (dtype) { \
        case ArrayType::float32: \
            return NESTED_FUNC<cl_float>(std::forward<Args>(args)...); \
        case ArrayType::float16: \
            return NESTED_FUNC<cl_half>(std::forward<Args>(args)...); \
        case ArrayType::float64: \
            return NESTED_FUNC<cl_double >(std::forward<Args>(args)...); \
        default: \
            throw std::invalid_argument("Unsupported data type"); \
    } \
}




#endif //AVALANCHE_MATH_OPS_MACROSES_H
