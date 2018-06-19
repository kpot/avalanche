#ifndef AVALANCHE_TRANSFORM_H
#define AVALANCHE_TRANSFORM_H

#include <string>
#include <sstream>
#include <vector>
#include <iostream>

#include "avalanche/CodeCache.h"
#include "avalanche/opencl_utils.h"
#include "avalanche/MultiArray.h"
#include "avalanche/BaseNode.h"
#include "avalanche/Shape.h"
#include "avalanche/casting.h"
#include "avalanche/macroses.h"
#include "avalanche/math_ops/messages.h"

namespace avalanche {

struct TransformVar {
    std::string name;
    std::string expression;
};

/**
 * A foundation for any operation that needs to take some array and
 * convert it into an another array of the same size and shape by using
 * a mathematical expression with a scalar constant. Like "multiply by 2"
 * or "x to the power of 10". A huge number of operations fall into
 * this category.
 *
 * @tparam NumParams how many constants the operation involves.
 */
template <int NumParams>
class ConstTransform {
public:
    ArrayType result_dtype;
    Shape result_shape;
    std::string left_name;
    std::string right_name;
    std::array<float, NumParams> params;
    std::vector<TransformVar> variables;
    std::string opencl_operation;
    std::string kernel_source;
    std::string program_name;

    /**
     * @param input
     * @param variables intermediary variables that must be calculated before
     *   the main expression
     * @param expression the final expression that needs to be calculated,
     *   including all extra variables
     * @param lh_name left hand name (only for presentation purposes)
     * @param rh_name right hand name (only for presentation purposes)
     * @param params
     */
    ConstTransform(const NodeRef &input,
                   ArrayType output_type,
                   const std::vector<TransformVar> &variables,
                   const std::string &expression,
                   const std::string &lh_name,
                   const std::string &rh_name,
                   const std::array<float, NumParams> &params)
        :result_dtype{output_type},
         result_shape{input->shape()},
         left_name{lh_name},
         right_name{rh_name},
         params{params},
         variables{variables},
         opencl_operation{expression},
         kernel_source{
             generate_kernel_source(cl_type_name_of_array(input->dtype()),
                                    cl_type_name_of_array(output_type),
                                    opencl_operation)},
         program_name{opencl_operation
                      + cl_type_name_of_array(input->dtype())}
    {
    }

    std::string lh_name() const { return left_name; }
    std::string rh_name() const { return right_name; }

    const Shape& shape() const { return result_shape; }

    virtual ArrayType dtype() const { return result_dtype; }

    const std::string& name() const { return opencl_operation; }

    MultiArrayRef forward(const MultiArrayRef &value) const {
        auto pool = value->buffer_unsafe()->pool();
        auto queue = pool->cl_queue();
        auto result = pool->make_array(value->shape(), result_dtype);
        result->set_label(opencl_operation + " via " + __func__, __LINE__);
        result->add_dependencies({value});
        auto wait_for_data = make_event_list(
            {value->buffer_unsafe()->completion_event()});
        auto is_ready = transforming_kernel_switch(
            result_dtype,
            queue,
            value->cl_buffer_unsafe(),
            result->cl_buffer_unsafe(),
            value->shape().size(),
            wait_for_data);
        result->set_completion_event(is_ready);
        return result;
    }

    bool use_in_back_propagation() const { return true; };

    virtual const NodeRef apply_chain_rule(const NodeRef &wrt_input,
                                   const NodeRef &d_target_wrt_this,
                                   const NodeRefList &all_inputs) const {
        if (all_inputs[0] == wrt_input) { // d(x^a) / dx == a * x ^ (a - 1)
            return F<ElemWiseMultiply>(
                d_target_wrt_this,
                partial_derivative(wrt_input));
        } else {
            throw std::logic_error(messages::CANT_DIFF_UNEXISTING_INPUT_MESSAGE);
        }
    }

    virtual const NodeRef partial_derivative(const NodeRef &input) const {
        throw std::logic_error("not implemented");
    }

    std::string generate_kernel_source(const std::string &input_type_name,
                                       const std::string &output_type_name,
                                       const std::string &expression) const {
        std::ostringstream o;
        o << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel __attribute__((reqd_work_group_size(64, 1, 1)))\n"
            "void transform(\n"
          << "\t__global " << input_type_name << " *source,\n"
          << "\t__global " << output_type_name << " *output,\n"
          << "\tconst ulong result_size\n" << (NumParams > 0 ? ',' : ')');
        for (std::size_t i = 0; i < NumParams; ++i) {
            o << "\tconst " << output_type_name << " p"
              << i << (i == NumParams - 1 ? ')' : ',') << '\n';
        }
        o << "{\n"
          << "\tconst size_t i = get_global_id(0);\n"
          << "\tconst " << output_type_name << " v = source[i];\n";
        for (auto &var: variables) {
            o << "\tconst " << output_type_name << " "
              << var.name << " = " << var.expression << ";\n";
        }
        o << "\tif (i < result_size) { output[i] = " << expression
          << "; }\n}\n";
        return o.str();
    }

    template <typename T>
    cl::Event call_transforming_kernel(
            cl::CommandQueue queue,
            const cl::Buffer &source_value,
            const cl::Buffer &result_buffer,
            const std::size_t result_size,
            const std::vector<cl::Event> &wait_for_events) const {
        auto context = get_context_from_queue(queue);
        auto program = CodeCache::get_default().get_program(
            context,
            queue,
            program_name,
            kernel_source,
            "");
        cl::Kernel kernel;
        std::vector<cl::Kernel> kernels;
        try {
            kernel = cl::Kernel(program, "transform");
        } catch (cl::Error &e) {
            throw std::runtime_error(get_opencl_error_string(e.err()));
        }
        kernel.setArg(0, source_value);
        kernel.setArg(1, result_buffer);
        kernel.setArg(2, static_cast<cl_ulong>(result_size));
        for (cl_uint i = 0; i < NumParams; ++i) {
            kernel.setArg(i + 3, to_array_type<T>(params[i]));
        }
        constexpr std::size_t work_group_size = 64;
        const auto work_items = make_divisible_by(work_group_size, result_size);
        cl::Event work_is_done;
        queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(work_items),
            cl::NDRange(work_group_size),
            &wait_for_events,
            &work_is_done);
        return work_is_done;
    }

    ARRAY_DTYPE_SWITCH_FUNCTION(
        transforming_kernel_switch,
        call_transforming_kernel,
        cl::Event,
        const)
};

class SPower : public ConstTransform<2> {
public:
    SPower(const NodeRef &input, float scale, float power)
        :ConstTransform<2>(
            input,
            input->dtype(),
            {},
            opencl_expression(input->dtype()),
            std::string("pow(") + std::to_string(scale) + "*",
            std::string(", ") + std::to_string(power) + ")",
            {scale, power}) {
    }

    static std::string opencl_expression(ArrayType dtype)  {
        return "p0 * pow(v, p1)";
    }

    const NodeRef partial_derivative(const NodeRef &input) const override;
};

class Square : public ConstTransform<0> {
public:
    Square(const NodeRef &input)
        :ConstTransform<0>(
        input,
        input->dtype(),
        {},
        opencl_expression(input->dtype()),
        "square(", ")",
        {}) {
    }

    static std::string opencl_expression(ArrayType dtype)  {
        return "v * v";
    }

    const NodeRef partial_derivative(const NodeRef &input) const override;
};

// Calculates reciprocal (1/x) of a given number
class Recip : public ConstTransform<0> {
public:
    Recip(const NodeRef &input)
        : ConstTransform<0>(input, input->dtype(), {},
                            opencl_expression(input->dtype()),
                            "recip(", ")", {}) {}

    static std::string opencl_expression(ArrayType dtype) {
        switch (dtype) {
            case ArrayType::float32:
                return "native_recip(v)";
            case ArrayType::float16:
                return "half_recip(v)";
            default:
                return "pown(v, -1)";
        }
    }

    const NodeRef partial_derivative(const NodeRef &input) const override;
};


class Scale : public ConstTransform<1> {
public:
    Scale(const NodeRef &input, float value)
        : ConstTransform<1>(input, input->dtype(), {},
                            opencl_expression(input->dtype()),
                            std::to_string(value) + " * ", "", {value})
    {}

    static std::string opencl_expression(ArrayType dtype) {
        return "p0 * v";
    }

    const NodeRef
    apply_chain_rule(const NodeRef &wrt_input,
                     const NodeRef &d_target_wrt_this,
                     const NodeRefList &all_inputs) const override;
};


class Cast : public ConstTransform<1> {
public:
    Cast(const NodeRef &input, ArrayType another_type)
        : ConstTransform<1>(
            input, another_type, {}, opencl_expression(input->dtype()),
            "cast(", std::string(", ") + array_type_name(another_type) + ")",
            {}),
          _another_type{another_type}
    {
    }

    static std::string opencl_expression(ArrayType dtype) {
        return "v";
    }

    const NodeRef
    apply_chain_rule(const NodeRef &wrt_input,
                     const NodeRef &d_target_wrt_this,
                     const NodeRefList &all_inputs) const override;

    ArrayType dtype() const override { return _another_type; }

private:
    ArrayType _another_type;
};


class Log : public ConstTransform<0> {
public:
    explicit Log(const NodeRef &input)
        : ConstTransform<0>(input, input->dtype(), {},
                            opencl_expression(input->dtype()),
                            "log(", ")", {}) {}

    static std::string opencl_expression(ArrayType dtype) {
        switch (dtype) {
            case ArrayType::float32:
                return "native_log(v)";
            case ArrayType::float16:
                return "half_log(v)";
            default:
                return "log(v)";
        }
    }

    const NodeRef partial_derivative(const NodeRef &input) const override;
};

class Sqrt : public ConstTransform<0> {
public:
    explicit Sqrt(const NodeRef &input)
        : ConstTransform<0>(input, input->dtype(), {},
                            opencl_expression(input->dtype()),
                            "sqrt(", ")", {}) {}

    static std::string opencl_expression(ArrayType dtype) {
        switch (dtype) {
            case ArrayType::float32:
                return "native_sqrt(v)";
            case ArrayType::float16:
                return "half_sqrt(v)";
            default:
                return "sqrt(v)";
        }
    }

    const NodeRef partial_derivative(const NodeRef &input) const override;
};

class Sigmoid : public ConstTransform<0> {
public:
    explicit Sigmoid(const NodeRef &input)
        : ConstTransform<0>(input, input->dtype(), {},
                            opencl_expression(input->dtype()),
                            "sigmoid(", ")", {}) {}

    static std::string opencl_expression(ArrayType dtype) {
        switch (dtype) {
            case ArrayType::float32:
                return "native_recip(1 + native_exp(-v))";
            case ArrayType::float16:
                return "half_recip(1 + native_exp(-v))";
            default:
                return "v";
        }
    }

    const NodeRef partial_derivative(const NodeRef &input) const override;
};

class Exp : public ConstTransform<0> {
public:
    explicit Exp(const NodeRef &input)
        : ConstTransform<0>(input, input->dtype(), {},
                            "native_exp(v)", "exp(", ")", {}) {}

    const NodeRef partial_derivative(const NodeRef &input) const override;
};

class Tanh : public ConstTransform<0> {
public:
    explicit Tanh(const NodeRef &input)
        : ConstTransform<0>(input, input->dtype(), {},
                            opencl_expression(input->dtype()),
                            "tanh(", ")", {}) {}

    static std::string opencl_expression(ArrayType dtype) {
        return "tanh(v)";
    }

    const NodeRef partial_derivative(const NodeRef &input) const override;
};


class ReLU : public ConstTransform<0> {
public:
    explicit ReLU(const NodeRef &input)
        : ConstTransform<0>(input, input->dtype(), {},
                            opencl_expression(input->dtype()),
                            "relu(", ")", {}) {}

    static std::string opencl_expression(ArrayType dtype) {
        const char *type_name = avalanche::cl_type_name_of_array(dtype);
        return std::string("max((") + type_name + ")0.0, v)";
    }

    const NodeRef partial_derivative(const NodeRef &input) const override;
};

class Negate : public Scale {
public:
    explicit Negate(const NodeRef &input) : Scale(input, -1) {}
};

inline NodeRef operator*(float scalar, const NodeRef &node2) {
    return FU<Scale>(node2, scalar);
}

inline NodeRef operator*(const NodeRef &node1, float scalar) {
    return FU<Scale>(node1, scalar);
}

inline NodeRef operator-(const NodeRef &node1) {
    return FU<Negate>(node1);
}


} // namespace

#endif //AVALANCHE_TRANSFORM_H
