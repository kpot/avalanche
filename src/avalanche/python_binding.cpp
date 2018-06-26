#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <Python.h>

#include "avalanche/terminal_nodes.h"
#include "avalanche/nodes.h"
#include "avalanche/Shape.h"
#include "avalanche/Executor.h"

namespace py = pybind11;

namespace avalanche {

class PyBaseNode : public BaseNode {
public:
    using BaseNode::BaseNode;

    MultiArrayRef
    eval(Context &context, ExecutionCache &cache) const override {
        PYBIND11_OVERLOAD_PURE(
            MultiArrayRef,
            BaseNode,
            eval,
            context, cache);
    }

    const NodeRef apply_chain_rule(
        const NodeRef &wrt_input,
        const NodeRef &d_target_wrt_this,
        const NodeRefList &all_inputs) const override {
        PYBIND11_OVERLOAD_PURE(
            NodeRef,
            BaseNode,
            apply_chain_rule,
            wrt_input, d_target_wrt_this, all_inputs);
    };

    std::string to_string() const override {
        PYBIND11_OVERLOAD_PURE(
            std::string,
            BaseNode,
            to_string);
    };

    NodeRefList inputs() const override {
        PYBIND11_OVERLOAD_PURE(
            NodeRefList,
            BaseNode,
            inputs);
    };

};


template<typename Op>
NodeRef StraightBinaryOp(const NodeRef &a, const NodeRef &b) {
    return std::static_pointer_cast<BaseNode>(
        std::make_shared<BinaryOp<Op>>(a, b));
}

NodeRef matmul(const NodeRef &a, const NodeRef &b,
               const bool transpose_left = false,
               const bool transpose_right = false) {
    return std::static_pointer_cast<BaseNode>(
        std::make_shared<BinaryOp<MatMul>>(
            a, b, transpose_left, transpose_right));
}

ArrayType dtype_to_avalanche_array_type(const py::dtype &dtype) {
    ArrayType array_type;
    switch (dtype.kind()) {
        case 'f':
        case 'd':
        case 'e':
            // identify dtype
            switch (dtype.itemsize()) {
                case 2:
                    array_type = ArrayType::float16;
                    break;
                case 4:
                    array_type = ArrayType::float32;
                    break;
                case 8:
                    array_type = ArrayType::float64;
                    break;
                default:
                    throw std::invalid_argument("Unsupported float type size");
            }
            break;
        case 'i':
        case 'l':
        case 'q':
            switch (dtype.itemsize()) {
                case 1:
                    array_type = ArrayType::int8;
                    break;
                case 2:
                    array_type = ArrayType::int16;
                    break;
                case 4:
                    array_type = ArrayType::int32;
                    break;
                case 8:
                    array_type = ArrayType::int64;
                    break;
                default:
                    throw std::invalid_argument("Unsupported int type size");
            }
            break;
        default:
            throw std::invalid_argument("Unsupported array kind");
    }
    return array_type;
}


using ContextInitArrayType = float;
using ContextInitArray = py::array_t<
    ContextInitArrayType,
    py::array::forcecast | py::array::c_style>;


std::vector<ShapeDim> convert_shape_to_avalanche(
        const std::vector<ssize_t> &shape) {
    std::vector<ShapeDim> result(shape.size());
    std::copy(shape.begin(), shape.end(), result.begin());
    return result;
}

std::vector<ssize_t> convert_shape_from_avalanche(
        const std::vector<ShapeDim> &shape) {
    std::vector<ssize_t> result(shape.size());
    std::copy(shape.begin(), shape.end(), result.begin());
    return result;
}


/** Initializes context with automatic conversion from given numpy array
 * to whatever the node requires */
template <typename T>
MultiArrayRef init_context_with_cast(
        const NodeRef &node, ContextRef &context, ContextInitArray &data) {
    std::vector<T> tmp_copy(static_cast<std::size_t>(data.size()));
    py::buffer_info info = data.request();
    std::copy((const ContextInitArrayType *)info.ptr,
              ((const ContextInitArrayType *)info.ptr) + data.size(),
              tmp_copy.begin());
    auto dims = convert_shape_to_avalanche(info.shape);
    return context->init(node, tmp_copy, Shape(dims));
}

/**
 * Initializes context, specialized version for case when dtype of the given
 * numpy array exactly matches the one of the node.
 */
template<>
MultiArrayRef init_context_with_cast<ContextInitArrayType>(
    const NodeRef &node, ContextRef &context, ContextInitArray &data) {
    py::buffer_info info = data.request();
    auto dims = convert_shape_to_avalanche(info.shape);
    return context->init(
        node, info.ptr, static_cast<std::size_t>(data.nbytes()),
        dtype_of_static_type<ContextInitArrayType>, Shape(dims));
}

ARRAY_DTYPE_SWITCH_FUNCTION(switch_init_context_with_cast, init_context_with_cast, MultiArrayRef,)

MultiArrayRef
init_context(ContextRef &context, const NodeRef &node, ContextInitArray data) {
    ArrayType required_array_type = node->dtype();
    return switch_init_context_with_cast(required_array_type, node, context, data);
}

template <typename T>
py::array array_to_numpy_template(MultiArrayRef &array) {
    auto dims = convert_shape_from_avalanche(array->shape().dims());
    py::array_t<T, py::array::c_style> result(dims);
    auto info = result.request(true);
    array->wait_until_ready();
    auto reading_is_done = (
        array->buffer_when_ready()
             ->read_data(info.ptr,
                         static_cast<std::size_t>(info.size * info.itemsize),
                         0));
    reading_is_done.wait();
    return result;
}

ARRAY_DTYPE_SWITCH_FUNCTION(array_to_numpy_switch, array_to_numpy_template, py::array,)


template <typename T>
MultiArrayRef numpy_array_to_multi_array(
        BufferPoolRef pool,
        py::array_t<T, py::array::c_style | py::array::forcecast> array) {
    py::buffer_info info = array.request(false);
    Shape shape(convert_shape_to_avalanche(info.shape));
    auto result = pool->make_array(shape, dtype_of_static_type<T>);
    auto writing_is_done = result->buffer_when_ready()->write_data(
        info.ptr, static_cast<std::size_t>(info.size * info.itemsize),
        0);
    return result;
}

ARRAY_DTYPE_SWITCH_FUNCTION(numpy_to_multi_array_switch, numpy_array_to_multi_array, MultiArrayRef,)


py::array array_to_numpy(MultiArrayRef &array) {
    ArrayType required_dtype = array->dtype();
    return array_to_numpy_switch(required_dtype, array);
}


Initializer numpy_value_initializer(py::array value) {
    Initializer initializer = {
        [value](Context &context, ExecutionCache &cache,
                ArrayRefList &dependencies) mutable {
            py::buffer_info info = value.request(false);
            if (!(value.flags() & py::array::c_style)) {
                throw std::invalid_argument(
                    "Only c-style arrays are supported");
            }
            Shape shape(convert_shape_to_avalanche(info.shape));
            ArrayType array_type = dtype_to_avalanche_array_type(value.dtype());
            auto result = context.device_pool()->make_array(shape, array_type);
            auto writing_is_done = result->buffer_when_ready()->write_data(
                info.ptr, static_cast<std::size_t>(info.size * info.itemsize),
                0);
            result->wait_until_ready();
            return result;
        },
        nullptr,
        dtype_to_avalanche_array_type(value.dtype()),
        {}
    };
    return initializer;
}


NodeRef make_constant_from_numpy(py::array value, const std::string &name) {
    py::buffer_info info = value.request(false);
    if (!(value.flags() & py::array::c_style)) {
        throw std::invalid_argument("Only c-style arrays are supported");
    }
    Shape shape(convert_shape_to_avalanche(info.shape));
    ArrayType array_type = dtype_to_avalanche_array_type(value.dtype());
    return Constant::tensor(
        name,
        info.ptr,
        static_cast<std::size_t>(info.size * info.itemsize),
        array_type,
        shape);
}


NodeRef cast(const NodeRef &value, ArrayType dtype) {
    return FU<Cast>(value, dtype);
}


NodeRef py_slice_node(const NodeRef &node, py::handle x, ShapeDim current_axis) {
    Py_ssize_t start, stop, step;
    PySlice_Unpack(x.ptr(), &start, &stop, &step);
    if (step != 1) {
        throw std::invalid_argument(
            "Slicing is currently supported only "
            "for ranges with step equal 1");
    }
    if (stop != PY_SSIZE_T_MAX || start != 0) {
        ShapeDim range_start = static_cast<ShapeDim>(start);
        ShapeDim range_stop = static_cast<ShapeDim>(
            stop == PY_SSIZE_T_MAX ? -1 : stop);
        return FU<SliceAxis>(node, current_axis, range_start, range_stop);
    }
    return node;
}


PYBIND11_MODULE(pyvalanche, m) {
    m.doc() = R"pbdoc(
        Avalanche ML framework
        -----------------------
        .. currentmodule:: pyvalanche
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    auto shape_class = py::class_<Shape>(m, "Shape")
        .def(py::init<const std::vector<ShapeDim> &>())
        .def(py::init<ShapeDim>())
        .def("__str__", &Shape::to_string)
        .def("dim", &Shape::dim)
        .def_property_readonly("dims", &Shape::dims)
        .def_property_readonly("size", &Shape::size)
        .def_property_readonly("rank", &Shape::rank)
        .def("is_scalar", &Shape::is_scalar)
        .def("reshape", &Shape::reshape)
        .def("__eq__", &Shape::operator==)
        .def("__ne__", &Shape::operator!=);
    shape_class.attr("UnknownDim") = UnknownDim;

    py::implicitly_convertible<ShapeDim, Shape>();

    py::enum_<ArrayType>(m, "ArrayType")
        .value("int8", ArrayType::int8)
        .value("int16", ArrayType::int16)
        .value("int32", ArrayType::int32)
        .value("int64", ArrayType::int64)
        .value("float16", ArrayType::float16)
        .value("float32", ArrayType::float32)
        .value("float64", ArrayType::float64)
        .export_values();

    py::class_<BaseNode, NodeRef>(m, "BaseNode", py::dynamic_attr())
        .def(py::init(&Constant::scalar<float>))
        .def(py::init([](long value) {
            return Constant::scalar(ArrayType::float32, static_cast<float>(value));
        }))
        .def("__str__", &BaseNode::to_string)
        .def("__repr__", &BaseNode::repr)
        .def("tree_repr", &BaseNode::tree_repr)
        .def("inputs", &BaseNode::inputs)
        .def_property_readonly("dtype", &BaseNode::dtype)
        .def_property_readonly("shape", &BaseNode::shape)
        .def("__sub__", [](const NodeRef &node, const NodeRef &other) -> NodeRef {
            return F<Minus>(node, other);
        })
        .def("__add__", [](const NodeRef &node, const NodeRef &other) -> NodeRef {
            return F<Plus>(node, other);
        })
        .def("__mul__", [](const NodeRef &node, const NodeRef &other) -> NodeRef {
            return F<Multiply>(node, other);
        })
        .def("__truediv__", [](const NodeRef &node, const NodeRef &other) -> NodeRef {
            return F<Divide>(node, other);
        })
        .def("__rmul__", [](const NodeRef &node, float value) -> NodeRef {
            return FU<Scale>(node, value);
        })
        .def("__rsub__", [](const NodeRef &node, float value) -> NodeRef {
            return F<Minus>(Constant::scalar(value), node);
        })
        .def("__getitem__", [](const NodeRef &node, ShapeDim axis) {
            return FU<SliceAxis>(node, 0, axis, axis, false);
        })
        .def("__getitem__", [](const NodeRef &node, py::slice single_slice) {
            return py_slice_node(node, single_slice, 0);
        })
        .def("__getitem__", [](const NodeRef &node, py::tuple many_slices) {
            ShapeDim current_axis = 0;
            NodeRef output = node;
            for (auto x: many_slices) {
                if (PySlice_Check(x.ptr())) {
                    output = py_slice_node(output, x, current_axis);
                } else if (x.ptr() != Py_Ellipsis) {
                    throw std::invalid_argument(
                        "Slicing is currently supported only for ranges");
                }
                current_axis += 1;
            }
            return output;
        });
        ;
    py::implicitly_convertible<float, BaseNode>();
    py::implicitly_convertible<long, BaseNode>();

    py::class_<MultiArray, MultiArrayRef>(m, "MultiArray")
        .def("asnumpy", &array_to_numpy);

    py::class_<Context, ContextRef>(m, "Context")
        .def_static("make_for_device", &Context::make_for_device)
        .def("init", &init_context)
        .def("eval", &Context::eval);

    py::class_<Executor>(m, "Executor")
        .def(py::init<const ContextRef&, const NodeRefList&>())
        .def(py::init<const ContextRef&, const NodeRefList&, const NodeRefList&>())
//        .def("run", &Executor::run)
        .def("run", [](Executor &executor, py::dict cache_initial_values) {
            NodeValueMap node_value_map;
            for (auto &item: cache_initial_values) {
                if (!py::isinstance<BaseNode>(item.first)) {
                    throw std::invalid_argument("Only nodes can be the keys");
                }
                auto node = item.first.cast<NodeRef>();
                if (py::isinstance<py::array>(item.second)
                        || py::isinstance<py::list>(item.second)
                        || py::isinstance<py::float_>(item.second)
                        || py::isinstance<py::int_>(item.second)) {
                    auto gpu_array = numpy_to_multi_array_switch(
                        node->dtype(),
                        executor.context()->device_pool(),
                        item.second.cast<py::array>());
                    node_value_map[node] = gpu_array;
                    gpu_array->wait_until_ready();
                } else if (py::isinstance<MultiArray>(item.second)) {
                    node_value_map[node] = item.second.cast<MultiArrayRef>();
                } else {
                    throw std::invalid_argument(
                        "Only scalars, numpy arrays or MultiArrays are allowed"
                        "as values");
                }
            }
            return executor.run(node_value_map);
        });

    py::class_<DeviceInfo>(m, "DeviceInfo")
        .def_readwrite("name", &DeviceInfo::name)
        .def_readwrite("platform", &DeviceInfo::platform)
        .def_readwrite("id", &DeviceInfo::id);

    py::class_<CLMemoryManager, MemoryManagerRef >(m, "MemoryManager")
        .def_property_readonly("num_devices", &CLMemoryManager::num_devices)
        .def("device_info", &CLMemoryManager::device_info)
        .def_property_readonly("all_devices", &CLMemoryManager::list_devices);

    py::class_<Initializer>(m, "Initializer");

    m.def("default_memory_manager", CLMemoryManager::get_default);

    m.def("build_back_propagation_graph", &build_back_propagation_graph);

    m.def("variable",
          &Variable::make,
          "Creates a new variable",
          py::arg("name"), py::arg("shape_dims"), py::arg("dtype"),
          py::arg("initializer"));
    m.def("placeholder",
          [](const std::string &name,
             const ShapeDimList &shape_dims,
             ArrayType dtype) {
              return Placeholder::make(name, shape_dims, dtype);
          },
          "Creates a new placeholder");
    m.def("placeholder_with_initializer",
          py::overload_cast<
              const std::string&,
              const ShapeDimList&,
              ArrayType,
              Initializer>(&Placeholder::make),
          "Creates a new placeholder with a default initalizer");

    m.def("variable_from_node",
          &Variable::make_from_node,
          "Creates a new variable initialized from a different "
          "computational node (tensor)",
          py::arg("name"), py::arg("initializer"));


    m.def("random_uniform", &UniformRandom::make);

    m.def("value_initializer", &numpy_value_initializer);
    m.def("gradients", &build_gradients);

    m.def_submodule("consts", "Available constants")
        .def("zeros", &Constant::zeros)
        .def("zeros_like_with_type", &Constant::zeros_like_with_type)
        .def("ones_like_with_type", &Constant::ones_like_with_type)
        .def("ones", &Constant::ones)
        .def("from_array", &make_constant_from_numpy, "Creates a new constant");


    py::module ops = m.def_submodule("ops", "Available operations");

#define REDUCE_ARGS \
        py::arg_v("node", "input tensor"), \
        py::arg_v("reduce_axis", std::vector<ShapeDim>(), "axis to reduce"), \
        py::arg_v("keep_dims", false, \
                  "True/False on should we keep the reduced dimensions or not")

    ops
        .def("relu", &FU<ReLU>)
        .def("tanh", &FU<Tanh>)
        .def("sigmoid", &FU<Sigmoid>)
        .def("log", &FU<Log>)
        .def("exp", &FU<Exp>)
        .def("square", &FU<Square>)
        .def("sqrt", &FU<Sqrt>)
        .def("cast", &cast)
        .def("scale_pow", [](const NodeRef &input, float scale, float power) {
            return FU<SPower>(input, scale, power);
        })
        .def("pow", &StraightBinaryOp<Power>)
        .def("plus", &StraightBinaryOp<Plus>,
             "Elem-wise addition with broadcasting")
        .def("minus", &StraightBinaryOp<Minus>,
             "Elem-wise subtraction with broadcasting")
        .def("divide", &StraightBinaryOp<Divide>,
             "Elem-wise division with broadcasting")
        .def("multiply", &StraightBinaryOp<Multiply>,
             "Elem-wise multiplication with broadcasting")
        .def("equal", &StraightBinaryOp<Equal>,
             "Elem-wise equality check with broadcasting")
        .def("not_equal", &StraightBinaryOp<NotEqual>,
             "Elem-wise inequality check with broadcasting")
        .def("less", &StraightBinaryOp<Less>,
             "Elem-wise truth value of (x < y) with broadcasting")
        .def("less_equal", &StraightBinaryOp<LessEqual>,
             "Elem-wise truth value of (x <= y) with broadcasting")
        .def("greater", &StraightBinaryOp<Greater>,
             "Elem-wise truth value of (x > y) with broadcasting")
        .def("greater_equal", &StraightBinaryOp<GreaterEqual>,
             "Elem-wise truth value of (x >= y) with broadcasting")
        .def("less", &StraightBinaryOp<Less>,
             "Elem-wise truth value of (x < y) with broadcasting")
        .def("less_equal", &StraightBinaryOp<LessEqual>,
             "Elem-wise truth value of (x <= y) with broadcasting")
        .def("update", &StraightBinaryOp<Update>,
             "In-place update (assignment) of a variable")
        .def("update_add", &StraightBinaryOp<UpdateAdd>,
             "In-place addition like +=")
        .def("update_sub", &StraightBinaryOp<UpdateSub>,
             "In-place subtraction like -=")
        .def("binary_crossentropy", &StraightBinaryOp<BinaryCrossEntropy>,
             "In-place subtraction like -=")
        .def("matmul", &matmul,
             py::arg_v("a", "Left matrix"),
             py::arg_v("b", "Right matrix"),
             py::arg_v("transpose_left", false,
                       "set to True if the first matrix needs "
                       "to be transposed before multiplication"),
             py::arg_v("transpose_right", false,
                       "set to True if the second matrix needs "
                       "to be transposed before multiplication"))
        .def("softmax", &softmax,
             py::arg_v("node", "Input tensor"),
             py::arg_v("axis", -1, "Dimension to perform on"))
        .def("reshape", &FU<Reshape, const Shape&>)
        .def("reshape_like", py::overload_cast<const NodeRef&, const NodeRef&>(&ReshapeLike::make))
        .def("reshape_like", py::overload_cast<const NodeRef&, const NodeRef&, const ShapeDimList&>(&ReshapeLike::make))
        .def("shape", &ShapeOf::make)
        .def("expand_dims", &FU<ExpandDims, ShapeDim>)
        .def("squeeze", &FU<Squeeze, ShapeDim>)
        .def("cond", py::overload_cast<const NodeRef&, CondExpression, CondExpression>(&Cond::make))
        .def("cond", py::overload_cast<const NodeRef&, const NodeRef&, const NodeRef&>(&Cond::make))
        .def("cond", py::overload_cast<const NodeRef&, CondExpression, const NodeRef&>(&Cond::make))
        .def("cond", py::overload_cast<const NodeRef&, const NodeRef&, CondExpression>(&Cond::make))
        .def("tile", &FU<Tile, const std::vector<ShapeDim>&>)
        .def("concatenate", &Concatenate::make)
        .def("stack", &stack_nodes)
        .def("reduce_sum", &FU<ReduceSum, std::vector<ShapeDim>, bool>, REDUCE_ARGS)
        .def("reduce_mean", &FU<ReduceMean, std::vector<ShapeDim>, bool>, REDUCE_ARGS)
        .def("reduce_prod", &FU<ReduceMean, std::vector<ShapeDim>, bool>, REDUCE_ARGS);
#undef REDUCE_ARGS

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

} // namespace
