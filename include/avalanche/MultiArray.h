#ifndef AVALANCHE_MULTIARRAY_H
#define AVALANCHE_MULTIARRAY_H

#include <memory>
#include <initializer_list>
#include <future>

#include "avalanche/CLMemoryManager.h"
#include "avalanche/CLBufferPool.h"
#include "avalanche/Shape.h"
#include "avalanche/CLBuffer.h"
#include "avalanche/ArrayType.h"

namespace avalanche {

class MultiArray;

using MultiArrayRef = std::shared_ptr<MultiArray>;
using ArrayRefList = std::vector<MultiArrayRef >;


/**
 * A high-level wrapper above plain OpenCL buffers, giving them
 * Shape and Data Type (dtype).
 */
class MultiArray {
public:

    // Creates a new multi-array sharing the same buffer and the same
    // promise object.
    MultiArrayRef reshape(const std::vector<ShapeDim> &shape_dims) {
        return std::shared_ptr<MultiArray>(
            new MultiArray(_buffer, _shape.reshape(shape_dims), _dtype));
    }

    std::size_t size() const { return _shape.size(); }
    const Shape& shape() const { return _shape; }
    ArrayType dtype() const { return _dtype; }
    /** Associates given OpenCL event with the readiness of the arrray */
    void set_completion_event(cl::Event &event);
    /** Associates given OpenCL event with the readiness of the arrray */
    void set_completion_event(cl_event event);
    /** Returns OpenCL buffer attached to this array, but only after
     * it has been marked as "ready" */
    const std::shared_ptr<CLBuffer>& buffer_when_ready() const {
        _buffer->wait_until_ready();
        return _buffer;
    };
    void wait_until_ready() const { _buffer->wait_until_ready(); }
    const cl::Buffer& cl_buffer() { return _buffer->cl_buffer_when_ready(); }
    const cl::Buffer& cl_buffer_unsafe() { return _buffer->cl_buffer_unsafe(); }

    /** Returns OpenCL buffer storing the array's data without checking
     * if the data are ready. */
    std::shared_ptr<CLBuffer>& buffer_unsafe() {
        return _buffer;
    };
    std::string to_string();
    /** As soon as the original array src is ready, creates a new
     * array containing exactly the same data. */
    MultiArrayRef ref_copy();
    /** Creates a new array with the same buffer underneath, but a new readiness
     * promise, which can be set independently. */
    MultiArrayRef ref_with_shared_buffer() {
        return std::shared_ptr<MultiArray>(
            new MultiArray(_buffer, _shape, _dtype));
    }


    /**
     * To keep the buffers alive until the computation is done we add them
     * as dependencies to the others.
     * It's better to do this later in the code, because we don't want
     * the compiler to destroy the buffers after their last mentioning,
     * which can lead  to troubles like accidental re-use of the same
     * OpenCL buffer for adjacent operations.
     */
    void add_dependencies(std::initializer_list<CLBufferRef> dependencies);
    void add_dependencies(std::initializer_list<MultiArrayRef> dependencies);
    void set_label(const std::string &label) { _buffer->set_label(label); }
    void set_label(const char *func, int line) { _buffer->set_label(func, line); }
    void set_label(const std::string &func, int line) { _buffer->set_label(func, line); }

    template <typename T>
    void fetch_data_into(std::vector<T> &data) const {
        wait_until_ready();
        if (_dtype != dtype_of_static_type<T>) {
            throw std::invalid_argument(
                "The vector for storing the data is incompatible "
                "with the type of the array.");
        }
        auto reading_is_done = buffer_when_ready()->read_into_vector(data);
        reading_is_done.wait();
    }

    template <typename T>
    void write_from_vector(const std::vector<T> &data) const {
        wait_until_ready();
        if (_dtype != dtype_of_static_type<T>) {
            throw std::invalid_argument(
                "The data vector is incompatible "
                    "with the type of the array.");
        }
        auto writing_is_done = buffer_when_ready()->write_from_vector(data);
        writing_is_done.wait();
    }

    static MultiArrayRef from_buffer(
            const CLBufferRef &buffer,
            const Shape &shape,
            const ArrayType dtype) {
        return std::shared_ptr<MultiArray>(
            new MultiArray(buffer, shape, dtype));
    }

    static MultiArrayRef make(BufferPoolRef device_pool, Shape shape,
                              ArrayType dtype) {
        return from_buffer(
            device_pool->reserve_buffer(array_type_size(dtype) * shape.size()),
            shape,
            dtype);
    }

    static MultiArrayRef make(DeviceIndex device_idx, Shape shape, ArrayType dtype) {
        return from_buffer(
            CLMemoryManager::get_default()
                ->buffer_pool(device_idx)
                ->reserve_buffer(array_type_size(dtype) * shape.size()),
            shape,
            dtype);
    }


private:
    MultiArray(
        const CLBufferRef &buffer,
        const Shape &shape,
        const ArrayType dtype)
        : _buffer{buffer},
          _shape{shape},
          _dtype{dtype}
    {}

    std::shared_ptr<CLBuffer> _buffer;
    Shape _shape;
    ArrayType _dtype;
};

} // namespace

// All operations must return futures
// Even filling the array with data must be just an operation, also returning future
// Along with the future the shape has to be returned

#endif //AVALANCHE_MULTIARRAY_H
