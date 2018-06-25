#include <random>

#include <fmt/format.h>

#include "avalanche/random_nodes.h"
#include "avalanche/Context.h"
#include "avalanche/ExecutionCache.h"
#include "avalanche/CodeCache.h"
#include "avalanche/MultiArray.h"
#include "avalanche/opencl_utils.h"


namespace avalanche {

constexpr char cl_sources_of_random_generators[] = {
#include "avalanche/kernels/random_generator.hex"
};

constexpr std::size_t WORK_GROUP_SIZE = 64;

cl::Program load_random_generators_program(cl::CommandQueue &queue) {
    auto context = get_context_from_queue(queue);
    return CodeCache::get_default().get_program(
        context,
        queue,
        "random_generators",
        cl_sources_of_random_generators,
        "");
}


MultiArrayRef
UniformRandom::generate_uniform_random(const MultiArrayRef &seeds) const {
    auto pool = seeds->buffer_unsafe()->pool();
    auto output = pool->make_array(seeds->shape(), dtype());
    auto queue = pool->cl_queue();
    auto program = load_random_generators_program(queue);
    cl::Kernel kernel(program, cached_generate_uniform_kernel_name().c_str());
    kernel.setArg(0, seeds->cl_buffer_unsafe());
    kernel.setArg(1, output->cl_buffer_unsafe());
    kernel.setArg(2, static_cast<cl_ulong>(seeds->size()));
    auto min_value_casted = cast_to_value_of_array_type(dtype(), _min_value),
         max_value_casted = cast_to_value_of_array_type(dtype(), _max_value);
    kernel.setArg(3, array_type_size(dtype()), &min_value_casted);
    kernel.setArg(4, array_type_size(dtype()), &max_value_casted);
    std::vector<cl::Event> wait_for_events;
    wait_for_events.push_back(seeds->buffer_unsafe()->completion_event());
    output->add_dependencies({seeds});
    const auto work_items = make_divisible_by(WORK_GROUP_SIZE, seeds->size());
    cl::Event result_event;
    queue.enqueueNDRangeKernel(
        kernel, cl::NullRange,
        cl::NDRange(work_items), cl::NDRange(WORK_GROUP_SIZE),
        &wait_for_events, &result_event);
    output->set_completion_event(result_event);
    return output;
}

void seed_uniform_random(MultiArrayRef &seeds, std::uint64_t base_seed) {
    auto pool = seeds->buffer_unsafe()->pool();
    auto queue = pool->cl_queue();
    auto program = load_random_generators_program(queue);
    using KernelType = cl::KernelFunctor<const cl::Buffer&, cl_ulong, cl_ulong>;
    KernelType kernel(program, "seed_uniform_random");
    CLBufferRef source_buffer = seeds->buffer_unsafe();
    std::vector<cl::Event> wait_for_events;
    const auto work_items = make_divisible_by(WORK_GROUP_SIZE, seeds->size());
    cl::Event generation_is_done = kernel(
        cl::EnqueueArgs(queue,
                        wait_for_events,
                        cl::NDRange(work_items),
                        cl::NDRange(WORK_GROUP_SIZE)),
        source_buffer->cl_buffer_unsafe(),
        static_cast<cl_ulong>(seeds->size()),
        static_cast<cl_ulong>(base_seed));
    seeds->set_completion_event(generation_is_done);
}

avalanche::MultiArrayRef
avalanche::UniformRandom::eval(avalanche::Context &context,
                               avalanche::ExecutionCache &cache) const {
    MultiArrayRef cached_value, seeds;
    if (!context.get(id, seeds)) {
        seeds = context.device_pool()->make_array(shape(), ArrayType::int64);
        seeds->set_label(to_string());
        context.init(id, seeds);
        seed_uniform_random(seeds, _seed);
    }
    if (!cache.get(id, cached_value)) {
        cached_value = generate_uniform_random(seeds);
        cache.put(id, cached_value);
    }
    return cached_value;
}


std::string avalanche::UniformRandom::to_string() const {
    return fmt::format("UniformRandom({}, {})", _min_value, _max_value);
}

std::string avalanche::UniformRandom::repr() const {
    return format_repr("UniformRandom", "", "");
}

const std::string& UniformRandom::cached_generate_uniform_kernel_name() const {
    if (_generate_uniform_kernel_name.empty()) {
        _generate_uniform_kernel_name = fmt::format(
            "generate_uniform_random_{}", array_type_name(dtype()));
    }
    return _generate_uniform_kernel_name;
}

UniformRandom::UniformRandom(const Shape &shape, double min_value,
                             double max_value, ArrayType dtype,
                             std::uint64_t seed)
    :_min_value{min_value}, _max_value{max_value}, _seed{seed}
{
    set_shape(shape);
    set_dtype(dtype);
    if (!_seed) {
        std::random_device rd;
        _seed = rd();
    }
}


} // namespace
