#include <sstream>
#include <iostream>

#include "avalanche/CLBufferPool.h"
#include "avalanche/CodeCache.h"
#include "avalanche/opencl_utils.h"

namespace avalanche {


static CodeCache default_code_cache;

cl::Program
CodeCache::get_program(const cl::Context &context, const cl::Device &device,
                       const std::string &program_name,
                       const std::string &source,
                       const std::string &extra_options) {
    cl::Program program;
    ProgramCacheKey program_cache_key =
        std::make_tuple(context(), device(), program_name);
    auto search = program_cache_.find(program_cache_key);
    if (search != program_cache_.end()) {
        program = search->second;
    } else {
        std::string device_name;
        device.getInfo(CL_DEVICE_NAME, &device_name);
        BinaryCacheKey binary_cache_key =
            std::make_tuple(device_name, program_name);
        auto search = binary_cache_.find(binary_cache_key);
        if (search == binary_cache_.end()) {
            program = cl::Program(context, source, false);
            std::string program_options("-cl-std=CL1.2 ");
            program_options += extra_options;
            try {
                program.build({device}, program_options.c_str());
            } catch (cl::Error &e) {
                if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
                    std::ostringstream full_log;
                    cl_build_status status =
                    program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
                    std::string name = device.getInfo<CL_DEVICE_NAME>();
                    std::string buildlog = (
                        program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
                    full_log << "Build log for " << name << ":\n"
                        << buildlog << std::endl;
                    std::cerr << full_log.str();
                }
                throw;
            }
            size_t binary_size;
            clGetProgramInfo(
                program(), CL_PROGRAM_BINARY_SIZES,
                sizeof(binary_size), &binary_size, nullptr);
            std::vector<char> binary_data(binary_size);
            char *program_pointers[1] = {binary_data.data()};
            clGetProgramInfo(
                program(), CL_PROGRAM_BINARIES,
                sizeof(program_pointers), &program_pointers, nullptr);
            binary_cache_[binary_cache_key] = std::move(binary_data);
        } else {
            auto &binary = search->second;
            size_t binary_size = binary.size();
            cl_device_id cl_device = device();
            const unsigned char *binary_ptr = (unsigned char *)binary.data();
            program = clCreateProgramWithBinary(
                context(), 1, &cl_device, &binary_size, &binary_ptr,
                nullptr, nullptr);
            program.build();
        }
        program_cache_[program_cache_key] = program;
    }
    return program;
}

cl::Program CodeCache::get_program(const cl::Context &context,
                                   const cl::CommandQueue &queue,
                                   const std::string &program_name,
                                   const std::string &source,
                                   const std::string &extra_options) {
    return get_program(context, get_device_from_queue(queue),
                       program_name, source, extra_options);
}

CodeCache &CodeCache::get_default() {
    return default_code_cache;
}

} // namespace
