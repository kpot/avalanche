#ifndef AVALANCHE_PROGRAMCACHE_H
#define AVALANCHE_PROGRAMCACHE_H

#include <string>
#include <vector>
#include <map>

#include "CL_cust/cl2.hpp"

namespace avalanche {

using BinaryCacheKey = std::tuple<std::string, std::string>;
using BinaryCache = std::map<BinaryCacheKey, std::vector<char>>;
using ProgramCacheKey = std::tuple<
    const cl_context,
    const cl_device_id,
    const std::string>;
using ProgramCache = std::map<ProgramCacheKey, cl::Program>;

class CodeCache {
public:
    cl::Program get_program(const cl::Context &context,
                            const cl::Device &device,
                            const std::string &program_name,
                            const std::string &source,
                            const std::string &extra_options);

    cl::Program get_program(const cl::Context &context,
                            const cl::CommandQueue &queue,
                            const std::string &program_name,
                            const std::string &source,
                            const std::string &extra_options);

    static CodeCache& get_default();

private:
    // Stores all binaries in a map using CL_DEVICE_NAME`s value as a key
    BinaryCache binary_cache_;
    // Stores all programs in a map using both cl_context and cl_device_id as keys
    ProgramCache program_cache_;
};


} // namespace


#endif //AVALANCHE_PROGRAMCACHE_H
