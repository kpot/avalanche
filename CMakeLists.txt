cmake_minimum_required(VERSION 3.3)

project(avalanche)

include(CTest)

if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

#set(BUILD_SHARED_LIBS OFF)
add_subdirectory(libs/pybind11)
add_subdirectory(libs/CLBlast EXCLUDE_FROM_ALL)
add_subdirectory(libs/fmt EXCLUDE_FROM_ALL)
add_subdirectory(libs/backward-cpp)

set(USE_OPENCL ON)

set(OPENCL_REQUIRED_DEFINITIONS
        -DCL_HPP_MINIMUM_OPENCL_VERSION=120
        -DCL_HPP_TARGET_OPENCL_VERSION=120
        -DCL_HPP_ENABLE_EXCEPTIONS
        -DCL_USE_DEPRECATED_OPENCL_2_0_APIS)


set(COMPILED_KERNELS_DIR ${CMAKE_CURRENT_BINARY_DIR}/include/avalanche/kernels)
file(MAKE_DIRECTORY ${COMPILED_KERNELS_DIR})
find_program(RESOURCE_COMPILER NAMES xxd)
file(GLOB NOT_COMPILED_RESOURCES "src/avalanche/kernels/*.opencl")
foreach(INPUT_FILE ${NOT_COMPILED_RESOURCES})
    get_filename_component(OUTPUT_VAR_NAME ${INPUT_FILE} NAME_WE)
    set(OUTPUT_FILE ${COMPILED_KERNELS_DIR}/${OUTPUT_VAR_NAME}.hex)
    add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            DEPENDS ${INPUT_FILE}
            COMMAND ${RESOURCE_COMPILER} -i < ${INPUT_FILE} > ${OUTPUT_FILE}
            COMMENT "Compiling ${INPUT_FILE} to binary")
    list(APPEND COMPILED_RESOURCES ${OUTPUT_FILE})
endforeach()
add_custom_target(avalanche_kernels SOURCES ${COMPILED_RESOURCES})

add_library(avalanche
        src/avalanche/logging.cpp
        src/avalanche/CLMemoryManager.cpp
        src/avalanche/CLBuffer.cpp
        src/avalanche/CLBufferPool.cpp
        src/avalanche/Shape.cpp
        src/avalanche/MultiArray.cpp
        src/avalanche/BaseNode.cpp
        src/avalanche/Context.cpp
        src/avalanche/ExecutionCache.cpp
        src/avalanche/terminal_nodes.cpp
        src/avalanche/base_ops_nodes.cpp
        src/avalanche/Executor.cpp
        src/avalanche/backprop.cpp
        src/avalanche/nodes.cpp
        src/avalanche/CodeCache.cpp
        src/avalanche/opencl_utils.cpp
        src/avalanche/casting.cpp
        src/avalanche/random_nodes.cpp
        src/avalanche/shape_nodes.cpp
        src/avalanche/conditional_nodes.cpp
        src/avalanche/math_ops/BroadcastedBinaryOp.cpp
        src/avalanche/math_ops/simple_arithemic.cpp
        src/avalanche/math_ops/MatMul.cpp
        src/avalanche/math_ops/const_transformation.cpp
        src/avalanche/math_ops/reductions.cpp
        src/avalanche/math_ops/comparisons.cpp
        src/avalanche/math_ops/updates.cpp
        src/avalanche/math_ops/ElemWiseBinaryOp.cpp
        src/avalanche/math_ops/losses.cpp
        ${BACKWARD_ENABLE})
target_compile_features(avalanche PRIVATE cxx_std_14)
target_link_libraries(avalanche
        PUBLIC ${OpenCL_LIBRARY} Threads::Threads
        PRIVATE clblast fmt-header-only)
target_include_directories(avalanche
        PUBLIC libs/cl_crossplatform include ${OpenCL_INCLUDE_DIRS}
            libs/spdlog/include)
target_include_directories(avalanche
        PUBLIC SYSTEM libs/cl_crossplatform)
target_include_directories(avalanche
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)
add_backward(avalanche)


target_compile_definitions(avalanche PUBLIC ${OPENCL_REQUIRED_DEFINITIONS})
# Allows to link with OpenCL without pointing to a specific SDK
set_target_properties(avalanche PROPERTIES SKIP_BUILD_RPATH ON)
add_dependencies(avalanche avalanche_kernels)


if(BUILD_TESTING)
    set(TESTS
            test_memory_manager
            test_math_ops
            test_opencl_libraries
            test_cl_buffer
            test_tree_evaluation
            test_issues
            test_matmul
            test_random_generators
            test_shape_transforms)

    foreach(TARGET ${TESTS})
        add_executable(${TARGET} test/${TARGET}.cpp)
        target_compile_features(${TARGET} PRIVATE cxx_std_14)
        target_include_directories(${TARGET} PRIVATE libs/Catch2)
        target_include_directories(${TARGET} PRIVATE ${TVM_INCLUDE_DIRS})
        target_link_libraries(${TARGET}
                PRIVATE avalanche fmt-header-only)
        # Allows to link with OpenCL without pointing to a specific SDK
        set_target_properties(${TARGET} PROPERTIES SKIP_BUILD_RPATH ON)
        add_test(NAME ${TARGET} COMMAND ${TARGET})
    endforeach()

    target_compile_definitions(test_opencl_libraries PUBLIC ${OPENCL_REQUIRED_DEFINITIONS})
    target_link_libraries(test_opencl_libraries PRIVATE ${OpenCL_LIBRARY} clblast)
    target_include_directories(test_opencl_libraries
            PUBLIC libs/cl_crossplatform include ${OpenCL_INCLUDE_DIRS})
endif()

pybind11_add_module(pyvalanche src/avalanche/python_binding.cpp)
target_compile_features(pyvalanche PRIVATE cxx_std_14)
target_link_libraries(pyvalanche PRIVATE ${OpenCL_LIBRARY} clblast avalanche)
