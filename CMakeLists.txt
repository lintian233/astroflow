cmake_minimum_required(VERSION 3.22)
project(astroflow)

set(TARGET_NAME demo)
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)

# 启用CUDA支持
enable_language(CUDA)
find_package(pybind11 REQUIRED)
find_package(GTest REQUIRED)

# 添加CUDA编译选项
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 75)  

# 启用编译命令导出
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -g -pg")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")

# Release 模式配置
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -fopenmp -fno-math-errno")

file(GLOB DEMO "src/demo/*.cpp" "src/demo/*.cu")
file(GLOB TESTS "tests/*.cpp")
file(GLOB SRC_DATA CONFIGURE_DEPENDS "src/data/*.cpp" "src/data/*.cu")
file(GLOB SRC_CORE CONFIGURE_DEPENDS "src/core/*.cpp" "src/core/*.cu")
file(GLOB SRC_DETE CONFIGURE_DEPENDS "src/detection/*.cpp" "src/detection/*.cu")
file(GLOB SRC_RFI CONFIGURE_DEPENDS "src/rfi/*.cpp" "src/rfi/*.cu")

set(SRC_FILES ${SRC_DATA} ${SRC_CORE} ${SRC_UTILS} ${SRC_DETE} ${SRC_RFI})

include_directories(include)

add_library(${TARGET_NAME} SHARED  ${SRC_FILES} ${DEMO})

# 链接依赖
target_link_libraries(${TARGET_NAME}
    PRIVATE pybind11::module
)

# 设置CUDA包含路径
target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 设置Python模块输出路径
set_target_properties(${TARGET_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../python
    PREFIX ""
    SUFFIX ".so"
)

#TEST
#------------------------------------------------------------------#
# 测试可执行文件名称
set(TEST_EXECUTABLE_NAME astroflow_test)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 添加测试可执行文件
add_executable(${TEST_EXECUTABLE_NAME} ${TESTS} ${SRC_FILES})

target_include_directories(${TEST_EXECUTABLE_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 链接 GoogleTest 
target_link_libraries(${TEST_EXECUTABLE_NAME} PRIVATE
    gtest::gtest
)

include(GoogleTest)
enable_testing()

# 自动发现并注册测试用例
gtest_discover_tests(${TEST_EXECUTABLE_NAME}
    EXTRA_ARGS --gtest_output=xml:${CMAKE_BINARY_DIR}/test_results.xml
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

