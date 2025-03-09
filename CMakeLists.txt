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

file(GLOB DEMO "src/demo/*")
file(GLOB TEST "tests/*.cpp")
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "src/*")

include_directories(include)

add_library(${TARGET_NAME} SHARED  ${SRC_FILES})

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
