cmake_minimum_required(VERSION 3.22)
project(astroflow)

set(TARGET_NAME _astroflow_core)
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)

# 启用CUDA支持
enable_language(CUDA)
find_package(pybind11 REQUIRED)
find_package(GTest REQUIRED)

# 添加CUDA编译选项
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -g -pg")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -pg")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp -Xcompiler -fno-math-errno")
# Release 模式配置
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -fopenmp -fno-math-errno")

file(GLOB SRC_BIND "src/bindpy/*.cpp" "src/bindpy/*.cu")
file(GLOB TESTS "tests/*.cpp")
file(GLOB SRC_DATA CONFIGURE_DEPENDS "src/data/*.cpp" "src/data/*.cu")
file(GLOB SRC_CORE CONFIGURE_DEPENDS "src/core/*.cpp" "src/core/*.cu")
file(GLOB SRC_DETE CONFIGURE_DEPENDS "src/detection/*.cpp" "src/detection/*.cu")
file(GLOB SRC_RFI CONFIGURE_DEPENDS "src/rfi/*.cpp" "src/rfi/*.cu")

set(SRC_FILES ${SRC_DATA} ${SRC_CORE} ${SRC_UTILS} ${SRC_DETE} ${SRC_RFI})

include_directories(include)

add_library(${TARGET_NAME} SHARED  ${SRC_FILES} ${SRC_BIND})

# 链接依赖
target_link_libraries(${TARGET_NAME}
    PRIVATE 
        pybind11::module
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

if(CMAKE_CUDA_COMPILER)
    target_include_directories(${TARGET_NAME} PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_options(${TARGET_NAME} PRIVATE
        -I${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

# 设置Python模块输出路径
set_target_properties(${TARGET_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../python
    PREFIX ""
    SUFFIX ".so"
)


#------------------------------------------------------------------#
# 测试可执行文件名称
set(TEST_EXECUTABLE_NAME astroflow_test)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 添加测试可执行文件
add_executable(${TEST_EXECUTABLE_NAME} ${TESTS} ${SRC_FILES} ${SRC_BIND})

target_include_directories(${TEST_EXECUTABLE_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 链接 GoogleTest 
target_link_libraries(${TEST_EXECUTABLE_NAME} PRIVATE
    gtest::gtest
    pybind11::embed
)

include(GoogleTest)
enable_testing()

# 自动发现并注册测试用例
gtest_discover_tests(${TEST_EXECUTABLE_NAME}
    EXTRA_ARGS --gtest_output=xml:${CMAKE_BINARY_DIR}/test_results.xml
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

