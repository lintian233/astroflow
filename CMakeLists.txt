cmake_minimum_required(VERSION 3.22)
project(astroflow)

set(TARGET_NAME _astroflow_core)
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)

# 启用CUDA支持
enable_language(CUDA)
find_package(OpenCV COMPONENTS core imgproc imgcodecs REQUIRED)
find_package(pybind11 REQUIRED)
find_package(GTest REQUIRED)

# 添加CUDA编译选项
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")

# GPU 编译优化配置
set(CUDA_OPTIMIZATION_FLAGS 
    "-O3"                           # 最高级别优化
    "--use_fast_math"               # 启用快速数学库
    "--ftz=true"                    # 将非正规数刷新为零
    "--prec-div=false"              # 使用快速除法
    "--prec-sqrt=false"             # 使用快速平方根
    "--fmad=true"                   # 启用融合乘加运算 (FMA)
    "--extra-device-vectorization"  # 额外设备向量化
    "--maxrregcount=64"             # 限制寄存器使用以提高占用率
    "-Xptxas=-v"                    # PTX汇编器详细输出
    "-Xptxas=-O3"                   # PTX级别的最高优化
    "--generate-line-info"          # 生成行信息用于调试
    "--relocatable-device-code=false" # 禁用可重定位设备代码以获得更好性能
)

# 内存访问优化
set(CUDA_MEMORY_FLAGS
    "-Xptxas=--allow-expensive-optimizations=true"  # 允许昂贵的优化
    "-Xcompiler=-O3"                                 # 主机代码优化
    "-Xcompiler=-march=native"                       # 针对本机架构优化
    "-Xcompiler=-fopenmp"                           # OpenMP支持
    "-Xcompiler=-fno-math-errno"                    # 禁用数学错误检查
    "-Xcompiler=-ffast-math"                        # 主机快速数学
    "-Xcompiler=-funroll-loops"                     # 循环展开
    "-Xcompiler=-fomit-frame-pointer"               # 省略帧指针
    "-Xcompiler=-pipe"                              # 使用管道加速编译
)

# 基础CUDA标志（不包含架构特定的gencode）
set(CUDA_BASE_FLAGS
    "--ptxas-options=-v"                # PTX汇编器选项
    "--ptxas-options=--warn-on-spills"  # 寄存器溢出警告
    "--expt-relaxed-constexpr"          # 允许在constexpr中使用扩展
    "--expt-extended-lambda"            # 扩展的lambda支持
    "--default-stream per-thread"       # 每线程默认流
)

# 调试模式配置
set(CMAKE_CUDA_FLAGS_DEBUG 
    "-O0 -g -G"                     # 无优化，启用调试符号和设备调试
    "-Xcompiler=-O0"
    "-Xcompiler=-g"
    "-lineinfo"                     # 生成行号信息
    "--ptxas-options=-v"
)

# Release模式CUDA配置 - 组合所有优化标志
set(CMAKE_CUDA_FLAGS_RELEASE "${CUDA_OPTIMIZATION_FLAGS} ${CUDA_MEMORY_FLAGS} ${CUDA_BASE_FLAGS}")
string(REPLACE ";" " " CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")

# 通用CUDA标志
set(CMAKE_CUDA_FLAGS "-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

# CPU编译优化 (主机代码)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -g -pg")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")

# CPU Release模式优化
set(CMAKE_CXX_FLAGS_RELEASE 
    "-Ofast"                        # 最高级别优化
    "-march=native"                 # 针对本机架构
    "-mtune=native"                 # 针对本机调优
    "-fopenmp"                      # OpenMP并行化
    "-fno-math-errno"               # 禁用数学错误检查
    "-ffast-math"                   # 快速数学
    "-funroll-loops"                # 循环展开
    "-fomit-frame-pointer"          # 省略帧指针
    "-pipe"                         # 使用管道
    "-mfma"                         # 启用FMA指令
    "-mavx2"                        # AVX2指令集
    "-msse4.2"                      # SSE4.2支持
    "-flto"                         # 链接时优化
    "-DNDEBUG"                      # 禁用断言
)
string(REPLACE ";" " " CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

# 链接器优化
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto -Wl,--as-needed -Wl,--gc-sections")

# 输出优化信息
message(STATUS "CUDA Optimization Flags: ${CMAKE_CUDA_FLAGS_RELEASE}")
message(STATUS "CXX Optimization Flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "Target CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

file(GLOB SRC_BIND "src/bindpy/*.cpp" "src/bindpy/*.cu")
file(GLOB TESTS "tests/*.cpp")
file(GLOB SRC_DATA CONFIGURE_DEPENDS "src/data/*.cpp" "src/data/*.cu")
file(GLOB SRC_CORE CONFIGURE_DEPENDS "src/core/*.cpp" "src/core/*.cu")
file(GLOB SRC_DETE CONFIGURE_DEPENDS "src/detection/*.cpp" "src/detection/*.cu")
file(GLOB SRC_RFI CONFIGURE_DEPENDS "src/rfi/*.cpp" "src/rfi/*.cu")

set(SRC_FILES ${SRC_DATA} ${SRC_CORE} ${SRC_UTILS} ${SRC_DETE} ${SRC_RFI})

include_directories(include)

add_library(${TARGET_NAME} SHARED  ${SRC_FILES} ${SRC_BIND})

# 设置库的编译属性
set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

# GPU特定编译选项
target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Release>:-O3;--use_fast_math;--fmad=true;--extra-device-vectorization>
        $<$<CONFIG:Debug>:-O0;-g;-G;-lineinfo>
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CONFIG:Release>:-Ofast;-march=native;-mfma;-mavx2;-flto>
        $<$<CONFIG:Debug>:-O0;-g;-ggdb>
        -fopenmp
        -fno-math-errno
    >
)

# 链接依赖
target_link_libraries(${TARGET_NAME}
    PRIVATE 
        pybind11::module
        opencv_core
        opencv_imgproc
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

# 设置测试可执行文件的编译属性
set_target_properties(${TEST_EXECUTABLE_NAME} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION OFF
)

# 为测试可执行文件应用相同的优化选项
target_compile_options(${TEST_EXECUTABLE_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Release>:-O3;--use_fast_math;--fmad=true;--extra-device-vectorization>
        $<$<CONFIG:Debug>:-O0;-g;-G;-lineinfo>
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CONFIG:Release>:-Ofast;-march=native;-mfma;-mavx2;-flto>
        $<$<CONFIG:Debug>:-O0;-g;-ggdb>
        -fopenmp
        -fno-math-errno
    >
)

target_include_directories(${TEST_EXECUTABLE_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 链接 GoogleTest 
target_link_libraries(${TEST_EXECUTABLE_NAME} PRIVATE
    gtest::gtest
    pybind11::embed
    opencv_core
    opencv_imgproc
)

include(GoogleTest)
enable_testing()

# 自动发现并注册测试用例
gtest_discover_tests(${TEST_EXECUTABLE_NAME}
    EXTRA_ARGS --gtest_output=xml:${CMAKE_BINARY_DIR}/test_results.xml
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

