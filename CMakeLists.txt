# ------------------------ 基础配置 ------------------------
cmake_minimum_required(VERSION 3.22)
project(filreader)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 导出编译命令（用于工具如 clangd）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置构建类型（Debug 或 Release）
#set(CMAKE_BUILD_TYPE "Debug")  # 默认使用 Debug 模式
set(CMAKE_BUILD_TYPE "Release")  # 如果需要 Release 模式，取消注释

# ------------------------ 源码文件配置 ------------------------
# 主程序源码
file(GLOB SOURCES "src/*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")  # 排除 main.cpp
file(GLOB MAIN "src/main.cpp")  # 主程序入口

# 测试源码
file(GLOB TESTS "tests/*.cpp")

# ------------------------ 依赖配置 ------------------------
# 引入 FetchContent 模块
include(FetchContent)

# GoogleTest 配置
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
FetchContent_MakeAvailable(googletest)

# Python 和 NumPy 配置
set(CONDA_ENV_PATH "$ENV{CONDA_PREFIX}")  # 自动获取 Conda 环境路径
set(Python3_ROOT_DIR ${CONDA_ENV_PATH})   # 设置 Python 根目录

# 查找 Python 开发环境
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# 获取 NumPy 头文件路径
execute_process(
  COMMAND python -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 打印调试信息
message(STATUS "Python3_ROOT_DIR: ${Python3_ROOT_DIR}")
message(STATUS "NUMPY_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")

# ------------------------ 编译选项配置 ------------------------
# Debug 模式配置
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -g -pg")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")

# Release 模式配置
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -fopenmp -fno-math-errno")

# ------------------------ 主程序配置 ------------------------
# 包含头文件目录
include_directories(include)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 添加主程序可执行文件
add_executable(fread ${SOURCES} ${MAIN})

# 为主程序设置包含目录
target_include_directories(fread PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
)

# 为主程序链接库
target_link_libraries(fread PRIVATE ${Python3_LIBRARIES})

# ------------------------ 测试配置 ------------------------
# 测试可执行文件名称
set(TEST_EXECUTABLE_NAME fread_test)

# 添加测试可执行文件
add_executable(${TEST_EXECUTABLE_NAME} ${TESTS} ${SOURCES})

# 为测试程序设置包含目录
target_include_directories(${TEST_EXECUTABLE_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
)

# 链接 GoogleTest 和 Python 库
target_link_libraries(${TEST_EXECUTABLE_NAME} PRIVATE
    gtest_main
    ${Python3_LIBRARIES}
)

# 启用测试
include(GoogleTest)
enable_testing()

# 自动发现并注册测试用例
gtest_discover_tests(${TEST_EXECUTABLE_NAME}
    EXTRA_ARGS --gtest_output=xml:${CMAKE_BINARY_DIR}/test_results.xml
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ------------------------ 动态库路径配置 ------------------------
# Linux 和 macOS 的 RPATH 配置
if(UNIX AND NOT APPLE)
    set(CMAKE_INSTALL_RPATH "${CONDA_ENV_PATH}/lib")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
elseif(APPLE)
    set(CMAKE_INSTALL_RPATH "${CONDA_ENV_PATH}/lib")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()
