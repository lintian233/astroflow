cmake_minimum_required(VERSION 3.22)

# ===================== Conan:  toolchain =====================
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  find_program(CONAN_CMD conan)
  if(CONAN_CMD)
    # 首次使用时生成 profile
    execute_process(COMMAND ${CONAN_CMD} profile list
                    RESULT_VARIABLE _has_profiles
                    OUTPUT_QUIET ERROR_QUIET)
    if(NOT _has_profiles EQUAL 0)
      execute_process(COMMAND ${CONAN_CMD} profile detect --force
                      COMMAND_ERROR_IS_FATAL ANY)
    endif()

    # 等价于：conan install .. --output-folder=. --build=missing -s build_type=Release -c ...
    # 使用当前 build type；scikit-build-core 会传入 -DCMAKE_BUILD_TYPE=Release(或RelWithDebInfo等)
    if(NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
    execute_process(
      COMMAND ${CONAN_CMD} install ${CMAKE_SOURCE_DIR}
              --output-folder=${CMAKE_BINARY_DIR}
              --build=missing
              -s build_type=${CMAKE_BUILD_TYPE}
              -c tools.system.package_manager:mode=install
              -c tools.system.package_manager:sudo=True
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMAND_ERROR_IS_FATAL ANY
    )

    if(EXISTS "${CMAKE_BINARY_DIR}/conan_toolchain.cmake")
      set(CMAKE_TOOLCHAIN_FILE "${CMAKE_BINARY_DIR}/conan_toolchain.cmake"
          CACHE FILEPATH "Conan toolchain" FORCE)
    endif()
  endif()
endif()

# ===================== 项目信息 =====================
project(astroflow LANGUAGES CXX CUDA)
set(TARGET_NAME _astroflow_core)

# ===================== 语言/架构/构建类型 =====================
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 只在用户未显式指定时给默认值，避免覆盖外部传入的设置
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 按需调整，你的卡覆盖 75/80/86/89
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# ===================== 依赖 =====================
#find_package(Python3 COMPONENTS Interpreter Development.Module Development.Embed REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Using Python: ${Python3_EXECUTABLE}")
message(STATUS "Python includes: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python library: ${Python3_LIBRARIES}")
find_package(CUDAToolkit)
find_package(pybind11 REQUIRED CONFIG)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
# 测试可选
option(BUILD_TESTING "Build C++/CUDA tests with GTest" OFF)
if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  include(GoogleTest)
  enable_testing()
endif()

# ===================== 优化/编译器标志 =====================
set(CUDA_OPTIMIZATION_FLAGS
  "-O3"
  "--use_fast_math"
  "--ftz=true"
  "--prec-div=false"
  "--prec-sqrt=false"
  "--fmad=true"
  "--extra-device-vectorization"
  "--maxrregcount=64"
  "-Xptxas=-v"
  "-Xptxas=-O3"
  "--generate-line-info"
  "--relocatable-device-code=false"
)
set(CUDA_MEMORY_FLAGS
  "-Xptxas=--allow-expensive-optimizations=true"
  "-Xcompiler=-O3"
  "-Xcompiler=-march=native"
  "-Xcompiler=-fopenmp"
  "-Xcompiler=-fno-math-errno"
  "-Xcompiler=-ffast-math"
  "-Xcompiler=-funroll-loops"
  "-Xcompiler=-fomit-frame-pointer"
  "-Xcompiler=-pipe"
)
set(CUDA_BASE_FLAGS
  "--ptxas-options=-v"
  "--ptxas-options=--warn-on-spills"
  "--expt-relaxed-constexpr"
  "--expt-extended-lambda"
  "--default-stream per-thread"
)
set(CMAKE_CUDA_FLAGS "-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G -lineinfo --ptxas-options=-v -Xcompiler=-O0 -Xcompiler=-g")

string(REPLACE ";" " " CMAKE_CUDA_FLAGS_RELEASE
  "${CUDA_OPTIMIZATION_FLAGS};${CUDA_MEMORY_FLAGS};${CUDA_BASE_FLAGS}")

# C++ 优化
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -g -pg")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")
set(CMAKE_CXX_FLAGS_RELEASE
  "-Ofast -march=native -mtune=native -fopenmp -fno-math-errno -ffast-math -funroll-loops -fomit-frame-pointer -pipe -mfma -mavx2 -msse4.2 -flto -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto -Wl,--as-needed -Wl,--gc-sections")

message(STATUS "CUDA Optimization Flags: ${CMAKE_CUDA_FLAGS_RELEASE}")
message(STATUS "CXX Optimization Flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "Target CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ===================== 源码收集 =====================
file(GLOB SRC_BIND CONFIGURE_DEPENDS "src/bindpy/*.cpp" "src/bindpy/*.cu")
file(GLOB SRC_DATA CONFIGURE_DEPENDS "src/data/*.cpp" "src/data/*.cu")
file(GLOB SRC_CORE CONFIGURE_DEPENDS "src/core/*.cpp" "src/core/*.cu")
file(GLOB SRC_DETE CONFIGURE_DEPENDS "src/detection/*.cpp" "src/detection/*.cu")
file(GLOB SRC_RFI  CONFIGURE_DEPENDS "src/rfi/*.cpp" "src/rfi/*.cu")
file(GLOB SRC_UTILS CONFIGURE_DEPENDS "src/utils/*.cpp" "src/utils/*.cu")  # 你之前缺失

set(SRC_FILES ${SRC_DATA} ${SRC_CORE} ${SRC_UTILS} ${SRC_DETE} ${SRC_RFI})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ===================== Python 扩展模块 =====================
# 用 pybind11_add_module 生成 Python 扩展（会自动处理 .so/.pyd 等平台后缀）
pybind11_add_module(${TARGET_NAME} ${SRC_FILES} ${SRC_BIND})

# 属性
set_target_properties(${TARGET_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
  CUDA_RUNTIME_LIBRARY Static
)

# 编译选项（区分语言和构建类型）
target_compile_options(${TARGET_NAME} PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    $<$<CONFIG:Release>:-O3;--use_fast_math;--fmad=true;--extra-device-vectorization>
    $<$<CONFIG:Debug>:-O0;-g;-G;-lineinfo>
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --cudart=static
    -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    $<$<CONFIG:Release>:-Ofast;-march=native;-mfma;-mavx2;-flto>
    $<$<CONFIG:Debug>:-O0;-g;-ggdb>
    -fopenmp
    -fno-math-errno
  >
)

# 头文件与库
target_include_directories(${TARGET_NAME} PRIVATE
  ${Python3_INCLUDE_DIRS}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${TARGET_NAME} PRIVATE
  CUDA::cudart_static
  pybind11::module
  Python3::Module
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
  dl pthread rt
)

# **关键**：安装到 Python 包内（wheel 中最终位于 astroflow/_astroflow_core.*）
install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION astroflow
        RUNTIME DESTINATION astroflow
        ARCHIVE DESTINATION astroflow)

# ===================== C++/CUDA 测试（可选） =====================
if(BUILD_TESTING)
  file(GLOB TESTS CONFIGURE_DEPENDS "tests/*.cpp")
  add_executable(astroflow_test ${TESTS} ${SRC_FILES} ${SRC_BIND})

  set_target_properties(astroflow_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION OFF
  )

  target_compile_options(astroflow_test PRIVATE
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

  target_include_directories(astroflow_test PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  target_link_libraries(astroflow_test PRIVATE
    gtest::gtest
    pybind11::embed
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
  )

  # 发现并注册测试
  gtest_discover_tests(astroflow_test
    EXTRA_ARGS --gtest_output=xml:${CMAKE_BINARY_DIR}/test_results.xml
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
