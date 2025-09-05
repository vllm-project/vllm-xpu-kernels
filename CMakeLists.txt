cmake_minimum_required(VERSION 3.26)

# When building directly using CMake, make sure you run the install step
# (it places the .so files in the correct location).
#
# Example:
# mkdir build && cd build
# cmake -G Ninja -DVLLM_PYTHON_EXECUTABLE=`which python3` -DCMAKE_INSTALL_PREFIX=.. ..
# cmake --build . --target install
#
# If you want to only build one target, make sure to install it manually:
# cmake --build . --target _C
# cmake --install . --component _C
project(vllm_extensions LANGUAGES CXX)

# XPU by default, used by setup.py
set(VLLM_TARGET_DEVICE "xpu" CACHE STRING "Target device backend for vLLM")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

# Prevent installation of dependencies (cutlass) by default.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12")

# Supported Intel GPU architectures.
set(SYCL_SUPPORTED_ARCHS "intel_gpu_pvc;intel_gpu_bmg_g21")

#
# Supported/expected torch versions for XPU.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# TODO: we need to align torch version with used in vLLM.
#
set(TORCH_SUPPORTED_VERSION_XPU "2.8.0")

#
# Try to find python package with an executable that exactly matches
# `VLLM_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

#
# Import torch cmake configuration.
find_package(Torch REQUIRED)

find_package(oneDNN REQUIRED)

#
# Forward the non-CUDA device extensions to external CMake scripts.
#
if (NOT VLLM_TARGET_DEVICE STREQUAL "xpu")
  message(STATUS "Not support building non-XPU device extensions.")
  return()
endif()


#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if(VLLM_TARGET_DEVICE STREQUAL "xpu")
  message(STATUS "Building XPU")
  set(VLLM_GPU_LANG "SYCL")
else()
  message(FATAL_ERROR "Can't find non-XPU installation.")
endif()



if(VLLM_TARGET_DEVICE STREQUAL "xpu")
  #
  # For other GPU targets override the GPU architectures detected by cmake/torch
  # and filter them by the supported versions for the current language.
  # The final set of arches is stored in `VLLM_GPU_ARCHES`.
  #

  # TODO: add sycl architectures
  override_gpu_arches(VLLM_GPU_ARCHES
    ${VLLM_GPU_LANG}
    "${${VLLM_GPU_LANG}_SUPPORTED_ARCHS}")
endif()

#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_GPU_FLAGS`.
#

message(STATUS "Querying torch for GPU compiler flags for ${VLLM_GPU_LANG}...")
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})
message(STATUS "Torch GPU compiler flags: ${VLLM_GPU_FLAGS}")


#
# Use FetchContent for C++ dependencies that are compiled as part of vLLM's build process.
# setup.py will override FETCHCONTENT_BASE_DIR to play nicely with sccache.
# Each dependency that produces build artifacts should override its BINARY_DIR to avoid
# conflicts between build types. It should instead be set to ${CMAKE_BINARY_DIR}/<dependency>.
#
include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

if(VLLM_GPU_LANG STREQUAL "SYCL")
  #
  # For SYCL we want to use the same flags as CUDA, so we set them here.
  # Note that SYCL does not support all CUDA flags, so some of them will be ignored.
  #
  # TODO: check SYCL flags
  set(CMAKE_${VLLM_GPU_LANG}_FLAGS "${CMAKE_${VLLM_GPU_LANG}_FLAGS}")
  set(SYCL_FIRST_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/csrc/sycl_first.h")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include ${SYCL_FIRST_HEADER}")

  # define link options for SYCL, including AOT targets and optimizations
  set(PERF_LINK_OPTS "-doubleGRF")
  set(PERF_LINK_OPTS "${PERF_LINK_OPTS} -vc-disable-indvars-opt")
  set(PERF_LINK_OPTS "${PERF_LINK_OPTS} -vc-codegen")
  set(PERF_LINK_OPTS "${PERF_LINK_OPTS} -Xfinalizer -printregusage")
  set(PERF_LINK_OPTS "${PERF_LINK_OPTS} -Xfinalizer -enableBCR")
  set(PERF_LINK_OPTS "${PERF_LINK_OPTS} -Xfinalizer -DPASTokenReduction")

  set(LINKE_OPTS ${LINKE_OPTS}
      -fsycl
      -fsycl-device-code-split=per_kernel
  )
  # TODO: make AOT device list configurable
  set(USE_AOT_DEVLIST "pvc,bmg")

  if (USE_AOT_DEVLIST)
    set(LINKE_OPTS ${LINKE_OPTS} -fsycl-targets=spir64_gen)
  endif()

  set(LINKE_OPTS ${LINKE_OPTS} -Xs )

  if (USE_AOT_DEVLIST)
    set(LINKE_OPTS ${LINKE_OPTS} "-device ${USE_AOT_DEVLIST} -options '${PERF_LINK_OPTS}'")
  else()
    set(LINKE_OPTS ${LINKE_OPTS} "${PERF_LINK_OPTS}")
  endif()
  set(SYCL_LINK_OPTION  ${LINKE_OPTS})

endif()

#
# Define other extension targets
#


#
# _C extension
#
if(VLLM_GPU_LANG STREQUAL "SYCL")
  set(VLLM_EXT_SRC
    "csrc/cache.cpp"
    "csrc/layernorm.cpp"
    "csrc/activation.cpp"
    "csrc/pos_encoding_kernels.cpp"
    "csrc/torch_bindings.cpp"
    "csrc/quantization/fp8/fp8_quant.cpp"
  )

  include_directories("/usr/include")
  set(CMPLR_ROOT $ENV{CMPLR_ROOT})
  set(CMAKE_CXX_COMPILER icpx)
  set(VLLM_EXTRA_INCLUDE_DIRECTORIES ${CMPLR_ROOT}/include/sycl)
  list(APPEND VLLM_GPU_FLAGS "-DVLLM_BUILD_XPU_OPS")
  list(APPEND VLLM_GPU_LINK_FLAGS ${SYCL_LINK_OPTION})
  list(APPEND VLLM_LINK_LIBRARIES "sycl" "OpenCL" "pthread" "m" "dl" "torch" )
endif()

message(STATUS "Enabling C extension.")
define_gpu_extension_target(
  _C
  DESTINATION vllm_xpu_kernels
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  LINK_FLAGS ${VLLM_GPU_LINK_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

#
# xpu only ops/kernels, implemented with cutlass/onednn/sycl.
#
if(VLLM_GPU_LANG STREQUAL "SYCL")
  set(VLLM_EXT_XPU_SRC
    "csrc/xpu/torch_bindings.cpp"
  )
  include_directories("/usr/include")
  set(CMPLR_ROOT $ENV{CMPLR_ROOT})
  set(CMAKE_CXX_COMPILER icpx)
  set(VLLM_EXTRA_INCLUDE_DIRECTORIES ${CMPLR_ROOT}/include/sycl)
  list(APPEND VLLM_GPU_FLAGS "-DVLLM_BUILD_XPU_OPS" )
  list(APPEND VLLM_GPU_LINK_FLAGS ${SYCL_LINK_OPTION})
  list(APPEND VLLM_LINK_LIBRARIES "sycl" "OpenCL" "pthread" "m" "dl" "torch" )
endif()

if(ONEDNN_FOUND)
  set(_ONEDNN_SRC)
  file(GLOB _ONEDNN_SRC csrc/xpu/onednn/*.cpp)
  list(APPEND VLLM_EXT_XPU_SRC
    ${_ONEDNN_SRC}
  )
  include_directories(${ONEDNN_INCLUDE_DIR})
  link_libraries(${ONEDNN_LIBRARY})
endif()

define_gpu_extension_target(
  _xpu_C
  DESTINATION vllm_xpu_kernels
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_EXT_XPU_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  LINK_FLAGS ${VLLM_GPU_LINK_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

#
# _moe_C extension
#
set(VLLM_MOE_EXT_SRC
  "csrc/moe/torch_bindings.cpp"
  "csrc/moe/moe_align_sum_kernels.cpp")

message(STATUS "Enabling moe extension.")
define_gpu_extension_target(
  _moe_C
  DESTINATION vllm_xpu_kernels
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  LINK_FLAGS ${VLLM_GPU_LINK_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)
