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

set(ENABLE_MOE_KERNEL OFF)
set(FA2_ENABLED ON)

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

#
# Define other extension targets
#


#
# _C extension
#

if(VLLM_GPU_LANG STREQUAL "SYCL")
  set(VLLM_EXT_SRC
    "csrc/xpu/cutlass_sycl_demo.cpp"
    "csrc/xpu/layernorm.cpp"
    "csrc/xpu/torch_bindings.cpp"
  )
  include_directories("/usr/include")
  list(APPEND VLLM_INCLUDE_DIR ${CMPLR_ROOT}/include/)
  list(APPEND VLLM_INCLUDE_DIR ${CMPLR_ROOT}/include/sycl/)
  list(APPEND VLLM_INCLUDE_DIR ${CMPLR_ROOT}/include/sycl/syclcompat/)
  message(STATUS "VLLM_INCLUDE_DIR: ${VLLM_INCLUDE_DIR}")
  set(VLLM_EXTRA_INCLUDE_DIRECTORIES ${CMPLR_ROOT}/include/sycl)
  list(APPEND VLLM_GPU_FLAGS "-DVLLM_BUILD_XPU_OPS" )
  list(APPEND VLLM_GPU_LINK_FLAGS "-fsycl" "-fsycl-targets=spir64" "-Xspirv-translator" "-spirv-ext=+SPV_INTEL_split_barrier")
  list(APPEND VLLM_LINK_LIBRARIES "sycl" "OpenCL" "pthread" "m" "dl" "torch" )


  # add cutlass dependency
  set(CUTLASS_ENABLE_HEADERS_ONLY "ON" CACHE BOOL "Enable only the header library")

  # Set CUTLASS_REVISION. Used for FetchContent. Also fixes some bogus messages when building.
  set(CUTLASS_REVISION "v3.9-0.2" CACHE STRING "CUTLASS revision to use")

  # Use the specified CUTLASS source directory for compilation if VLLM_CUTLASS_SRC_DIR is provided
  FetchContent_Declare(
      cutlass-sycl
      GIT_REPOSITORY https://github.com/codeplaysoftware/cutlass-sycl.git
      # Please keep this in sync with CUTLASS_REVISION line above.
      GIT_TAG ${CUTLASS_REVISION}
      GIT_PROGRESS TRUE

      # Speed up CUTLASS download by retrieving only the specified GIT_TAG instead of the history.
      # Important: If GIT_SHALLOW is enabled then GIT_TAG works only with branch names and tags.
      # So if the GIT_TAG above is updated to a commit hash, GIT_SHALLOW must be set to FALSE
      GIT_SHALLOW TRUE
  )

  # cutlass compilation flags
  set(CUTLASS_ENABLE_SYCL "ON")
  # set(DPCPP_SYCL_TARGET "intel_gpu_pvc;intel_gpu_bmg_g21" CACHE STRING "DPC++ SYCL target architectures")
  set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
  set(CUTLASS_ENABLE_BENCHMARKS "OFF")
  # disable cuda
  set(CUTLASS_ENABLE_GDC_FOR_SM100_DEFAULT OFF CACHE BOOL "DISABLE CUDA")
  # list(APPEND CMAKE_CXX_FLAGS "-ftemplate-backtrace-limit=0 " )
  # list(APPEND CMAKE_CXX_FLAGS "-fdiagnostics-color=always " )
  

  FetchContent_MakeAvailable(cutlass-sycl)
  set(CUTLASS_INCLUDE_DIR ${cutlass-sycl_SOURCE_DIR}/include CACHE PATH "CUTLASS Header Library")
  set(CUTLASS_TOOLS_UTIL_INCLUDE_DIR ${cutlass-sycl_SOURCE_DIR}/tools/util/include CACHE INTERNAL "")
  message(STATUS "cutlass dir: ${CUTLASS_INCLUDE_DIR} and ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}")

  # header only library
  list(APPEND VLLM_GPU_FLAGS "-DCUTLASS_ENABLE_SYCL")
  list(APPEND VLLM_GPU_FLAGS "-DSYCL_INTEL_TARGET")
  list(APPEND VLLM_GPU_FLAGS "-DCUTLASS_VERSIONS_GENERATED")
  list(APPEND VLLM_GPU_FLAGS "-ftemplate-backtrace-limit=0")
  list(APPEND VLLM_GPU_FLAGS "-fdiagnostics-color=always")

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
  INCLUDE_DIRECTORIES ${VLLM_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

#
# flash attention _C extension
#

if (FA2_ENABLED)
    message(STATUS "Enabling fa2 extension.")
    file(GLOB FA2_GEN_SRCS "csrc/flash_attn/*.cpp")

    define_gpu_extension_target(
        _vllm_fa2_C
        DESTINATION vllm_xpu_kernels
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES
            csrc/flash_attn/flash_api.cpp
            ${FA2_GEN_SRCS}
        COMPILE_FLAGS ${VLLM_GPU_FLAGS}
        LINK_FLAGS ${VLLM_GPU_LINK_FLAGS}
        ARCHITECTURES ${VLLM_GPU_ARCHES}
        USE_SABI 3
        WITH_SOABI)

    # target_include_directories(_vllm_fa2_C PRIVATE
    #     csrc/flash_attn
    #     csrc/flash_attn/src)
endif ()


#
# _moe_C extension
#
# TODO: add this as a placeholder for now.

if (ENABLE_MOE_KERNEL)
  set(VLLM_MOE_EXT_SRC
    "csrc/moe/torch_bindings.cpp"
  )

  message(STATUS "Enabling moe extension.")
  define_gpu_extension_target(
    _moe_C
    DESTINATION vllm_xpu_kernels
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_MOE_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
    INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
    USE_SABI 3
    WITH_SOABI)
endif()

