cmake_minimum_required(VERSION 3.26)

# When building directly using CMake, make sure you run the install step (it
# places the .so files in the correct location).
#
# Example: mkdir build && cd build cmake -G Ninja
# -DVLLM_PYTHON_EXECUTABLE=`which python3` -DCMAKE_INSTALL_PREFIX=.. .. cmake
# --build . --target install
#
# If you want to only build one target, make sure to install it manually: cmake
# --build . --target _C cmake --install . --component _C
project(vllm_extensions LANGUAGES CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)
# XPU by default, used by setup.py
set(VLLM_TARGET_DEVICE
    "xpu"
    CACHE STRING "Target device backend for vLLM")
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
set(SYCL_SUPPORTED_ARCHS "intel_gpu_pvc;intel_gpu_bmg_g21;intel_gpu_bmg_g31")

#
# Supported/expected torch versions for XPU.
#
# Currently, having an incorrect pytorch version results in a warning rather
# than an error.
#
# TODO: we need to align torch version with used in vLLM.
#
set(TORCH_SUPPORTED_VERSION_XPU "2.9.0")

set(BUILD_SYCL_TLA_KERNELS
    ON
    CACHE BOOL "Build SYCL-TLA based kernels for XPU")
# ARCHITECTURE OPTIONS
set(VLLM_XPU_ENABLE_XE2 ON)
set(VLLM_XPU_ENABLE_XE_DEFAULT ON)

set(BASIC_KERNELS_ENABLED ON)
set(FA2_KERNELS_ENABLED ON)
set(MOE_KERNELS_ENABLED ON)
set(XPU_SPECIFIC_KERNELS_ENABLED ON)

#
# Try to find python package with an executable that exactly matches
# `VLLM_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if(VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE}
                              "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(
    FATAL_ERROR
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
if(NOT VLLM_TARGET_DEVICE STREQUAL "xpu")
  message(STATUS "Not support building non-XPU device extensions.")
  return()
endif()

#
# Set up GPU language and check the torch version and warn if it isn't what is
# expected.
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
  # and filter them by the supported versions for the current language. The
  # final set of arches is stored in `VLLM_GPU_ARCHES`.
  #

  # TODO: add sycl architectures
  override_gpu_arches(VLLM_GPU_ARCHES ${VLLM_GPU_LANG}
                      "${${VLLM_GPU_LANG}_SUPPORTED_ARCHS}")
endif()

#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`. The final set of arches is stored in `VLLM_GPU_FLAGS`.
#

message(STATUS "Querying torch for GPU compiler flags for ${VLLM_GPU_LANG}...")
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})
message(STATUS "Torch GPU compiler flags: ${VLLM_GPU_FLAGS}")

#
# Use FetchContent for C++ dependencies that are compiled as part of vLLM's
# build process. setup.py will override FETCHCONTENT_BASE_DIR to play nicely
# with sccache. Each dependency that produces build artifacts should override
# its BINARY_DIR to avoid conflicts between build types. It should instead be
# set to ${CMAKE_BINARY_DIR}/<dependency>.
#
include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

# Set RPATH to $ORIGIN so that the extensions can find their shared libraries
# when loaded by Python.
set(CMAKE_INSTALL_RPATH "$ORIGIN")

if(VLLM_GPU_LANG STREQUAL "SYCL")
  #
  # For SYCL we want to use the same flags as CUDA, so we set them here. Note
  # that SYCL does not support all CUDA flags, so some of them will be ignored.
  #
  # TODO: check SYCL flags
  set(CMAKE_${VLLM_GPU_LANG}_FLAGS "${CMAKE_${VLLM_GPU_LANG}_FLAGS}")
  set(SYCL_FIRST_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/csrc/sycl_first.h")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include ${SYCL_FIRST_HEADER}")

  # ============= COMPILE OPTIONS ==================
  set(SYCL_FLAGS "")
  set(SYCL_KERNEL_OPTIONS)

  list(APPEND SYCL_FLAGS "-fsycl")

  set(SYCL_COMPILE_FLAGS ${SYCL_FLAGS})

  set(SYCL_TARGETS_OPTION -fsycl-targets=spir64_gen)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} ${SYCL_TARGETS_OPTION})
  set(SYCL_COMPILE_FLAGS ${SYCL_COMPILE_FLAGS} ${SYCL_KERNEL_OPTIONS})

  set(SYCL_COMPILE_FLAGS ${SYCL_COMPILE_FLAGS} "-O3" "-DNDEBUG")
  # Final build option be like: icpx -fsycl -fsycl-target=spir64_gen
  # ${SYCL_KERNEL_OPTIONS} -fsycl-host-compiler=gcc
  # -fsycl-host-compiler-options='${CMAKE_HOST_FLAGS}' kernel.cpp -o kernel.o
  set(VLLM_GPU_COMPILE_FLAGS ${SYCL_COMPILE_FLAGS})

  # ============= LINK OPTIONS ==================
  set(SYCL_LINK_FLAGS "")
  list(APPEND SYCL_LINK_FLAGS "-fsycl")
  set(SYCL_DEVICE_LINK_FLAGS ${SYCL_LINK_FLAGS})
  set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} ${SYCL_TARGETS_OPTION})
  set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS}
                             -fsycl-max-parallel-link-jobs=16)
  set(SYCL_DEVICE_LINK_FLAGS
      ${SYCL_DEVICE_LINK_FLAGS}
      "-Xspirv-translator;-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"
  )

  # TODO: make AOT configurable
  set(AOT_DEVICES "pvc,bmg-g21-a0,bmg-g31-a0")
  set(XE2_AOT_DEVICES "pvc,bmg-g21-a0,bmg-g31-a0")

  # Final link option be like: icpx -fsycl -fsycl-target=spir64_gen
  # ${SYCL_DEVICE_LINK_FLAGS} -Xsycl-target-backend=spir64_gen
  # '${SYCL_OFFLINE_COMPILER_FLAGS}' kernel.o -o device-code.o
  set(FINAL_LINK_OPTIONS "")
  list(APPEND FINAL_LINK_OPTIONS ${SYCL_DEVICE_LINK_FLAGS})
  list(APPEND FINAL_LINK_OPTIONS -Xsycl-target-backend=spir64_gen
       "-device ${AOT_DEVICES} -internal_options -cl-intel-256-GRF-per-thread")

  set(VLLM_GPU_LINK_FLAGS ${FINAL_LINK_OPTIONS})

  message(STATUS "Final SYCL compile options: ${VLLM_GPU_COMPILE_FLAGS}")
  message(STATUS "Final SYCL link options: ${VLLM_GPU_LINK_FLAGS}")
endif()

# sycl-tla setup. Fetch SYCL-TLA code base, and setup related FLAGs
if(VLLM_GPU_LANG STREQUAL "SYCL")
  message(STATUS "Setting up SYCL-TLA dependency...")
  # add cutlass dependency
  set(CUTLASS_ENABLE_HEADERS_ONLY
      "ON"
      CACHE BOOL "Enable only the header library")

  # Set CUTLASS_REVISION. Used for FetchContent. Also fixes some bogus messages
  # when building.
  set(CUTLASS_REVISION
      "3f2a337e885db0fb97b2a6ba514eb7a2a734ac4a"
      CACHE STRING "CUTLASS revision to use")

  if(DEFINED ENV{VLLM_CUTLASS_SRC_DIR})
    set(VLLM_CUTLASS_SRC_DIR $ENV{VLLM_CUTLASS_SRC_DIR})
  endif()

  if(VLLM_CUTLASS_SRC_DIR)
    if(NOT IS_ABSOLUTE VLLM_CUTLASS_SRC_DIR)
      get_filename_component(VLLM_CUTLASS_SRC_DIR "${VLLM_CUTLASS_SRC_DIR}"
                             ABSOLUTE)
    endif()
    message(
      STATUS
        "The VLLM_CUTLASS_SRC_DIR is set, using ${VLLM_CUTLASS_SRC_DIR} for compilation"
    )
    FetchContent_Declare(cutlass-sycl SOURCE_DIR ${VLLM_CUTLASS_SRC_DIR})
  else()
    # Use the specified CUTLASS source directory for compilation if
    # VLLM_CUTLASS_SRC_DIR  is provided
    FetchContent_Declare(
      cutlass-sycl
      GIT_REPOSITORY https://github.com/intel/sycl-tla.git
      # Please keep this in sync with CUTLASS_REVISION line above.
      GIT_TAG ${CUTLASS_REVISION}
      GIT_PROGRESS TRUE
      # Speed up CUTLASS download by retrieving only the specified GIT_TAG
      # instead of   the history. Important: If GIT_SHALLOW is enabled then
      # GIT_TAG works only with branch names  and tags. So if the GIT_TAG above
      # is updated to a commit hash, GIT_SHALLOW must be set to   FALSE
      GIT_SHALLOW FALSE)
  endif()

  # cutlass compilation flags
  set(CUTLASS_ENABLE_SYCL "ON")
  # TODO: make this a list
  set(DPCPP_SYCL_TARGET
      "intel_gpu_bmg_g21"
      CACHE STRING "DPC++ SYCL target architectures")
  set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
  set(CUTLASS_ENABLE_BENCHMARKS "OFF")
  # disable cuda
  set(CUTLASS_ENABLE_GDC_FOR_SM100_DEFAULT
      OFF
      CACHE BOOL "DISABLE CUDA")

  FetchContent_MakeAvailable(cutlass-sycl)
  set(CUTLASS_INCLUDE_DIR
      ${cutlass-sycl_SOURCE_DIR}/include
      CACHE PATH "CUTLASS Header Library")
  set(CUTLASS_TOOLS_UTIL_INCLUDE_DIR
      ${cutlass-sycl_SOURCE_DIR}/tools/util/include
      CACHE INTERNAL "")
  set(CUTLASS_APP_INCLUDE_DIR
      ${cutlass-sycl_SOURCE_DIR}/applications
      CACHE INTERNAL "")
  message(
    STATUS
      "cutlass dir: ${CUTLASS_INCLUDE_DIR} and ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR} and ${CUTLASS_APP_INCLUDE_DIR}"
  )

  # header only library
  list(APPEND VLLM_CUTLASS_FLAGS "-DCUTLASS_ENABLE_HEADERS_ONLY")
  list(APPEND VLLM_CUTLASS_FLAGS "-DCUTLASS_ENABLE_SYCL")
  list(APPEND VLLM_CUTLASS_FLAGS "-DSYCL_INTEL_TARGET")
  list(APPEND VLLM_CUTLASS_FLAGS "-DCUTLASS_VERSIONS_GENERATED")
  list(APPEND VLLM_CUTLASS_FLAGS "-ftemplate-backtrace-limit=0")
  list(APPEND VLLM_CUTLASS_FLAGS "-fdiagnostics-color=always")
endif()

set(ATTN_KERNEL_LIB_NAME "")
set(GROUPED_GEMM_LIB_NAME "")

if(BUILD_SYCL_TLA_KERNELS)

  set(SYCL_TLA_INCLUDE_DIRS
      ${CUTLASS_INCLUDE_DIR} ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
      ${CUTLASS_APP_INCLUDE_DIR})

  set(SYCL_TLA_KERNELS_COMPILE_FLAGS ${VLLM_GPU_COMPILE_FLAGS})
  list(APPEND SYCL_TLA_KERNELS_COMPILE_FLAGS ${VLLM_CUTLASS_FLAGS})

  # compile static library for attn and grouped_gemm, link this library to vLLM
  # extensions shared library
  set(SYCL_TLA_COMPILE_OPTIONS "")
  if(VLLM_XPU_ENABLE_XE_DEFAULT)
    add_subdirectory(csrc/xpu/grouped_gemm/xe_default)
    list(APPEND GROUPED_GEMM_LIB_NAME "grouped_gemm_xe_default")
    list(APPEND SYCL_TLA_COMPILE_OPTIONS -DVLLM_XPU_ENABLE_XE_DEFAULT)
  endif()
  if(VLLM_XPU_ENABLE_XE2)
    add_subdirectory(csrc/xpu/grouped_gemm/xe_2)
    add_subdirectory(csrc/xpu/attn/xe_2)
    list(APPEND GROUPED_GEMM_LIB_NAME "grouped_gemm_xe_2")
    list(APPEND ATTN_KERNEL_LIB_NAME "attn_kernels_xe_2")
    list(APPEND SYCL_TLA_COMPILE_OPTIONS -DVLLM_XPU_ENABLE_XE2)
  endif()
  list(APPEND VLLM_GPU_COMPILE_FLAGS ${SYCL_TLA_COMPILE_OPTIONS})

endif()

# define vLLM XPU cmake variables

set(VLLM_XPU_INCLUDE_DIR "")
list(APPEND VLLM_XPU_INCLUDE_DIR ${CMPLR_ROOT}/include/)
list(APPEND VLLM_XPU_INCLUDE_DIR ${CMPLR_ROOT}/include/sycl/)
list(APPEND VLLM_XPU_INCLUDE_DIR ${CMPLR_ROOT}/include/syclcompat/)
message(STATUS "VLLM_XPU_INCLUDE_DIR: ${VLLM_XPU_INCLUDE_DIR}")

set(VLLM_XPU_LINK_LIBRARIES "")
list(
  APPEND
  VLLM_XPU_LINK_LIBRARIES
  "sycl"
  "OpenCL"
  "pthread"
  "m"
  "dl"
  "torch")

#
# Define other extension targets
#

#
# xpumem_allocator extension - XPU memory allocator with Python callbacks
#
set(XPUMEM_ALLOCATOR_ENABLED ON CACHE BOOL "Build xpumem_allocator extension")
if(XPUMEM_ALLOCATOR_ENABLED)
  message(STATUS "Enabling xpumem_allocator extension.")

  set(XPUMEM_ALLOCATOR_SRC
    "csrc/xpu/xpu_mem.cpp"
  )
  include_directories("/usr/include")

  # Create Python module for xpumem_allocator
  # This is a standalone Python extension (not a torch extension)
  # Use the same SABI settings as other extensions for consistency
  find_package(Python REQUIRED COMPONENTS Development)
  find_package(Torch REQUIRED)
  list(APPEND PYTORCH_INCLUDES
    ${TORCH_INCLUDE_DIRS}
    #${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
    )
  message(STATUS "All Torch include dirs: ${TORCH_INCLUDE_DIRS}")
  Python_add_library(xpumem_allocator MODULE USE_SABI 3 WITH_SOABI "${XPUMEM_ALLOCATOR_SRC}")

  set_property(TARGET xpumem_allocator PROPERTY CXX_STANDARD 17)

  # Set compile options for SYCL
  if(VLLM_GPU_LANG STREQUAL "SYCL")
    target_compile_options(xpumem_allocator PRIVATE ${VLLM_GPU_COMPILE_FLAGS})
  endif()
  get_target_property(EXISTING_DEFS xpumem_allocator COMPILE_DEFINITIONS)
  if(EXISTING_DEFS)
    list(REMOVE_ITEM EXISTING_DEFS "Py_LIMITED_API=3")
    set_target_properties(xpumem_allocator PROPERTIES COMPILE_DEFINITIONS "${EXISTING_DEFS}")
  endif()
  message("!!!!!output")
  message(${Python_INCLUDE_DIRS})
  # Include directories
  target_include_directories(xpumem_allocator PRIVATE
    ${Python_INCLUDE_DIRS}
    ${PYTORCH_INCLUDES}
    csrc
    #${VLLM_XPU_INCLUDE_DIR}
    #${Python_INCLUDE_DIRS}
    #${PYTORCH_INCLUDES}
  )

  # Link libraries
  target_link_libraries(xpumem_allocator PRIVATE
    torch
    ${TORCH_LIBRARIES}
  )

  # Link options for SYCL
  if(VLLM_GPU_LANG STREQUAL "SYCL")
    target_link_options(xpumem_allocator PRIVATE ${VLLM_GPU_LINK_FLAGS})
  endif()

  # Install the module
  install(TARGETS xpumem_allocator
    LIBRARY DESTINATION vllm_xpu_kernels
    COMPONENT xpumem_allocator)
endif()

#
# _C extension
#
if(BASIC_KERNELS_ENABLED)
  message(STATUS "Enabling C extension.")
  set(VLLM_EXT_SRC
      "csrc/cache.cpp"
      "csrc/layernorm.cpp"
      "csrc/activation.cpp"
      "csrc/pos_encoding_kernels.cpp"
      "csrc/torch_bindings.cpp"
      "csrc/quantization/fp8/fp8_quant.cpp"
      "csrc/xpu_view.cpp"
      "csrc/tensor_utils.cpp")
  include_directories("/usr/include")

  define_gpu_extension_target(
    _C
    DESTINATION
    vllm_xpu_kernels
    LANGUAGE
    ${VLLM_GPU_LANG}
    SOURCES
    ${VLLM_EXT_SRC}
    COMPILE_FLAGS
    ${VLLM_GPU_COMPILE_FLAGS}
    LINK_FLAGS
    ${VLLM_GPU_LINK_FLAGS}
    ARCHITECTURES
    ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES
    ${VLLM_XPU_INCLUDE_DIR}
    USE_SABI
    3
    WITH_SOABI)

endif()

#
# flash attention _C extension
#
if(FA2_KERNELS_ENABLED)
  message(STATUS "Enabling fa2 extension.")
  file(GLOB FA2_GEN_SRCS "csrc/flash_attn/*.cpp"
       "csrc/xpu/attn/attn_interface.cpp")

  define_gpu_extension_target(
    _vllm_fa2_C
    DESTINATION
    vllm_xpu_kernels
    LANGUAGE
    ${VLLM_GPU_LANG}
    SOURCES
    ${FA2_GEN_SRCS}
    COMPILE_FLAGS
    ${VLLM_GPU_COMPILE_FLAGS}
    LINK_FLAGS
    ${VLLM_GPU_LINK_FLAGS}
    ARCHITECTURES
    ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES
    ${VLLM_XPU_INCLUDE_DIR}
    USE_SABI
    3
    LIBRARIES
    ${ATTN_KERNEL_LIB_NAME}
    WITH_SOABI)
endif()

#
# xpu only ops/kernels, implemented with cutlass/onednn/sycl.
#
if(XPU_SPECIFIC_KERNELS_ENABLED)
  message(STATUS "Enabling _xpu_C extension.")

  set(VLLM_EXT_XPU_SRC
      "csrc/xpu/torch_bindings.cpp" "csrc/xpu/lora/lora_shrink.cpp"
      "csrc/xpu/lora/lora_expand.cpp" "csrc/xpu/sycl/deepseek_scaling_rope.cpp"
      "csrc/xpu/grouped_gemm/grouped_gemm_interface.cpp")
  include_directories("/usr/include")
  # TODO: check if we need this flags list(APPEND VLLM_GPU_FLAGS
  # "-gline-tables-only")

  if(ONEDNN_FOUND)
    set(_ONEDNN_SRC)
    file(GLOB _ONEDNN_SRC csrc/xpu/onednn/*.cpp)
    list(APPEND VLLM_EXT_XPU_SRC ${_ONEDNN_SRC})
    include_directories(${ONEDNN_INCLUDE_DIR})
    link_libraries(${ONEDNN_LIBRARY})
  else()
    message(
      FATAL_ERROR "onednn not found but xpu specific kernels are enabled.")
  endif()

  define_gpu_extension_target(
    _xpu_C
    DESTINATION
    vllm_xpu_kernels
    LANGUAGE
    ${VLLM_GPU_LANG}
    SOURCES
    ${VLLM_EXT_XPU_SRC}
    COMPILE_FLAGS
    ${VLLM_GPU_COMPILE_FLAGS}
    LINK_FLAGS
    ${VLLM_GPU_LINK_FLAGS}
    ARCHITECTURES
    ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES
    ${VLLM_XPU_INCLUDE_DIR}
    LIBRARIES
    ${GROUPED_GEMM_LIB_NAME}
    USE_SABI
    3
    WITH_SOABI)
endif()

#
# _moe_C extension
#
if(MOE_KERNELS_ENABLED)
  message(STATUS "Enabling _moe_C extension.")
  file(GLOB VLLM_MOE_EXT_SRC "csrc/moe/*.cpp")

  define_gpu_extension_target(
    _moe_C
    DESTINATION
    vllm_xpu_kernels
    LANGUAGE
    ${VLLM_GPU_LANG}
    SOURCES
    ${VLLM_MOE_EXT_SRC}
    COMPILE_FLAGS
    ${VLLM_GPU_COMPILE_FLAGS}
    LINK_FLAGS
    ${VLLM_GPU_LINK_FLAGS}
    ARCHITECTURES
    ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES
    ${VLLM_XPU_INCLUDE_DIR}
    USE_SABI
    3
    WITH_SOABI)
endif()
