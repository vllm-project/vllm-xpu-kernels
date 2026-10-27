cmake_minimum_required(VERSION 3.18)

# The CuTe grouped GEMM tiles are sized for 512 GRF per thread. Override the
# Xe3p pin for this library whenever the toolchain's SYCL grf_size property
# accepts 512 (add_xe3_kernel_library still forces 256 for NVL).
get_filename_component(_icpx_bindir "${CMAKE_CXX_COMPILER}" DIRECTORY)
set(_sycl_grf_props
    "${_icpx_bindir}/../include/sycl/ext/intel/experimental/grf_size_properties.hpp"
)
if(EXISTS "${_sycl_grf_props}")
  file(STRINGS "${_sycl_grf_props}" _grf_512_lines REGEX "Size == 512")
  if(_grf_512_lines)
    set(VLLM_XPU_XE3P_GRF_SIZE 512)
  endif()
endif()

add_xe3_kernel_library(grouped_gemm_xe_3 INCLUDE_CMAKE_SOURCE_DIR)

# Match the CUTLASS MoE benchmark's vector-alias spill workaround.
target_link_options(grouped_gemm_xe_3 PRIVATE -Xs
                    "-options \"-igc_opts 'VectorAliasBBThreshold=10000'\"")
