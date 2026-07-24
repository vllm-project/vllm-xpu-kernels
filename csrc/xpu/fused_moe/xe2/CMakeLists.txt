cmake_minimum_required(VERSION 3.18)

# Shared, arch-agnostic template instantiations live in ../common. Compile them
# (with this backend's XE2 flags) into fused_moe_xe2 via EXTRA_SOURCES.
file(GLOB FUSED_MOE_COMMON_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/../common/*.cpp")

add_xe2_kernel_library(fused_moe_xe2 INCLUDE_CMAKE_SOURCE_DIR
                       EXTRA_SOURCES ${FUSED_MOE_COMMON_SRCS})
