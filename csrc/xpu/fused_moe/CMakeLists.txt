cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# Generate the (weight_type x activation) up-projection instantiation TUs.
#
# These are pure, arch-independent boilerplate: each file just instantiates the
# FusedMOEUpLaunch<> template for one (weight, activation) combination. Keeping
# them as separate translation units preserves parallel compilation, but rather
# than checking ~18 near-identical files into the tree we materialize them at
# configure time from common/fused_moe_up_instantiation.cpp.in.
#
# The resulting list (FUSED_MOE_GENERATED_UP_SRCS) is inherited by the xe2 (and
# future xe3) subdirectories and fed to add_xe2_kernel_library via
# EXTRA_SOURCES, so each backend library compiles the sources with its own arch
# flags.
# ---------------------------------------------------------------------------
set(FUSED_MOE_UP_WEIGHTS w4a16 w8a16 w16a16)
set(FUSED_MOE_UP_ACTS silu gelu gelu_tanh swigluoai relu2_no_mul swiglustep)

set(FUSED_MOE_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(FUSED_MOE_GENERATED_UP_SRCS "")
foreach(weight IN LISTS FUSED_MOE_UP_WEIGHTS)
  string(TOUPPER "${weight}" FMOE_WEIGHT_UPPER)
  set(FMOE_WEIGHT_LOWER "${weight}")
  foreach(act IN LISTS FUSED_MOE_UP_ACTS)
    string(TOUPPER "${act}" FMOE_ACT_UPPER)
    set(FMOE_ACT_LOWER "${act}")
    set(_gen_src "${FUSED_MOE_GENERATED_DIR}/fused_moe_up_${weight}_${act}.cpp")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/common/fused_moe_up_instantiation.cpp.in"
      "${_gen_src}" @ONLY)
    list(APPEND FUSED_MOE_GENERATED_UP_SRCS "${_gen_src}")
  endforeach()
endforeach()

# Each architecture backend builds its own kernel library
add_subdirectory(xe2)
# add_subdirectory(xe3)  # future
