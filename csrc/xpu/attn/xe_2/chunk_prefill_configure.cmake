# =============================================================================
# Chunk Prefill Kernel Configuration
# =============================================================================
# This function generates kernel source files based on a configuration file that
# specifies which head sizes to build.
#
# CMake Options: VLLM_CHUNK_PREFILL_CONFIG - Path to kernel config file
# (default: chunk_prefill_full.conf) Config files located in:
# csrc/xpu/attn/kernel_configs/ chunk_prefill_full.conf    - All policies
# chunk_prefill_default.conf - Default model configs
#
# Config file format: - Lines starting with # are comments - Empty lines are
# ignored - 'all' keyword builds everything - Each line:
# headsize[,paged,causal,local,sink,lse]
#
# Only headsize matters now; trailing boolean fields are ignored for backward
# compatibility.
#
# Both standard and b16 policies are generated for each headsize.
# =============================================================================

# Default config path
if(NOT DEFINED VLLM_CHUNK_PREFILL_CONFIG)
  set(VLLM_CHUNK_PREFILL_CONFIG
      "${CMAKE_CURRENT_LIST_DIR}/../kernel_configs/chunk_prefill_full.conf")
endif()

# =============================================================================
# Helper: Parse chunk prefill config file
# =============================================================================
function(_chunk_prefill_parse_config CONFIG_FILE OUT_HEADS OUT_IS_FULL)
  set(_heads)
  set(_is_full FALSE)

  if(NOT EXISTS "${CONFIG_FILE}")
    message(
      FATAL_ERROR
        "Chunk prefill kernel config not found: ${CONFIG_FILE}\n"
        "Available presets: chunk_prefill_full.conf, chunk_prefill_default.conf\n"
        "Set via: cmake -DVLLM_CHUNK_PREFILL_CONFIG=<path>")
  endif()

  file(STRINGS "${CONFIG_FILE}" _lines)
  foreach(_line ${_lines})
    string(STRIP "${_line}" _line)
    if("${_line}" STREQUAL "" OR "${_line}" MATCHES "^#")
      continue()
    endif()
    if("${_line}" STREQUAL "all")
      set(_is_full TRUE)
      break()
    endif()

    string(REPLACE "," ";" _parts "${_line}")
    list(GET _parts 0 _headsize)
    string(STRIP "${_headsize}" _headsize)
    list(APPEND _heads "${_headsize}")
  endforeach()

  set(${OUT_HEADS}
      "${_heads}"
      PARENT_SCOPE)
  set(${OUT_IS_FULL}
      ${_is_full}
      PARENT_SCOPE)
endfunction()

function(fmha_forward_configure FILENAME_SUFFIX)
  set(GEN_KERNEL_SRCS) # output
  set(ENABLED_POLICIES) # track enabled policies

  set(policy_list
      "chunk_policy_head64"
      "chunk_policy_head96"
      "chunk_policy_head128"
      "chunk_policy_head192"
      "chunk_policy_head256"
      "chunk_policy_head512"
      "chunk_policy_head64_b16"
      "chunk_policy_head96_b16"
      "chunk_policy_head128_b16"
      "chunk_policy_head192_b16"
      "chunk_policy_head256_b16"
      "chunk_policy_head512_b16")

  # Map headsize to policy names
  set(std_policy_64 "chunk_policy_head64")
  set(std_policy_96 "chunk_policy_head96")
  set(std_policy_128 "chunk_policy_head128")
  set(std_policy_192 "chunk_policy_head192")
  set(std_policy_256 "chunk_policy_head256")
  set(std_policy_512 "chunk_policy_head512")
  set(b16_policy_64 "chunk_policy_head64_b16")
  set(b16_policy_96 "chunk_policy_head96_b16")
  set(b16_policy_128 "chunk_policy_head128_b16")
  set(b16_policy_192 "chunk_policy_head192_b16")
  set(b16_policy_256 "chunk_policy_head256_b16")
  set(b16_policy_512 "chunk_policy_head512_b16")

  # =============================================================================
  # Parse Configuration File
  # =============================================================================
  message(STATUS "Chunk prefill kernel config: ${VLLM_CHUNK_PREFILL_CONFIG}")
  _chunk_prefill_parse_config("${VLLM_CHUNK_PREFILL_CONFIG}" CONFIG_HEADS
                              CONFIG_IS_FULL)

  # =============================================================================
  # Build the list of policies to generate
  # =============================================================================
  set(BUILD_POLICIES)

  if(CONFIG_IS_FULL)
    set(BUILD_POLICIES ${policy_list})
  else()
    foreach(_headsize ${CONFIG_HEADS})
      # Guard against malformed entries (for example, BOM-prefixed comment
      # lines) that would otherwise expand to an empty policy name.
      if("${_headsize}" MATCHES "[^0-9]"
         OR "${std_policy_${_headsize}}" STREQUAL ""
         OR "${b16_policy_${_headsize}}" STREQUAL "")
        message(WARNING "Skipping invalid config headsize entry: ${_headsize}")
        continue()
      endif()

      list(APPEND BUILD_POLICIES "${std_policy_${_headsize}}")
      list(APPEND BUILD_POLICIES "${b16_policy_${_headsize}}")
    endforeach()
  endif()

  list(REMOVE_DUPLICATES BUILD_POLICIES)

  # =============================================================================
  # Generate Kernel Sources
  # =============================================================================
  foreach(IMPL_POLICY ${BUILD_POLICIES})
    if("${IMPL_POLICY}" STREQUAL "")
      continue()
    endif()

    list(APPEND ENABLED_POLICIES "${IMPL_POLICY}")

    configure_file(${FILENAME_SUFFIX}.cpp.in
                   "${FILENAME_SUFFIX}_${IMPL_POLICY}.cpp")
    list(APPEND GEN_KERNEL_SRCS
         "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_SUFFIX}_${IMPL_POLICY}.cpp")
  endforeach()

  # =============================================================================
  # Generate extern template header
  # =============================================================================
  list(REMOVE_DUPLICATES ENABLED_POLICIES)

  # Build the X-macro policy list content
  set(CHUNK_POLICY_LIST_ENTRIES "")
  list(LENGTH ENABLED_POLICIES _num_policies)
  math(EXPR _last_idx "${_num_policies} - 1")
  set(_idx 0)
  foreach(_pol ${ENABLED_POLICIES})
    if(_idx EQUAL _last_idx)
      set(CHUNK_POLICY_LIST_ENTRIES
          "${CHUNK_POLICY_LIST_ENTRIES}  X(${_pol})\n")
    else()
      set(CHUNK_POLICY_LIST_ENTRIES
          "${CHUNK_POLICY_LIST_ENTRIES}  X(${_pol}) \\\n")
    endif()
    math(EXPR _idx "${_idx} + 1")
  endforeach()

  configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/chunk_prefill_extern.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/chunk_prefill_extern_gen.hpp" @ONLY)

  # Build compile-time policy trait specializations
  set(CHUNK_ENABLED_POLICY_TRAITS "")
  foreach(_pol ${ENABLED_POLICIES})
    set(CHUNK_ENABLED_POLICY_TRAITS
        "${CHUNK_ENABLED_POLICY_TRAITS}template <>\nstruct is_chunk_policy_enabled<${_pol}> : std::true_type {};\n"
    )
  endforeach()

  configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/chunk_prefill_enabled_policies.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/chunk_prefill_enabled_policies_gen.hpp" @ONLY)

  # =============================================================================
  # Output Results
  # =============================================================================
  list(REMOVE_DUPLICATES GEN_KERNEL_SRCS)
  list(LENGTH GEN_KERNEL_SRCS GEN_KERNEL_SRCS_LENGTH)
  message(
    STATUS
      "Generated ${FILENAME_SUFFIX} kernel sources: ${GEN_KERNEL_SRCS_LENGTH} files "
      "(config: ${VLLM_CHUNK_PREFILL_CONFIG})")

  set(GEN_KERNEL_SRCS
      ${GEN_KERNEL_SRCS}
      PARENT_SCOPE)
  set(GEN_KERNEL_SRCS_LENGTH
      ${GEN_KERNEL_SRCS_LENGTH}
      PARENT_SCOPE)
  set(CHUNK_PREFILL_ENABLED_POLICIES
      ${ENABLED_POLICIES}
      PARENT_SCOPE)

  list(APPEND ATTN_KERNEL_SRCS_GEN ${GEN_KERNEL_SRCS})
  set(ATTN_KERNEL_SRCS_GEN
      ${ATTN_KERNEL_SRCS_GEN}
      PARENT_SCOPE)

endfunction()
