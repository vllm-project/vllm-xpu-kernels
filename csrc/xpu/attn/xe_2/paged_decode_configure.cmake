# =============================================================================
# Paged Decode Kernel Configuration
# =============================================================================
# This function generates kernel source files based on a configuration file that
# specifies which (qgroup, headsize, pagesize) combinations to build.
#
# Usage: paged_decode_configure(paged_decode_kernel_template)
#
# CMake Options: VLLM_PAGED_DECODE_CONFIG - Path to kernel config file (default:
# paged_decode_full.conf) Config files located in: csrc/xpu/attn/kernel_configs/
# paged_decode_full.conf    - All policies paged_decode_default.conf - Default
# model configs
#
# Config file format: - Lines starting with # are comments - Empty lines are
# ignored - 'all' keyword builds everything - Each line:
# qgroup,headsize,pagesize[,causal,local,sink]
#
# Only qgroup, headsize, and pagesize matter now; trailing boolean fields are
# ignored for backward compatibility.
#
# Parameters: FILENAME_SUFFIX - Base name for generated .cpp files
#
# Output: GEN_KERNEL_SRCS - List of generated source file paths
# GEN_KERNEL_SRCS_LENGTH - Number of generated files ATTN_KERNEL_SRCS_GEN -
# Updated global list with appended sources PAGED_DECODE_ENABLED_POLICIES - List
# of enabled policy names (for extern hdr)
# =============================================================================

# Default config path (can be overridden via cmake
# -DVLLM_PAGED_DECODE_CONFIG=...)
if(NOT DEFINED VLLM_PAGED_DECODE_CONFIG)
  set(VLLM_PAGED_DECODE_CONFIG
      "${CMAKE_CURRENT_LIST_DIR}/../kernel_configs/paged_decode_full.conf")
endif()

# =============================================================================
# Helper: Parse kernel config file
# =============================================================================
# Reads the config file and populates OUT_IS_FULL flag. If not full, populates
# OUT_TUPLES with "qgroup|headsize|pagesize" entries (using | as separator to
# avoid CMake list flattening).
function(_paged_decode_parse_config CONFIG_FILE OUT_TUPLES OUT_IS_FULL)
  set(_tuples)
  set(_is_full FALSE)

  if(NOT EXISTS "${CONFIG_FILE}")
    message(
      FATAL_ERROR
        "Paged decode kernel config not found: ${CONFIG_FILE}\n"
        "Available presets: paged_decode_full.conf, paged_decode_default.conf\n"
        "Set via: cmake -DVLLM_PAGED_DECODE_CONFIG=<path>")
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
    list(LENGTH _parts _nparts)
    if(_nparts LESS 3)
      message(WARNING "Skipping invalid config entry: ${_line}")
      continue()
    endif()

    list(GET _parts 0 _qgroup)
    list(GET _parts 1 _headsize)
    list(GET _parts 2 _pagesize)
    string(STRIP "${_qgroup}" _qgroup)
    string(STRIP "${_headsize}" _headsize)
    string(STRIP "${_pagesize}" _pagesize)
    list(APPEND _tuples "${_qgroup}|${_headsize}|${_pagesize}")
  endforeach()

  set(${OUT_TUPLES}
      "${_tuples}"
      PARENT_SCOPE)
  set(${OUT_IS_FULL}
      ${_is_full}
      PARENT_SCOPE)
endfunction()

function(paged_decode_configure FILENAME_SUFFIX)
  set(GEN_KERNEL_SRCS)
  set(ENABLED_POLICIES)

  # =============================================================================
  # Policy Configuration Mapping
  # =============================================================================
  # Maps (q_group_size, head_size, page_size) to policy type names. These must
  # match the policies defined in paged_decode_policy.hpp.

  # Q-group size 8 policies
  set(policy_8_64_16 "decode_policy_q8_h64_p16")
  set(policy_8_96_16 "decode_policy_q8_h96_p16")
  set(policy_8_128_16 "decode_policy_q8_h128_p16")
  set(policy_8_192_16 "decode_policy_q8_h192_p16")
  set(policy_8_256_16 "decode_policy_q8_h256_p16")
  set(policy_8_512_16 "decode_policy_q8_h512_p16")

  set(policy_8_64_32 "decode_policy_q8_h64_p32")
  set(policy_8_96_32 "decode_policy_q8_h96_p32")
  set(policy_8_128_32 "decode_policy_q8_h128_p32")
  set(policy_8_192_32 "decode_policy_q8_h192_p32")
  set(policy_8_256_32 "decode_policy_q8_h256_p32")
  set(policy_8_512_32 "decode_policy_q8_h512_p32")

  set(policy_8_64_64 "decode_policy_q8_h64_p64")
  set(policy_8_96_64 "decode_policy_q8_h96_p64")
  set(policy_8_128_64 "decode_policy_q8_h128_p64")
  set(policy_8_192_64 "decode_policy_q8_h192_p64")
  set(policy_8_256_64 "decode_policy_q8_h256_p64")
  set(policy_8_512_64 "decode_policy_q8_h512_p64")
  set(policy_8_576_64 "decode_policy_q8_h576_p64")

  set(policy_8_64_128 "decode_policy_q8_h64_p128")
  set(policy_8_96_128 "decode_policy_q8_h96_p128")
  set(policy_8_128_128 "decode_policy_q8_h128_p128")
  set(policy_8_192_128 "decode_policy_q8_h192_p128")
  set(policy_8_256_128 "decode_policy_q8_h256_p128")
  set(policy_8_512_128 "decode_policy_q8_h512_p128")
  set(policy_8_576_128 "decode_policy_q8_h576_p128")

  # Q-group size 16 policies
  set(policy_16_64_16 "decode_policy_q16_h64_p16")
  set(policy_16_96_16 "decode_policy_q16_h96_p16")
  set(policy_16_128_16 "decode_policy_q16_h128_p16")
  set(policy_16_192_16 "decode_policy_q16_h192_p16")
  set(policy_16_256_16 "decode_policy_q16_h256_p16")
  set(policy_16_512_16 "decode_policy_q16_h512_p16")

  set(policy_16_64_32 "decode_policy_q16_h64_p32")
  set(policy_16_96_32 "decode_policy_q16_h96_p32")
  set(policy_16_128_32 "decode_policy_q16_h128_p32")
  set(policy_16_192_32 "decode_policy_q16_h192_p32")
  set(policy_16_256_32 "decode_policy_q16_h256_p32")
  set(policy_16_512_32 "decode_policy_q16_h512_p32")

  set(policy_16_64_64 "decode_policy_q16_h64_p64")
  set(policy_16_96_64 "decode_policy_q16_h96_p64")
  set(policy_16_128_64 "decode_policy_q16_h128_p64")
  set(policy_16_192_64 "decode_policy_q16_h192_p64")
  set(policy_16_256_64 "decode_policy_q16_h256_p64")
  set(policy_16_512_64 "decode_policy_q16_h512_p64")
  set(policy_16_576_64 "decode_policy_q16_h576_p64")

  set(policy_16_64_128 "decode_policy_q16_h64_p128")
  set(policy_16_96_128 "decode_policy_q16_h96_p128")
  set(policy_16_128_128 "decode_policy_q16_h128_p128")
  set(policy_16_192_128 "decode_policy_q16_h192_p128")
  set(policy_16_256_128 "decode_policy_q16_h256_p128")
  set(policy_16_512_128 "decode_policy_q16_h512_p128")
  set(policy_16_576_128 "decode_policy_q16_h576_p128")

  set(qgroup_list "8" "16")
  set(headsize_list
      "64"
      "96"
      "128"
      "192"
      "256"
      "512"
      "576")
  set(pagesize_list "16" "32" "64" "128")

  # =============================================================================
  # Parse Configuration File
  # =============================================================================
  message(STATUS "Paged decode kernel config: ${VLLM_PAGED_DECODE_CONFIG}")
  _paged_decode_parse_config("${VLLM_PAGED_DECODE_CONFIG}" CONFIG_TUPLES
                             CONFIG_IS_FULL)

  # =============================================================================
  # Build the list of policies to generate
  # =============================================================================
  set(BUILD_POLICIES)

  if(CONFIG_IS_FULL)
    foreach(IMPL_QGROUP ${qgroup_list})
      foreach(IMPL_HEADSIZE ${headsize_list})
        foreach(IMPL_PAGESIZE ${pagesize_list})
          set(IMPL_POLICY
              ${policy_${IMPL_QGROUP}_${IMPL_HEADSIZE}_${IMPL_PAGESIZE}})
          if(NOT "${IMPL_POLICY}" STREQUAL "")
            list(APPEND BUILD_POLICIES "${IMPL_POLICY}")
          endif()
        endforeach()
      endforeach()
    endforeach()
  else()
    foreach(_tuple ${CONFIG_TUPLES})
      string(REPLACE "|" ";" _tuple_parts "${_tuple}")
      list(GET _tuple_parts 0 IMPL_QGROUP)
      list(GET _tuple_parts 1 IMPL_HEADSIZE)
      list(GET _tuple_parts 2 IMPL_PAGESIZE)

      set(IMPL_POLICY
          ${policy_${IMPL_QGROUP}_${IMPL_HEADSIZE}_${IMPL_PAGESIZE}})

      if("${IMPL_POLICY}" STREQUAL "")
        message(
          WARNING
            "No policy defined for qgroup=${IMPL_QGROUP}, "
            "headsize=${IMPL_HEADSIZE}, pagesize=${IMPL_PAGESIZE}. Skipping.")
        continue()
      endif()

      list(APPEND BUILD_POLICIES "${IMPL_POLICY}")
    endforeach()
  endif()

  list(REMOVE_DUPLICATES BUILD_POLICIES)
  set(ENABLED_POLICIES ${BUILD_POLICIES})

  # =============================================================================
  # Generate Kernel Sources
  # =============================================================================
  foreach(IMPL_POLICY ${BUILD_POLICIES})
    string(REGEX MATCH "_q([0-9]+)_h([0-9]+)_p([0-9]+)$" _unused
                 "${IMPL_POLICY}")
    if(NOT CMAKE_MATCH_1)
      message(
        WARNING "Could not derive filename suffix for policy ${IMPL_POLICY}")
      continue()
    endif()

    set(FILE_SUFFIX "_q${CMAKE_MATCH_1}_h${CMAKE_MATCH_2}_p${CMAKE_MATCH_3}")

    configure_file(${FILENAME_SUFFIX}.cpp.in
                   "${FILENAME_SUFFIX}${FILE_SUFFIX}.cpp")

    list(APPEND GEN_KERNEL_SRCS
         "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_SUFFIX}${FILE_SUFFIX}.cpp")
  endforeach()

  # =============================================================================
  # Generate extern template header
  # =============================================================================
  list(REMOVE_DUPLICATES ENABLED_POLICIES)

  set(POLICY_LIST_ENTRIES "")
  list(LENGTH ENABLED_POLICIES _num_policies)
  math(EXPR _last_idx "${_num_policies} - 1")
  set(_idx 0)
  foreach(_pol ${ENABLED_POLICIES})
    if(_idx EQUAL _last_idx)
      set(POLICY_LIST_ENTRIES "${POLICY_LIST_ENTRIES}  X(${_pol})\n")
    else()
      set(POLICY_LIST_ENTRIES "${POLICY_LIST_ENTRIES}  X(${_pol}) \\\n")
    endif()
    math(EXPR _idx "${_idx} + 1")
  endforeach()

  configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/paged_decode_extern.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/paged_decode_extern_gen.hpp" @ONLY)

  set(ENABLED_POLICY_TRAITS "")
  foreach(_pol ${ENABLED_POLICIES})
    set(ENABLED_POLICY_TRAITS
        "${ENABLED_POLICY_TRAITS}template <>\nstruct is_decode_policy_enabled<${_pol}> : std::true_type {};\n"
    )
  endforeach()

  configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/paged_decode_enabled_policies.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/paged_decode_enabled_policies_gen.hpp" @ONLY)

  # =============================================================================
  # Output Results
  # =============================================================================
  list(REMOVE_DUPLICATES GEN_KERNEL_SRCS)
  list(LENGTH GEN_KERNEL_SRCS GEN_KERNEL_SRCS_LENGTH)

  message(
    STATUS
      "Generated ${FILENAME_SUFFIX} sources: ${GEN_KERNEL_SRCS_LENGTH} files "
      "(config: ${VLLM_PAGED_DECODE_CONFIG})")

  set(GEN_KERNEL_SRCS
      ${GEN_KERNEL_SRCS}
      PARENT_SCOPE)
  set(GEN_KERNEL_SRCS_LENGTH
      ${GEN_KERNEL_SRCS_LENGTH}
      PARENT_SCOPE)
  set(PAGED_DECODE_ENABLED_POLICIES
      ${ENABLED_POLICIES}
      PARENT_SCOPE)

  list(APPEND ATTN_KERNEL_SRCS_GEN ${GEN_KERNEL_SRCS})
  set(ATTN_KERNEL_SRCS_GEN
      ${ATTN_KERNEL_SRCS_GEN}
      PARENT_SCOPE)

  message(
    STATUS
      "Total ATTN kernel sources after ${FILENAME_SUFFIX}: ${ATTN_KERNEL_SRCS_GEN}"
  )
endfunction()
