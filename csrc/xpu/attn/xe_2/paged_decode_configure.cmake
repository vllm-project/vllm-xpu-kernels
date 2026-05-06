# =============================================================================
# Paged Decode Kernel Configuration
# =============================================================================
# This function generates kernel source files based on a configuration file that
# specifies which (qgroup, headsize, pagesize) combinations to build.
#
# Usage: paged_decode_configure(paged_decode_kernel_template)
#
# CMake Options: VLLM_PAGED_DECODE_CONFIG - Path to kernel config file (default:
# full.conf) Presets available in kernel_configs/: full.conf    - All 384
# combinations (current behavior) common.conf  - Popular models (~72 kernels,
# ~81% reduction) llama.conf   - Llama family (~24 kernels, ~94% reduction)
# qwen.conf    - Qwen family (~24 kernels, ~94% reduction) deepseek.conf -
# DeepSeek family (~48 kernels, ~88% reduction)
#
# Config file format: - Lines starting with # are comments - Empty lines are
# ignored - 'all' keyword builds everything - Each line:
# qgroup,headsize,pagesize[,causal,local,sink] - If boolean flags omitted, all 8
# combinations are generated
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
      "${CMAKE_CURRENT_LIST_DIR}/kernel_configs/full.conf")
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
        "Available presets: full.conf, common.conf, llama.conf, qwen.conf, deepseek.conf\n"
        "Set via: cmake -DVLLM_PAGED_DECODE_CONFIG=<path>")
  endif()

  file(STRINGS "${CONFIG_FILE}" _lines)
  foreach(_line ${_lines})
    # Strip whitespace
    string(STRIP "${_line}" _line)
    # Skip empty lines and comments
    if("${_line}" STREQUAL "" OR "${_line}" MATCHES "^#")
      continue()
    endif()
    # Check for 'all' keyword
    if("${_line}" STREQUAL "all")
      set(_is_full TRUE)
      break()
    endif()
    # Replace commas with pipe delimiter (avoids CMake list flattening)
    string(REPLACE "," "|" _entry "${_line}")
    list(APPEND _tuples "${_entry}")
  endforeach()

  set(${OUT_TUPLES}
      "${_tuples}"
      PARENT_SCOPE)
  set(${OUT_IS_FULL}
      ${_is_full}
      PARENT_SCOPE)
endfunction()

function(paged_decode_configure FILENAME_SUFFIX)
  set(GEN_KERNEL_SRCS) # Initialize output list
  set(ENABLED_POLICIES) # Track which policies are enabled

  # Boolean flag values and their single-character representations
  set(L_BOOLS "false" "true")
  set(BOOL_FLAG_false "f")
  set(BOOL_FLAG_true "t")

  # =============================================================================
  # Policy Configuration Mapping
  # =============================================================================
  # Maps (q_group_size, head_size, page_size) to policy type names These must
  # match the policies defined in paged_decode_policy.hpp

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

  # Configuration space dimensions (for "all" mode)
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
  # Build the list of (qgroup, headsize, pagesize, causal, local, sink) tuples
  # =============================================================================
  set(BUILD_TUPLES)

<<<<<<< HEAD
        # Skip combinations that have no corresponding policy defined
        if(NOT IMPL_POLICY)
          continue()
        endif()

        foreach(IMPL_KISCAUSAL ${L_BOOLS})
          foreach(IMPL_KISLOCAL ${L_BOOLS})
            foreach(IMPL_KISSINK ${L_BOOLS})
              # Construct unique filename suffix: e.g., _q8_h64_fff
              set(FILE_SUFFIX
                  "_q${IMPL_QGROUP}_h${IMPL_HEADSIZE}_p${IMPL_PAGESIZE}_")
              set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISCAUSAL}}")
              set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISLOCAL}}")
              set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISSINK}}")

              # Generate .cpp file from template
              configure_file(${FILENAME_SUFFIX}.cpp.in
                             "${FILENAME_SUFFIX}${FILE_SUFFIX}.cpp")

              # Add to output list
              list(
                APPEND
                GEN_KERNEL_SRCS
                "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_SUFFIX}${FILE_SUFFIX}.cpp"
              )
=======
  if(CONFIG_IS_FULL)
    # Full mode: generate all combinations (original behavior)
    foreach(IMPL_QGROUP ${qgroup_list})
      foreach(IMPL_HEADSIZE ${headsize_list})
        foreach(IMPL_PAGESIZE ${pagesize_list})
          foreach(IMPL_KISCAUSAL ${L_BOOLS})
            foreach(IMPL_KISLOCAL ${L_BOOLS})
              foreach(IMPL_KISSINK ${L_BOOLS})
                list(
                  APPEND
                  BUILD_TUPLES
                  "${IMPL_QGROUP}|${IMPL_HEADSIZE}|${IMPL_PAGESIZE}|${IMPL_KISCAUSAL}|${IMPL_KISLOCAL}|${IMPL_KISSINK}"
                )
              endforeach()
>>>>>>> d9153fd (simplify template)
            endforeach()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  else()
    # Selective mode: only generate entries from config
    foreach(_entry ${CONFIG_TUPLES})
      # Split pipe-delimited entry
      string(REPLACE "|" ";" _parts "${_entry}")
      list(LENGTH _parts _nparts)
      if(_nparts LESS 3)
        message(WARNING "Skipping invalid config entry: ${_entry}")
        continue()
      endif()
      list(GET _parts 0 _qgroup)
      list(GET _parts 1 _headsize)
      list(GET _parts 2 _pagesize)

      if(_nparts GREATER_EQUAL 6)
        # Explicit boolean values provided
        list(GET _parts 3 _causal)
        list(GET _parts 4 _local)
        list(GET _parts 5 _sink)
        list(
          APPEND BUILD_TUPLES
          "${_qgroup}|${_headsize}|${_pagesize}|${_causal}|${_local}|${_sink}")
      else()
        # No booleans specified: generate all 8 combinations
        foreach(IMPL_KISCAUSAL ${L_BOOLS})
          foreach(IMPL_KISLOCAL ${L_BOOLS})
            foreach(IMPL_KISSINK ${L_BOOLS})
              list(
                APPEND
                BUILD_TUPLES
                "${_qgroup}|${_headsize}|${_pagesize}|${IMPL_KISCAUSAL}|${IMPL_KISLOCAL}|${IMPL_KISSINK}"
              )
            endforeach()
          endforeach()
        endforeach()
      endif()
    endforeach()
  endif()

  # =============================================================================
  # Generate Kernel Sources
  # =============================================================================
  foreach(_tuple ${BUILD_TUPLES})
    # Parse pipe-delimited tuple
    string(REPLACE "|" ";" _tuple_parts "${_tuple}")
    list(GET _tuple_parts 0 IMPL_QGROUP)
    list(GET _tuple_parts 1 IMPL_HEADSIZE)
    list(GET _tuple_parts 2 IMPL_PAGESIZE)
    list(GET _tuple_parts 3 IMPL_KISCAUSAL)
    list(GET _tuple_parts 4 IMPL_KISLOCAL)
    list(GET _tuple_parts 5 IMPL_KISSINK)

    # Lookup policy name from mapping
    set(IMPL_POLICY ${policy_${IMPL_QGROUP}_${IMPL_HEADSIZE}_${IMPL_PAGESIZE}})

    if("${IMPL_POLICY}" STREQUAL "")
      message(
        WARNING
          "No policy defined for qgroup=${IMPL_QGROUP}, "
          "headsize=${IMPL_HEADSIZE}, pagesize=${IMPL_PAGESIZE}. Skipping.")
      continue()
    endif()

    # Track enabled policies for extern header generation
    list(APPEND ENABLED_POLICIES "${IMPL_POLICY}")

    # Construct unique filename suffix: e.g., _q8_h64_p64_fff
    set(FILE_SUFFIX "_q${IMPL_QGROUP}_h${IMPL_HEADSIZE}_p${IMPL_PAGESIZE}_")
    set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISCAUSAL}}")
    set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISLOCAL}}")
    set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISSINK}}")

    # Generate .cpp file from template
    configure_file(${FILENAME_SUFFIX}.cpp.in
                   "${FILENAME_SUFFIX}${FILE_SUFFIX}.cpp")

    # Add to output list
    list(APPEND GEN_KERNEL_SRCS
         "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_SUFFIX}${FILE_SUFFIX}.cpp")
  endforeach()

  # =============================================================================
  # Generate extern template header
  # =============================================================================
  list(REMOVE_DUPLICATES ENABLED_POLICIES)

  # Build the X-macro policy list content
  set(POLICY_LIST_ENTRIES "")
  list(LENGTH ENABLED_POLICIES _num_policies)
  math(EXPR _last_idx "${_num_policies} - 1")
  set(_idx 0)
  foreach(_pol ${ENABLED_POLICIES})
    if(_idx EQUAL _last_idx)
      # Last entry: no trailing backslash
      set(POLICY_LIST_ENTRIES "${POLICY_LIST_ENTRIES}  X(${_pol})\n")
    else()
      set(POLICY_LIST_ENTRIES "${POLICY_LIST_ENTRIES}  X(${_pol}) \\\n")
    endif()
    math(EXPR _idx "${_idx} + 1")
  endforeach()

  # Generate the extern header from template
  configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/paged_decode_extern.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/paged_decode_extern_gen.hpp" @ONLY)

  # Build the compile-time policy trait specializations
  set(ENABLED_POLICY_TRAITS "")
  foreach(_pol ${ENABLED_POLICIES})
    set(ENABLED_POLICY_TRAITS
        "${ENABLED_POLICY_TRAITS}template <>\nstruct is_decode_policy_enabled<${_pol}> : std::true_type {};\n"
    )
  endforeach()

  # Generate the policy-enabled traits header
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

  # Export to parent scope
  set(GEN_KERNEL_SRCS
      ${GEN_KERNEL_SRCS}
      PARENT_SCOPE)
  set(GEN_KERNEL_SRCS_LENGTH
      ${GEN_KERNEL_SRCS_LENGTH}
      PARENT_SCOPE)
  set(PAGED_DECODE_ENABLED_POLICIES
      ${ENABLED_POLICIES}
      PARENT_SCOPE)

  # Update global kernel source list
  list(APPEND ATTN_KERNEL_SRCS_GEN ${GEN_KERNEL_SRCS})
  set(ATTN_KERNEL_SRCS_GEN
      ${ATTN_KERNEL_SRCS_GEN}
      PARENT_SCOPE)

  message(
    STATUS
      "Total ATTN kernel sources after ${FILENAME_SUFFIX}: ${ATTN_KERNEL_SRCS_GEN}"
  )

endfunction()
