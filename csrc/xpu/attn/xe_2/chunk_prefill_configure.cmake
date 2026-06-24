# =============================================================================
# Chunk Prefill Kernel Configuration
# =============================================================================
# This function generates kernel source files based on a configuration file that
# specifies which (headsize, paged, causal, local, sink, lse) combinations to
# build.
#
# CMake Options: VLLM_CHUNK_PREFILL_CONFIG - Path to kernel config file
# (default: chunk_prefill_full.conf) Config files located in:
# csrc/xpu/attn/kernel_configs/ chunk_prefill_full.conf    - All 216
# combinations chunk_prefill_default.conf - Default model configs
#
# Config file format: - Lines starting with # are comments - Empty lines are
# ignored - 'all' keyword builds everything - Each line:
# headsize[,paged,causal,local,sink,lse] - If boolean flags omitted, all 18
# valid combinations are generated - LSE constraint: lse=true only valid when
# paged=false,local=false,sink=false
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
function(_chunk_prefill_parse_config CONFIG_FILE OUT_TUPLES OUT_IS_FULL)
  set(_tuples)
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

function(fmha_forward_configure FILENAME_SUFFIX)
  set(GEN_KERNEL_SRCS) # output
  set(ENABLED_POLICIES) # track enabled policies
  set(L_BOOLS "false" "true")
  set(BOOL_FLAG_false "f")
  set(BOOL_FLAG_true "t")

  set(headsize_list "64" "96" "128" "192" "256" "512")
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
      "chunk_policy_head512_b16"
      "chunk_policy_head192_vo128"
      "chunk_policy_head192_vo128_b16")

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

  # Asymmetric V/O policies: keyed by QK headsize, emitted in addition to the
  # symmetric policy so an asym model (e.g. MiMo QK=192,V=128) has a tile whose
  # output width follows V. Only headsize 192 has an asym variant for now.
  set(asym_std_policy_192 "chunk_policy_head192_vo128")
  set(asym_b16_policy_192 "chunk_policy_head192_vo128_b16")

  set(IMPL_KV_T "fp16")

  # =============================================================================
  # Parse Configuration File
  # =============================================================================
  message(STATUS "Chunk prefill kernel config: ${VLLM_CHUNK_PREFILL_CONFIG}")
  _chunk_prefill_parse_config("${VLLM_CHUNK_PREFILL_CONFIG}" CONFIG_TUPLES
                              CONFIG_IS_FULL)

  # =============================================================================
  # Build the list of (policy, paged, causal, local, sink, lse) tuples
  # =============================================================================
  set(BUILD_TUPLES)

  if(CONFIG_IS_FULL)
    # Full mode: generate all valid combinations (original behavior)
    foreach(IMPL_POLICY ${policy_list})
      foreach(IMPL_KISPAGED ${L_BOOLS})
        foreach(IMPL_KISCAUSAL ${L_BOOLS})
          foreach(IMPL_KISLOCAL ${L_BOOLS})
            foreach(IMPL_KISSINK ${L_BOOLS})
              # LSE constraint
              set(LSE_BOOLS "false")
              if(IMPL_KISPAGED STREQUAL "false"
                 AND IMPL_KISLOCAL STREQUAL "false"
                 AND IMPL_KISSINK STREQUAL "false")
                set(LSE_BOOLS ${L_BOOLS})
              endif()
              foreach(IMPL_KISLSE ${LSE_BOOLS})
                list(
                  APPEND
                  BUILD_TUPLES
                  "${IMPL_POLICY}|${IMPL_KISPAGED}|${IMPL_KISCAUSAL}|${IMPL_KISLOCAL}|${IMPL_KISSINK}|${IMPL_KISLSE}"
                )
              endforeach()
            endforeach()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  else()
    # Selective mode: only generate entries from config
    foreach(_entry ${CONFIG_TUPLES})
      string(REPLACE "|" ";" _parts "${_entry}")
      list(LENGTH _parts _nparts)
      if(_nparts LESS 1)
        message(WARNING "Skipping invalid config entry: ${_entry}")
        continue()
      endif()
      list(GET _parts 0 _headsize)

      # Guard against malformed entries (for example, BOM-prefixed comment
      # lines) that would otherwise expand to an empty policy name.
      if("${_headsize}" MATCHES "[^0-9]"
         OR "${std_policy_${_headsize}}" STREQUAL ""
         OR "${b16_policy_${_headsize}}" STREQUAL "")
        message(WARNING "Skipping invalid config headsize entry: ${_entry}")
        continue()
      endif()

      if(_nparts GREATER_EQUAL 6)
        # Explicit boolean values provided
        list(GET _parts 1 _paged)
        list(GET _parts 2 _causal)
        list(GET _parts 3 _local)
        list(GET _parts 4 _sink)
        list(GET _parts 5 _lse)

        # Validate boolean values
        set(_invalid_bool FALSE)
        foreach(_v ${_paged} ${_causal} ${_local} ${_sink} ${_lse})
          if(NOT (_v STREQUAL "true" OR _v STREQUAL "false"))
            message(WARNING "Skipping invalid config boolean entry: ${_entry}")
            set(_invalid_bool TRUE)
            break()
          endif()
        endforeach()
        if(_invalid_bool)
          continue()
        endif()

        # Validate LSE constraint
        if(_lse STREQUAL "true")
          if(NOT
             (_paged STREQUAL "false"
              AND _local STREQUAL "false"
              AND _sink STREQUAL "false"))
            message(
              WARNING
                "Skipping invalid config: lse=true requires paged=false,local=false,sink=false: ${_entry}"
            )
            continue()
          endif()
        endif()

        # Generate for both standard and b16 policies
        list(
          APPEND
          BUILD_TUPLES
          "${std_policy_${_headsize}}|${_paged}|${_causal}|${_local}|${_sink}|${_lse}"
        )
        list(
          APPEND
          BUILD_TUPLES
          "${b16_policy_${_headsize}}|${_paged}|${_causal}|${_local}|${_sink}|${_lse}"
        )
        # Also emit the asymmetric V/O variant for this headsize, if any.
        if(NOT "${asym_std_policy_${_headsize}}" STREQUAL "")
          list(
            APPEND
            BUILD_TUPLES
            "${asym_std_policy_${_headsize}}|${_paged}|${_causal}|${_local}|${_sink}|${_lse}"
          )
          list(
            APPEND
            BUILD_TUPLES
            "${asym_b16_policy_${_headsize}}|${_paged}|${_causal}|${_local}|${_sink}|${_lse}"
          )
        endif()
      else()
        # No booleans specified: generate all 18 valid combinations
        foreach(IMPL_KISPAGED ${L_BOOLS})
          foreach(IMPL_KISCAUSAL ${L_BOOLS})
            foreach(IMPL_KISLOCAL ${L_BOOLS})
              foreach(IMPL_KISSINK ${L_BOOLS})
                set(LSE_BOOLS "false")
                if(IMPL_KISPAGED STREQUAL "false"
                   AND IMPL_KISLOCAL STREQUAL "false"
                   AND IMPL_KISSINK STREQUAL "false")
                  set(LSE_BOOLS ${L_BOOLS})
                endif()
                foreach(IMPL_KISLSE ${LSE_BOOLS})
                  list(
                    APPEND
                    BUILD_TUPLES
                    "${std_policy_${_headsize}}|${IMPL_KISPAGED}|${IMPL_KISCAUSAL}|${IMPL_KISLOCAL}|${IMPL_KISSINK}|${IMPL_KISLSE}"
                  )
                  list(
                    APPEND
                    BUILD_TUPLES
                    "${b16_policy_${_headsize}}|${IMPL_KISPAGED}|${IMPL_KISCAUSAL}|${IMPL_KISLOCAL}|${IMPL_KISSINK}|${IMPL_KISLSE}"
                  )
                  if(NOT "${asym_std_policy_${_headsize}}" STREQUAL "")
                    list(
                      APPEND
                      BUILD_TUPLES
                      "${asym_std_policy_${_headsize}}|${IMPL_KISPAGED}|${IMPL_KISCAUSAL}|${IMPL_KISLOCAL}|${IMPL_KISSINK}|${IMPL_KISLSE}"
                    )
                    list(
                      APPEND
                      BUILD_TUPLES
                      "${asym_b16_policy_${_headsize}}|${IMPL_KISPAGED}|${IMPL_KISCAUSAL}|${IMPL_KISLOCAL}|${IMPL_KISSINK}|${IMPL_KISLSE}"
                    )
                  endif()
                endforeach()
              endforeach()
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
    string(REPLACE "|" ";" _tuple_parts "${_tuple}")
    list(GET _tuple_parts 0 IMPL_POLICY)
    list(GET _tuple_parts 1 IMPL_KISPAGED)
    list(GET _tuple_parts 2 IMPL_KISCAUSAL)
    list(GET _tuple_parts 3 IMPL_KISLOCAL)
    list(GET _tuple_parts 4 IMPL_KISSINK)
    list(GET _tuple_parts 5 IMPL_KISLSE)

    if("${IMPL_POLICY}" STREQUAL "")
      continue()
    endif()

    # Track enabled policies
    list(APPEND ENABLED_POLICIES "${IMPL_POLICY}")

    set(FILE_SUFFIX "${IMPL_POLICY}_")
    set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISPAGED}}")
    set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISCAUSAL}}")
    set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISSINK}}")
    set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISLOCAL}}")
    set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISLSE}}")
    configure_file(${FILENAME_SUFFIX}.cpp.in
                   "${FILENAME_SUFFIX}_${FILE_SUFFIX}.cpp")
    list(APPEND GEN_KERNEL_SRCS
         "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_SUFFIX}_${FILE_SUFFIX}.cpp")
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

  # Build compile-time policy+bool tuple trait specializations
  set(CHUNK_ENABLED_TUPLE_TRAITS "")
  set(ENABLED_TUPLES ${BUILD_TUPLES})
  list(REMOVE_DUPLICATES ENABLED_TUPLES)
  foreach(_tuple ${ENABLED_TUPLES})
    string(REPLACE "|" ";" _tuple_parts "${_tuple}")
    list(GET _tuple_parts 0 _pol)
    list(GET _tuple_parts 1 _paged)
    list(GET _tuple_parts 2 _causal)
    list(GET _tuple_parts 3 _local)
    list(GET _tuple_parts 4 _sink)
    list(GET _tuple_parts 5 _lse)

    if("${_pol}" STREQUAL "")
      continue()
    endif()

    set(CHUNK_ENABLED_TUPLE_TRAITS
        "${CHUNK_ENABLED_TUPLE_TRAITS}template <>\nstruct is_chunk_policy_tuple_enabled<${_pol}, ${_paged}, ${_causal}, ${_local}, ${_sink}, ${_lse}> : std::true_type {};\n"
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
