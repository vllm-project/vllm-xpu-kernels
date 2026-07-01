# =============================================================================
# Sparse MLA Kernel Configuration
# =============================================================================
# This function selects sparse MLA instantiation units based on a preset
# selector or a configuration file.
#
# CMake option: VLLM_SPARSE_MLA_CONFIG
#
# Selector/file format: - 'all' enables all sparse MLA generated variants - For
# file mode: lines starting with # are comments, empty lines are ignored -
# Entries: prefill,headsize[,topklen[,attn_sink]]
# decode_fp8,headsize[,topklen[,attn_sink]]
#
# Notes: - prefill headsize: 512 or 576 - decode_fp8 headsize: 512 - bool values
# must be 'true' or 'false'
# =============================================================================

if(NOT DEFINED VLLM_SPARSE_MLA_CONFIG OR "${VLLM_SPARSE_MLA_CONFIG}" STREQUAL
                                         "")
  set(VLLM_SPARSE_MLA_CONFIG "all")
endif()

function(_sparse_mla_parse_config CONFIG_FILE OUT_ENTRIES OUT_IS_FULL)
  set(_entries)
  set(_is_full FALSE)

  if("${CONFIG_FILE}" STREQUAL "all")
    set(${OUT_ENTRIES}
        "${_entries}"
        PARENT_SCOPE)
    set(${OUT_IS_FULL}
        TRUE
        PARENT_SCOPE)
    return()
  endif()

  if(NOT EXISTS "${CONFIG_FILE}")
    get_filename_component(_cfg_name "${CONFIG_FILE}" NAME)
    if(_cfg_name STREQUAL "sparse_mla_full.conf")
      message(WARNING "Sparse MLA kernel config not found: ${CONFIG_FILE}. "
                      "Falling back to built-in preset 'all'.")
      set(${OUT_ENTRIES}
          "${_entries}"
          PARENT_SCOPE)
      set(${OUT_IS_FULL}
          TRUE
          PARENT_SCOPE)
      return()
    else()
      file(GLOB _available_cfgs
           "${CMAKE_CURRENT_LIST_DIR}/kernel_configs/*.conf")
      set(_available_names "")
      foreach(_cfg ${_available_cfgs})
        get_filename_component(_cfg_base "${_cfg}" NAME)
        list(APPEND _available_names "${_cfg_base}")
      endforeach()
      list(APPEND _available_names "all")
      list(REMOVE_DUPLICATES _available_names)
      if(_available_names)
        string(REPLACE ";" ", " _available_text "${_available_names}")
      else()
        set(_available_text "<none>")
      endif()
      message(
        FATAL_ERROR
          "Sparse MLA kernel config not found: ${CONFIG_FILE}\n"
          "Available presets: ${_available_text}\n"
          "Set via: cmake -DVLLM_SPARSE_MLA_CONFIG=<path>")
    endif()
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
    list(APPEND _entries "${_entry}")
  endforeach()

  set(${OUT_ENTRIES}
      "${_entries}"
      PARENT_SCOPE)
  set(${OUT_IS_FULL}
      ${_is_full}
      PARENT_SCOPE)
endfunction()

function(sparse_mla_configure OUT_SRCS OUT_DEFINES)
  set(_srcs)
  set(_defines)

  message(STATUS "Sparse MLA kernel config: ${VLLM_SPARSE_MLA_CONFIG}")
  _sparse_mla_parse_config("${VLLM_SPARSE_MLA_CONFIG}" _config_entries
                           _config_is_full)

  if(_config_is_full)
    set(_entries "prefill|512" "prefill|576" "decode_fp8|512")
  else()
    set(_entries ${_config_entries})
  endif()

  foreach(_entry ${_entries})
    string(REPLACE "|" ";" _parts "${_entry}")
    list(LENGTH _parts _nparts)
    if(_nparts LESS 2)
      message(WARNING "Skipping invalid sparse MLA config entry: ${_entry}")
      continue()
    endif()

    list(GET _parts 0 _kind)
    list(GET _parts 1 _headsize)

    if(_kind STREQUAL "prefill")
      if(NOT (_headsize STREQUAL "512" OR _headsize STREQUAL "576"))
        message(
          WARNING
            "Skipping sparse MLA prefill entry with unsupported headsize: ${_entry}"
        )
        continue()
      endif()

      if(_nparts GREATER_EQUAL 3)
        list(GET _parts 2 _topk_single)
        set(_topk_values ${_topk_single})
      else()
        set(_topk_values false true)
      endif()

      if(_nparts GREATER_EQUAL 4)
        list(GET _parts 3 _sink_single)
        set(_sink_values ${_sink_single})
      else()
        set(_sink_values false true)
      endif()

      foreach(_topk ${_topk_values})
        if(NOT (_topk STREQUAL "true" OR _topk STREQUAL "false"))
          message(
            WARNING
              "Skipping invalid sparse MLA prefill topk bool entry: ${_entry}")
          continue()
        endif()

        foreach(_sink ${_sink_values})
          if(NOT (_sink STREQUAL "true" OR _sink STREQUAL "false"))
            message(
              WARNING
                "Skipping invalid sparse MLA prefill attn_sink bool entry: ${_entry}"
            )
            continue()
          endif()

          if(_topk STREQUAL "true")
            set(_topk_suffix "_topklen")
            set(_topk_def "_TOPKLEN")
          else()
            set(_topk_suffix "")
            set(_topk_def "")
          endif()

          if(_sink STREQUAL "true")
            set(_sink_suffix "_sink")
            set(_sink_def "_SINK")
          else()
            set(_sink_suffix "_nosink")
            set(_sink_def "_NOSINK")
          endif()

          set(_output_name
              "sparse_mla_prefill_fwd${_topk_suffix}_k${_headsize}${_sink_suffix}.cpp"
          )
          set(_def
              "VLLM_SPARSE_MLA_PREFILL${_topk_def}_K${_headsize}${_sink_def}")

          set(IMPL_HEADSIZE ${_headsize})
          set(IMPL_HAVE_TOPK_LENGTH ${_topk})
          set(IMPL_HAS_ATTN_SINK ${_sink})
          configure_file(
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/sparse_mla_prefill_kernel_template.cpp.in"
            "${CMAKE_CURRENT_BINARY_DIR}/${_output_name}")

          list(APPEND _srcs "${CMAKE_CURRENT_BINARY_DIR}/${_output_name}")
          list(APPEND _defines "${_def}")
        endforeach()
      endforeach()

    elseif(_kind STREQUAL "decode_fp8")
      if(NOT _headsize STREQUAL "512")
        message(
          WARNING
            "Skipping sparse MLA decode_fp8 entry with unsupported headsize: ${_entry}"
        )
        continue()
      endif()

      if(_nparts GREATER_EQUAL 3)
        list(GET _parts 2 _topk_single)
        set(_topk_values ${_topk_single})
      else()
        set(_topk_values false true)
      endif()

      if(_nparts GREATER_EQUAL 4)
        list(GET _parts 3 _sink_single)
        set(_sink_values ${_sink_single})
      else()
        set(_sink_values false true)
      endif()

      foreach(_topk ${_topk_values})
        if(NOT (_topk STREQUAL "true" OR _topk STREQUAL "false"))
          message(
            WARNING
              "Skipping invalid sparse MLA decode topk bool entry: ${_entry}")
          continue()
        endif()

        foreach(_sink ${_sink_values})
          if(NOT (_sink STREQUAL "true" OR _sink STREQUAL "false"))
            message(
              WARNING
                "Skipping invalid sparse MLA decode attn_sink bool entry: ${_entry}"
            )
            continue()
          endif()

          if(_topk STREQUAL "true")
            set(_topk_suffix "_topklen")
            set(_topk_def "_TOPKLEN")
          else()
            set(_topk_suffix "")
            set(_topk_def "")
          endif()

          if(_sink STREQUAL "true")
            set(_sink_suffix "_sink")
            set(_sink_def "_SINK")
          else()
            set(_sink_suffix "_nosink")
            set(_sink_def "_NOSINK")
          endif()

          set(_output_name
              "sparse_mla_decode_fp8_fwd${_topk_suffix}_k${_headsize}${_sink_suffix}.cpp"
          )
          set(_def
              "VLLM_SPARSE_MLA_DECODE_FP8${_topk_def}_K${_headsize}${_sink_def}"
          )

          set(IMPL_HEADSIZE ${_headsize})
          set(IMPL_HAVE_TOPK_LENGTH ${_topk})
          set(IMPL_HAS_ATTN_SINK ${_sink})
          configure_file(
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/sparse_mla_decode_fp8_kernel_template.cpp.in"
            "${CMAKE_CURRENT_BINARY_DIR}/${_output_name}")

          list(APPEND _srcs "${CMAKE_CURRENT_BINARY_DIR}/${_output_name}")
          list(APPEND _defines "${_def}")
        endforeach()
      endforeach()

    else()
      message(WARNING "Skipping unknown sparse MLA kernel type: ${_entry}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES _srcs)
  list(REMOVE_DUPLICATES _defines)

  list(LENGTH _srcs _src_count)
  message(
    STATUS
      "Generated sparse MLA kernel sources: ${_src_count} files (config: ${VLLM_SPARSE_MLA_CONFIG})"
  )

  set(${OUT_SRCS}
      "${_srcs}"
      PARENT_SCOPE)
  set(${OUT_DEFINES}
      "${_defines}"
      PARENT_SCOPE)
endfunction()
