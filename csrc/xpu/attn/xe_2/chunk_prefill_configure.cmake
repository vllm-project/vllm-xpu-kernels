function(fmha_forward_configure FILENAME_SUFFIX)
  set(GEN_KERNEL_SRCS) # output
  set(GEN_KERNEL_SRCS_FP8Q)
  set(GEN_KERNEL_SRCS_NON_FP8Q)
  set(L_BOOLS "false" "true")
  set(BOOL_FLAG_false "f")
  set(BOOL_FLAG_true "t")
  set(policy_list
      "chunk_policy_head64" "chunk_policy_head96" "chunk_policy_head128"
      "chunk_policy_head192" "chunk_policy_head256")

  # Allowed dtype combinations must match runtime dispatch constraints. Format:
  # Q_TYPE|KV_TYPE|O_TYPE|FILE_TAG
  set(dtype_combo_list
      "half_t|half_t|half_t|h_h_h"
      "half_t|float_e4m3_t|half_t|h_e4_h"
      "half_t|float_e5m2_t|half_t|h_e5_h"
      "bfloat16_t|bfloat16_t|bfloat16_t|b_b_b"
      "bfloat16_t|float_e4m3_t|bfloat16_t|b_e4_b"
      "bfloat16_t|float_e5m2_t|bfloat16_t|b_e5_b"
      "float_e4m3_t|float_e4m3_t|half_t|e4_e4_h"
      "float_e4m3_t|float_e4m3_t|bfloat16_t|e4_e4_b"
      "float_e5m2_t|float_e5m2_t|half_t|e5_e5_h"
      "float_e5m2_t|float_e5m2_t|bfloat16_t|e5_e5_b")

  foreach(IMPL_POLICY ${policy_list})
    foreach(dtype_combo ${dtype_combo_list})
      string(REPLACE "|" ";" dtype_parts "${dtype_combo}")
      list(GET dtype_parts 0 IMPL_Q_T)
      list(GET dtype_parts 1 IMPL_KV_T)
      list(GET dtype_parts 2 IMPL_O_T)
      list(GET dtype_parts 3 DTYPE_TAG)

      foreach(IMPL_KISPAGED ${L_BOOLS})
        foreach(IMPL_KISCAUSAL ${L_BOOLS})
          foreach(IMPL_KISLOCAL ${L_BOOLS})
            foreach(IMPL_KISSINK ${L_BOOLS})
              set(FILE_SUFFIX "${IMPL_POLICY}_${DTYPE_TAG}_")
              set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISPAGED}}")
              set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISCAUSAL}}")
              set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISSINK}}")
              set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISLOCAL}}")
              configure_file(${FILENAME_SUFFIX}.cpp.in
                             "${FILENAME_SUFFIX}_${FILE_SUFFIX}.cpp")
              set(GEN_SRC
                  "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_SUFFIX}_${FILE_SUFFIX}.cpp"
              )
              list(APPEND GEN_KERNEL_SRCS ${GEN_SRC})

              if(DTYPE_TAG MATCHES "^(e4_e4|e5_e5)_")
                list(APPEND GEN_KERNEL_SRCS_FP8Q ${GEN_SRC})
              else()
                list(APPEND GEN_KERNEL_SRCS_NON_FP8Q ${GEN_SRC})
              endif()
            endforeach()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()

  list(REMOVE_DUPLICATES GEN_KERNEL_SRCS)
  list(REMOVE_DUPLICATES GEN_KERNEL_SRCS_FP8Q)
  list(REMOVE_DUPLICATES GEN_KERNEL_SRCS_NON_FP8Q)
  list(LENGTH GEN_KERNEL_SRCS GEN_KERNEL_SRCS_LENGTH)
  list(LENGTH GEN_KERNEL_SRCS_FP8Q GEN_KERNEL_SRCS_FP8Q_LENGTH)
  list(LENGTH GEN_KERNEL_SRCS_NON_FP8Q GEN_KERNEL_SRCS_NON_FP8Q_LENGTH)
  message(
    STATUS
      "Generated ${FILENAME_SUFFIX} kernel sources: ${GEN_KERNEL_SRCS_LENGTH}")
  message(
    STATUS
      "Generated ${FILENAME_SUFFIX} fp8-q kernel sources: ${GEN_KERNEL_SRCS_FP8Q_LENGTH}"
  )
  message(
    STATUS
      "Generated ${FILENAME_SUFFIX} non-fp8-q kernel sources: ${GEN_KERNEL_SRCS_NON_FP8Q_LENGTH}"
  )
  set(GEN_KERNEL_SRCS
      ${GEN_KERNEL_SRCS}
      PARENT_SCOPE)
  set(GEN_KERNEL_SRCS_FP8Q
      ${GEN_KERNEL_SRCS_FP8Q}
      PARENT_SCOPE)
  set(GEN_KERNEL_SRCS_NON_FP8Q
      ${GEN_KERNEL_SRCS_NON_FP8Q}
      PARENT_SCOPE)
  set(GEN_KERNEL_SRCS_LENGTH
      ${GEN_KERNEL_SRCS_LENGTH}
      PARENT_SCOPE)

  list(APPEND ATTN_KERNEL_SRCS_GEN ${GEN_KERNEL_SRCS})
  set(ATTN_KERNEL_SRCS_GEN
      ${ATTN_KERNEL_SRCS_GEN}
      PARENT_SCOPE)

  set(ATTN_KERNEL_SRCS_GEN_FP8Q
      ${GEN_KERNEL_SRCS_FP8Q}
      PARENT_SCOPE)
  set(ATTN_KERNEL_SRCS_GEN_NON_FP8Q
      ${GEN_KERNEL_SRCS_NON_FP8Q}
      PARENT_SCOPE)
endfunction()
