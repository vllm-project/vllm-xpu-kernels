function(fmha_forward_configure FILENAME_SUFFIX)
  set(GEN_KERNEL_SRCS) # output
  set(L_BOOLS "false" "true")
  set(BOOL_FLAG_false "f")
  set(BOOL_FLAG_true "t")
  set(policy_list
      "chunk_policy_head64" "chunk_policy_head96" "chunk_policy_head128"
      "chunk_policy_head192" "chunk_policy_head256")

  # Dtype combinations as parallel lists (aligned by index)
  set(DTYPE_EQ half_t half_t half_t bfloat16_t bfloat16_t bfloat16_t)
  set(DTYPE_EK half_t float_e4m3_t float_e5m2_t bfloat16_t float_e4m3_t
               float_e5m2_t)
  set(DTYPE_EV half_t float_e4m3_t float_e5m2_t bfloat16_t float_e4m3_t
               float_e5m2_t)
  set(DTYPE_EO half_t half_t half_t bfloat16_t bfloat16_t bfloat16_t)
  set(DTYPE_TAGS hh he4 he5 bb be4 be5)
  list(LENGTH DTYPE_TAGS DTYPE_COUNT)
  math(EXPR DTYPE_LAST "${DTYPE_COUNT} - 1")

  foreach(IMPL_POLICY ${policy_list})
    foreach(IMPL_KISPAGED ${L_BOOLS})
      foreach(IMPL_KISCAUSAL ${L_BOOLS})
        foreach(IMPL_KISLOCAL ${L_BOOLS})
          foreach(IMPL_KISSINK ${L_BOOLS})
            foreach(DTYPE_IDX RANGE 0 ${DTYPE_LAST})
              list(GET DTYPE_EQ ${DTYPE_IDX} IMPL_EQ)
              list(GET DTYPE_EK ${DTYPE_IDX} IMPL_EK)
              list(GET DTYPE_EV ${DTYPE_IDX} IMPL_EV)
              list(GET DTYPE_EO ${DTYPE_IDX} IMPL_EO)
              list(GET DTYPE_TAGS ${DTYPE_IDX} DTYPE_TAG)

              set(FILE_SUFFIX "${IMPL_POLICY}_")
              set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISPAGED}}")
              set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISCAUSAL}}")
              set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISSINK}}")
              set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISLOCAL}}")
              set(FILE_SUFFIX "${FILE_SUFFIX}_${DTYPE_TAG}")
              configure_file(${FILENAME_SUFFIX}.cpp.in
                             "${FILENAME_SUFFIX}_${FILE_SUFFIX}.cpp")
              list(
                APPEND
                GEN_KERNEL_SRCS
                "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_SUFFIX}_${FILE_SUFFIX}.cpp"
              )
            endforeach()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()

  list(REMOVE_DUPLICATES GEN_KERNEL_SRCS)
  list(LENGTH GEN_KERNEL_SRCS GEN_KERNEL_SRCS_LENGTH)
  message(
    STATUS
      "Generated ${FILENAME_SUFFIX} kernel sources: ${GEN_KERNEL_SRCS_LENGTH}")
  set(GEN_KERNEL_SRCS
      ${GEN_KERNEL_SRCS}
      PARENT_SCOPE)
  set(GEN_KERNEL_SRCS_LENGTH
      ${GEN_KERNEL_SRCS_LENGTH}
      PARENT_SCOPE)

  list(APPEND ATTN_KERNEL_SRCS_GEN ${GEN_KERNEL_SRCS})
  set(ATTN_KERNEL_SRCS_GEN
      ${ATTN_KERNEL_SRCS_GEN}
      PARENT_SCOPE)

endfunction()
