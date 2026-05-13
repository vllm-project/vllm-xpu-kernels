function(fmha_forward_configure FILENAME_SUFFIX)
  set(GEN_KERNEL_SRCS) # output
  set(L_BOOLS "false" "true")
  set(BOOL_FLAG_false "f")
  set(BOOL_FLAG_true "t")
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

  # Valid (Q, KV) dtype combinations. Q dictates O; KV dictates K and V.
  # Encoded as "Q_TYPE|KV_TYPE|FILENAME_TOKEN" — keep in sync with the
  # CHUNK_DTYPE_COMBINATIONS X-macro in chunk_prefill_extern.hpp.
  set(dtype_combos
      "half_t|half_t|hh"
      "half_t|float_e4m3_t|h4"
      "half_t|float_e5m2_t|h5"
      "bfloat16_t|bfloat16_t|bb"
      "bfloat16_t|float_e4m3_t|b4"
      "bfloat16_t|float_e5m2_t|b5")

  foreach(IMPL_POLICY ${policy_list})
    foreach(IMPL_KISPAGED ${L_BOOLS})
      foreach(IMPL_KISCAUSAL ${L_BOOLS})
        foreach(IMPL_KISLOCAL ${L_BOOLS})
          foreach(IMPL_KISSINK ${L_BOOLS})
            # softmax_lse is only supported on the (Paged=false, Local=false,
            # Sink=false) specialization. For all other combos force
            # IMPL_KISLSE=false to keep the number of generated TUs bounded.
            set(LSE_BOOLS "false")
            if(IMPL_KISPAGED STREQUAL "false"
               AND IMPL_KISLOCAL STREQUAL "false"
               AND IMPL_KISSINK STREQUAL "false")
              set(LSE_BOOLS ${L_BOOLS})
            endif()
            foreach(IMPL_KISLSE ${LSE_BOOLS})
              foreach(_combo ${dtype_combos})
                string(REPLACE "|" ";" _combo_list "${_combo}")
                list(GET _combo_list 0 IMPL_Q_DTYPE)
                list(GET _combo_list 1 IMPL_KV_DTYPE)
                list(GET _combo_list 2 DTYPE_TOKEN)

                set(FILE_SUFFIX "${IMPL_POLICY}_")
                set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISPAGED}}")
                set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISCAUSAL}}")
                set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISSINK}}")
                set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISLOCAL}}")
                set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISLSE}}")
                set(FILE_SUFFIX "${FILE_SUFFIX}_${DTYPE_TOKEN}")
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
