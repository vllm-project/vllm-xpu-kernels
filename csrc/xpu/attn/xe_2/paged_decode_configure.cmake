# =============================================================================
# Paged Decode Kernel Configuration
# =============================================================================
# This function generates kernel source files for all combinations of: - Policy
# types (q_group_size × head_size) - Boolean flags (Causal, Local, Sink)
#
# Each generated file instantiates one specific kernel configuration to enable
# parallel compilation and reduce individual object file sizes.
#
# Usage: paged_decode_configure(paged_decode_kernel_template)
#
# Parameters: FILENAME_SUFFIX - Base name for generated .cpp files (without
# extension)
#
# Output: GEN_KERNEL_SRCS - List of generated source file paths
# GEN_KERNEL_SRCS_LENGTH - Number of generated files ATTN_KERNEL_SRCS_GEN -
# Updated global list with appended sources
# =============================================================================

function(paged_decode_configure FILENAME_SUFFIX)
  set(GEN_KERNEL_SRCS) # Initialize output list

  # Boolean flag values and their single-character representations
  set(L_BOOLS "false" "true")
  set(BOOL_FLAG_false "f")
  set(BOOL_FLAG_true "t")

  # =============================================================================
  # Policy Configuration Mapping
  # =============================================================================
  # Maps (q_group_size, head_size) pairs to policy type names These must match
  # the policies defined in paged_decode_policy.hpp

  # Q-group size 8 policies
  set(policy_8_64 "decode_policy_q8_h64")
  set(policy_8_96 "decode_policy_q8_h96")
  set(policy_8_128 "decode_policy_q8_h128")
  set(policy_8_192 "decode_policy_q8_h192")
  set(policy_8_256 "decode_policy_q8_h256")

  # Q-group size 16 policies
  set(policy_16_64 "decode_policy_q16_h64")
  set(policy_16_96 "decode_policy_q16_h96")
  set(policy_16_128 "decode_policy_q16_h128")
  set(policy_16_192 "decode_policy_q16_h192")
  set(policy_16_256 "decode_policy_q16_h256")

  # Configuration space dimensions
  set(qgroup_list "8" "16")
  set(headsize_list "64" "96" "128" "192" "256")

  # =============================================================================
  # Generate Kernel Sources
  # =============================================================================
  # Iterate over all combinations: policy × causal × local × sink

  foreach(IMPL_QGROUP ${qgroup_list})
    foreach(IMPL_HEADSIZE ${headsize_list})
      # Lookup policy name from mapping
      set(IMPL_POLICY ${policy_${IMPL_QGROUP}_${IMPL_HEADSIZE}})

      foreach(IMPL_KISCAUSAL ${L_BOOLS})
        foreach(IMPL_KISLOCAL ${L_BOOLS})
          foreach(IMPL_KISSINK ${L_BOOLS})
            # Construct unique filename suffix: e.g., _q8_h64_fff
            set(FILE_SUFFIX "_q${IMPL_QGROUP}_h${IMPL_HEADSIZE}_")
            set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISCAUSAL}}")
            set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISLOCAL}}")
            set(FILE_SUFFIX "${FILE_SUFFIX}${BOOL_FLAG_${IMPL_KISSINK}}")

            # Generate .cpp file from template
            configure_file(${FILENAME_SUFFIX}.cpp.in
                           "${FILENAME_SUFFIX}${FILE_SUFFIX}.cpp")

            # Add to output list
            list(
              APPEND GEN_KERNEL_SRCS
              "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_SUFFIX}${FILE_SUFFIX}.cpp"
            )
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()

  # =============================================================================
  # Output Results
  # =============================================================================

  list(REMOVE_DUPLICATES GEN_KERNEL_SRCS)
  list(LENGTH GEN_KERNEL_SRCS GEN_KERNEL_SRCS_LENGTH)

  message(
    STATUS
      "Generated ${FILENAME_SUFFIX} sources: ${GEN_KERNEL_SRCS_LENGTH} files")

  # Export to parent scope
  set(GEN_KERNEL_SRCS
      ${GEN_KERNEL_SRCS}
      PARENT_SCOPE)
  set(GEN_KERNEL_SRCS_LENGTH
      ${GEN_KERNEL_SRCS_LENGTH}
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
