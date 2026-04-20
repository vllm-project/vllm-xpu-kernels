function(gemm_xe2_configure FILENAME_SUFFIX)
  set(GEN_KERNEL_SRCS)
  set(TILE_LIST
      "8"
      "16"
      "32"
      "64"
      "128"
      "256"
      "128n"
      "256n"
      "384")
  set(LAYOUT_LIST "RR" "RC")

  foreach(IMPL_TILE ${TILE_LIST})
    foreach(IMPL_LAYOUT ${LAYOUT_LIST})
      set(FILE_SUFFIX "_${IMPL_TILE}_${IMPL_LAYOUT}")

      configure_file(${FILENAME_SUFFIX}.cpp.in
                     "${FILENAME_SUFFIX}${FILE_SUFFIX}.cpp")
      list(APPEND GEN_KERNEL_SRCS
           "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_SUFFIX}${FILE_SUFFIX}.cpp")
    endforeach()
  endforeach()

  list(REMOVE_DUPLICATES GEN_KERNEL_SRCS)
  list(LENGTH GEN_KERNEL_SRCS GEN_KERNEL_SRCS_LENGTH)
  message(STATUS "Generated GEMM kernel sources: ${GEN_KERNEL_SRCS_LENGTH}")

  set(GEN_KERNEL_SRCS
      ${GEN_KERNEL_SRCS}
      PARENT_SCOPE)
  list(APPEND GEMM_KERNEL_SRCS_GEN ${GEN_KERNEL_SRCS})
  set(GEMM_KERNEL_SRCS_GEN
      ${GEMM_KERNEL_SRCS_GEN}
      PARENT_SCOPE)
endfunction()
