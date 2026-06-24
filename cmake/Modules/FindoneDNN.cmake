# * Try to find oneDNN
#
# The following are set after configuration is done: ONEDNN_FOUND          : set
# to true if oneDNN is found. ONEDNN_INCLUDE_DIR    : path to oneDNN include
# dir. ONEDNN_LIBRARY        : list of libraries for oneDNN
#

if(NOT ONEDNN_FOUND)
  set(ONEDNN_FOUND OFF)

  set(ONEDNN_LIBRARY)
  set(ONEDNN_INCLUDE_DIR)
  set(DNNL_INCLUDES)

  include(FetchContent)

  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()

  message(
    STATUS
      "oneDNN: fetching from ${ONEDNN_GIT_REPO} (commit: ${ONEDNN_GIT_TAG})")
  FetchContent_Declare(
    oneDNN
    GIT_REPOSITORY ${ONEDNN_GIT_REPO}
    GIT_TAG ${ONEDNN_GIT_TAG}
    GIT_PROGRESS TRUE
    GIT_SHALLOW FALSE)

  set(DNNL_ENABLE_PRIMITIVE_CACHE
      TRUE
      CACHE BOOL "oneDNN sycl primitive cache" FORCE)

  set(DNNL_LIBRARY_TYPE
      STATIC
      CACHE STRING "" FORCE)

  set(DNNL_CPU_RUNTIME
      "NONE"
      CACHE STRING "oneDNN cpu backend" FORCE)
  set(DNNL_GPU_RUNTIME
      "SYCL"
      CACHE STRING "oneDNN gpu backend" FORCE)
  set(DNNL_BUILD_TESTS
      FALSE
      CACHE BOOL "build with oneDNN tests" FORCE)
  set(DNNL_BUILD_EXAMPLES
      FALSE
      CACHE BOOL "build with oneDNN examples" FORCE)
  set(DNNL_ENABLE_CONCURRENT_EXEC
      TRUE
      CACHE BOOL "multi-thread primitive execution" FORCE)
  set(DNNL_EXPERIMENTAL
      TRUE
      CACHE BOOL "use one pass for oneDNN BatchNorm" FORCE)

  FetchContent_Populate(oneDNN)
  add_subdirectory(${onednn_SOURCE_DIR} ${onednn_BINARY_DIR} EXCLUDE_FROM_ALL)

  set(ONEDNN_LIBRARY ${DNNL_LIBRARY_NAME})
  if(NOT TARGET ${ONEDNN_LIBRARY})
    message(FATAL_ERROR "Failed to include oneDNN target")
  endif(NOT TARGET ${ONEDNN_LIBRARY})

  if(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
    target_compile_options(${ONEDNN_LIBRARY} PRIVATE -Wno-uninitialized)
    target_compile_options(${ONEDNN_LIBRARY} PRIVATE -Wno-strict-overflow)
    target_compile_options(${ONEDNN_LIBRARY} PRIVATE -Wno-error=strict-overflow)
  endif(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)

  target_compile_options(${ONEDNN_LIBRARY} PRIVATE -Wno-tautological-compare)
  get_target_property(DNNL_INCLUDES ${ONEDNN_LIBRARY} INCLUDE_DIRECTORIES)
  list(APPEND ONEDNN_INCLUDE_DIR ${DNNL_INCLUDES})

  # Upper level targets should not load header files from oneDNN's third party.
  list(FILTER ONEDNN_INCLUDE_DIR EXCLUDE REGEX ".*/third_party.*")

  set(ONEDNN_FOUND ON)
  message(STATUS "Found oneDNN: TRUE")

endif(NOT ONEDNN_FOUND)
