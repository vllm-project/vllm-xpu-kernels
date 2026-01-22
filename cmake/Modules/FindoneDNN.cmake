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

  set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party")
  set(ONEDNN_DIR "oneDNN")
  set(ONEDNN_ROOT "${THIRD_PARTY_DIR}/${ONEDNN_DIR}")

  find_path(
    ONEDNN_INCLUDE_DIR dnnl.hpp dnnl.h
    PATHS ${ONEDNN_ROOT}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)
  if(NOT ONEDNN_INCLUDE_DIR)
    find_package(Git)
    if(NOT Git_FOUND)
      message(FATAL_ERROR "Can not find Git executable!")
    endif()
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init ${ONEDNN_DIR}
      WORKING_DIRECTORY ${THIRD_PARTY_DIR} COMMAND_ERROR_IS_FATAL ANY)
    find_path(
      ONEDNN_INCLUDE_DIR dnnl.hpp dnnl.h
      PATHS ${ONEDNN_ROOT}
      PATH_SUFFIXES include
      NO_DEFAULT_PATH)
  endif(NOT ONEDNN_INCLUDE_DIR)

  if(NOT ONEDNN_INCLUDE_DIR)
    message(FATAL_ERROR "oneDNN source files not found!")
  endif(NOT ONEDNN_INCLUDE_DIR)

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

  add_subdirectory(${ONEDNN_ROOT} oneDNN EXCLUDE_FROM_ALL)
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
  list(FILTER ONEDNN_INCLUDE_DIR EXCLUDE REGEX
       ".*third_party/oneDNN/third_party.*")

  set(ONEDNN_FOUND ON)
  message(STATUS "Found oneDNN: TRUE")

endif(NOT ONEDNN_FOUND)
