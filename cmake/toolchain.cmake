# use this file to set the compiler and flags for SYCL

set(CMPLR_ROOT $ENV{CMPLR_ROOT})
message(STATUS "CMPLR_ROOT: ${CMPLR_ROOT}")
set(CMAKE_CXX_COMPILER ${CMPLR_ROOT}/bin/icpx)
set(CMAKE_C_COMPILER ${CMPLR_ROOT}/bin/icx)