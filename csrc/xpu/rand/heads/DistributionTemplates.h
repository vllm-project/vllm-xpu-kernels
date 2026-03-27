#pragma once
#include "Philox4x32.h"

namespace RAND {
template <typename scalar_t, typename accscalar_t>
struct ExponentialFunctor {
  auto operator()(accscalar_t val) const {
    // BEFORE TOUCHING THIS CODE READ:
    // https://github.com/pytorch/pytorch/issues/16706
    // rand_uniform has (0,1] bounds. log(1) is 0 and exponential
    // excludes 0. we need log to be not 0, and not underflow when
    // converted to half
    accscalar_t log;
    if (val >= static_cast<accscalar_t>(1.f) -
                   std::numeric_limits<scalar_t>::epsilon() / 2.f) {
      log = -std::numeric_limits<scalar_t>::epsilon() / 2.f;
    } else {
      log = std::log(val);
    }
    return static_cast<accscalar_t>(-1.f) / lambd_ * log;
  }
  ExponentialFunctor(accscalar_t lambd) : lambd_(lambd) {}

 private:
  accscalar_t lambd_;
};

// for double
struct Uniform2DistributionFunctor {
  auto operator()(randStatePhilox4_32_10_t* state) const {
    return rand_uniform2_double(state);
  }
};

// for float
struct Uniform4DistributionFunctor {
  auto operator()(randStatePhilox4_32_10_t* state) const {
    return rand_uniform4(state);
  }
};
}  // namespace RAND