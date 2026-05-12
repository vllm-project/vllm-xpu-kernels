#pragma once
#include <sycl/sycl.hpp>
#include "xpu/rand/heads/DistributionTemplates.h"

namespace TopkToppSamplerImpl {

enum class LogprobsMode {
  default_mode,
  raw_logits,
  raw_logprobs,
  processed_logits,
  processed_logprobs
};

template <LogprobsMode logprobs_mode>
struct random_sampler_only_kernel {
 public:
  static constexpr int sub_group_size = 16;
  static constexpr int group_size = 512;
  static constexpr int VEC_SIZE = 4;

  using scalar_t = float;
  using acc_scalar_t = float;

  random_sampler_only_kernel(
      int64_t* random_sampled,
      float* logits_to_return,
      float* logits,
      const int batch_size,
      const int vocab_size,
      const int64_t seed,
      const int64_t offset,
      const float lambda)
      : random_sampled(random_sampled),
        logits_to_return(logits_to_return),
        logits(logits),
        batch_size(batch_size),
        vocab_size(vocab_size),
        seed(seed),
        offset(offset),
        lambda(lambda) {}

  static inline sycl::nd_range<1>
  get_nd_range(const int batch_size, const int vocab_size) {
    int local_size = group_size;
    if (vocab_size < group_size) {
      local_size =
          (vocab_size + sub_group_size - 1) / sub_group_size * sub_group_size;
    }
    sycl::range<1> local(local_size);
    sycl::range<1> global(batch_size);
    return sycl::nd_range<1>(global * local, local);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<1> item) const {
    const int batch_id = item.get_group(0);
    const int local_id = item.get_local_linear_id();
    const int local_range = item.get_local_range(0);

    const int global_id = item.get_global_linear_id();
    uint64_t philox_seed = seed;
    uint64_t philox_offset = offset;
    RAND::randStatePhilox4_32_10_t state;
    RAND::rand_init(philox_seed, global_id, philox_offset, &state);

    RAND::Uniform4DistributionFunctor dist_func;
    RAND::ExponentialFunctor<scalar_t, acc_scalar_t> exponential_func(lambda);

    auto group = item.get_group();

    const int local_handle_size = (vocab_size + local_range - 1) / local_range;
    const int local_offset = local_id * local_handle_size;
    const int remained_size = vocab_size - local_offset;
    const int handle_size = sycl::min(local_handle_size, remained_size);

    int64_t* random_sampled_ptr = random_sampled + batch_id;
    float* logits_ptr = logits + batch_id * vocab_size + local_offset;
    float* logits_to_return_ptr =
        logits_to_return + batch_id * vocab_size + local_offset;

    float local_data[VEC_SIZE];
    const int loop_count = (handle_size + VEC_SIZE - 1) / VEC_SIZE;
    const int remained_vec_size = handle_size - (loop_count - 1) * VEC_SIZE;
    const int loop_times =
        (remained_vec_size == VEC_SIZE) ? loop_count : (loop_count - 1);
    const bool has_last_loop = (remained_vec_size == VEC_SIZE) ? false : true;

    float max_softmax_value = -INFINITY;

    // low, high, and max value for softmax
    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];

        if (logit > max_softmax_value) {
          max_softmax_value = logit;
        }
      }
    }

    if (has_last_loop) {
#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        float logit = local_data[e];

        if (logit > max_softmax_value) {
          max_softmax_value = logit;
        }
      }
    }

    max_softmax_value =
        sycl::reduce_over_group(group, max_softmax_value, sycl::maximum<>());

    // get sum_softmax after mask with pivot
    float sum_softmax = 0.0f;
    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];
        sum_softmax += sycl::native::exp(logit - max_softmax_value);
      }
    }

    if (has_last_loop) {
#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        float logit = local_data[e];
        sum_softmax += sycl::native::exp(logit - max_softmax_value);
      }
    }

    sum_softmax = sycl::reduce_over_group(group, sum_softmax, sycl::plus<>());

    float max_value_local = -INFINITY;
    int max_idx_local = 0;

    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

      auto rand4 = dist_func(&state);

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];
        float rand = exponential_func(static_cast<acc_scalar_t>((&rand4.x)[e]));

        if constexpr (logprobs_mode == LogprobsMode::processed_logits) {
          logits_to_return_ptr[l * VEC_SIZE + e] = logit;
        }
        logit = sycl::native::exp(logit - max_softmax_value) / sum_softmax;
        if constexpr (logprobs_mode == LogprobsMode::processed_logprobs) {
          logits_to_return_ptr[l * VEC_SIZE + e] = sycl::log(logit);
        }
        logit /= rand;
        if (logit > max_value_local) {
          max_value_local = logit;
          max_idx_local = local_offset + l * VEC_SIZE + e;
        }
      }
    }

    if (has_last_loop) {
#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
      }

      auto rand4 = dist_func(&state);

#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        float logit = local_data[e];
        float rand = exponential_func(static_cast<acc_scalar_t>((&rand4.x)[e]));

        if constexpr (logprobs_mode == LogprobsMode::processed_logits) {
          logits_to_return_ptr[loop_times * VEC_SIZE + e] = logit;
        }
        logit = sycl::native::exp(logit - max_softmax_value) / sum_softmax;
        if constexpr (logprobs_mode == LogprobsMode::processed_logprobs) {
          logits_to_return_ptr[loop_times * VEC_SIZE + e] = sycl::log(logit);
        }
        logit /= rand;
        if (logit > max_value_local) {
          max_value_local = logit;
          max_idx_local = local_offset + loop_times * VEC_SIZE + e;
        }
      }
    }

    float max_val_global =
        sycl::reduce_over_group(group, max_value_local, sycl::maximum<>());
    bool is_max = (max_val_global == max_value_local);
    int64_t first_max_id = sycl::reduce_over_group(
        group, is_max ? max_idx_local : (vocab_size - 1), sycl::minimum<>());

    if (0 == local_id) {
      random_sampled_ptr[0] = first_max_id;
    }
  }

 private:
  int64_t* random_sampled;
  float* logits_to_return;
  float* logits;
  const int batch_size;
  const int vocab_size;
  const int64_t seed;
  const int64_t offset;
  const float lambda;
};

template <LogprobsMode logprobs_mode>
struct top_k_only_kernel {
 public:
  static constexpr int sub_group_size = 16;
  static constexpr int group_size = 512;
  static constexpr int VEC_SIZE = 4;

  using scalar_t = float;
  using acc_scalar_t = float;

  top_k_only_kernel(
      int64_t* random_sampled,
      float* logits_to_return,
      float* logits,
      const int64_t* top_k,
      const int batch_size,
      const int vocab_size,
      const int64_t seed,
      const int64_t offset,
      const float lambda)
      : random_sampled(random_sampled),
        logits_to_return(logits_to_return),
        logits(logits),
        top_k(top_k),
        batch_size(batch_size),
        vocab_size(vocab_size),
        seed(seed),
        offset(offset),
        lambda(lambda) {}

  static inline sycl::nd_range<1>
  get_nd_range(const int batch_size, const int vocab_size) {
    int local_size = group_size;
    if (vocab_size < group_size) {
      local_size =
          (vocab_size + sub_group_size - 1) / sub_group_size * sub_group_size;
    }
    sycl::range<1> local(local_size);
    sycl::range<1> global(batch_size);
    return sycl::nd_range<1>(global * local, local);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<1> item) const {
    const int batch_id = item.get_group(0);
    const int local_id = item.get_local_linear_id();
    const int local_range = item.get_local_range(0);

    const int global_id = item.get_global_linear_id();
    uint64_t philox_seed = seed;
    uint64_t philox_offset = offset;
    RAND::randStatePhilox4_32_10_t state;
    RAND::rand_init(philox_seed, global_id, philox_offset, &state);

    RAND::Uniform4DistributionFunctor dist_func;
    RAND::ExponentialFunctor<scalar_t, acc_scalar_t> exponential_func(lambda);

    auto group = item.get_group();

    const int top_k_value = top_k[batch_id];

    const int local_handle_size = (vocab_size + local_range - 1) / local_range;
    const int local_offset = local_id * local_handle_size;
    const int remained_size = vocab_size - local_offset;
    const int handle_size = sycl::min(local_handle_size, remained_size);

    int64_t* random_sampled_ptr = random_sampled + batch_id;
    float* logits_ptr = logits + batch_id * vocab_size + local_offset;
    float* logits_to_return_ptr =
        logits_to_return + batch_id * vocab_size + local_offset;

    double low = INFINITY, high = -INFINITY;
    double pivot = -INFINITY;
    int pivot_count = top_k_value;
    double eps = 1e-9;

    float local_data[VEC_SIZE];
    const int loop_count = (handle_size + VEC_SIZE - 1) / VEC_SIZE;
    const int remained_vec_size = handle_size - (loop_count - 1) * VEC_SIZE;
    const int loop_times =
        (remained_vec_size == VEC_SIZE) ? loop_count : (loop_count - 1);
    const bool has_last_loop = (remained_vec_size == VEC_SIZE) ? false : true;

    float max_softmax_value = -INFINITY;

    // low, high, and max value for softmax
    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];

        if (!sycl::isfinite(logit)) continue;

        if (logit < low) {
          low = logit;
        }

        if (logit > high) {
          high = logit;
        }
      }
    }

    if (has_last_loop) {
#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        float logit = local_data[e];

        if (!sycl::isfinite(logit)) continue;

        if (logit < low) {
          low = logit;
        }

        if (logit > high) {
          high = logit;
        }
      }
    }

    low = sycl::reduce_over_group(group, low, sycl::minimum<>());
    high = sycl::reduce_over_group(group, high, sycl::maximum<>());
    pivot = low;
    max_softmax_value = high;

    if (!sycl::isfinite(max_softmax_value)) {
      max_softmax_value = INFINITY;
    }

    // topk
    if ((top_k_value != vocab_size) &&
        (sycl::isfinite(low) && sycl::isfinite(high))) {
      do {
        int pivot_count_local = 0;

        pivot = (low + high) / 2;

        for (int l = 0; l < loop_times; ++l) {
#pragma unroll
          for (int e = 0; e < VEC_SIZE; ++e) {
            local_data[e] = logits_ptr[l * VEC_SIZE + e];
          }

#pragma unroll
          for (int e = 0; e < VEC_SIZE; ++e) {
            float logit = local_data[e];

            if (logit >= pivot) {
              pivot_count_local += 1;
            }
          }
        }

        if (has_last_loop) {
#pragma unroll
          for (int e = 0; e < remained_vec_size; ++e) {
            local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
          }

#pragma unroll
          for (int e = 0; e < remained_vec_size; ++e) {
            float logit = local_data[e];

            if (logit >= pivot) {
              pivot_count_local += 1;
            }
          }
        }

        pivot_count =
            sycl::reduce_over_group(group, pivot_count_local, sycl::plus<>());

        if (pivot_count == top_k_value) {
          break;
        } else if (pivot_count < top_k_value) {
          high = pivot;
        } else {
          low = pivot;
        }
      } while (((high - low) > eps));

      if (pivot_count < top_k_value) {
        pivot = low;
      }
    }

    // get sum_softmax after mask with pivot
    float sum_softmax = 0.0f;
    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];

        if (logit >= pivot) {
          sum_softmax += sycl::native::exp(logit - max_softmax_value);
        }
      }
    }

    if (has_last_loop) {
#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        float logit = local_data[e];

        if (logit >= pivot) {
          sum_softmax += sycl::native::exp(logit - max_softmax_value);
        }
      }
    }

    sum_softmax = sycl::reduce_over_group(group, sum_softmax, sycl::plus<>());

    float max_value_local = -INFINITY;
    int max_idx_local = 0;

    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

      auto rand4 = dist_func(&state);

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];
        float rand = exponential_func(static_cast<acc_scalar_t>((&rand4.x)[e]));

        if (logit >= pivot) {
          if constexpr (logprobs_mode == LogprobsMode::processed_logits) {
            logits_to_return_ptr[l * VEC_SIZE + e] = logit;
          }
          logit = sycl::native::exp(logit - max_softmax_value) / sum_softmax;
          if constexpr (logprobs_mode == LogprobsMode::processed_logprobs) {
            logits_to_return_ptr[l * VEC_SIZE + e] = sycl::log(logit);
          }
          logit /= rand;
          if (logit > max_value_local) {
            max_value_local = logit;
            max_idx_local = local_offset + l * VEC_SIZE + e;
          }
        } else {
          if constexpr (
              logprobs_mode == LogprobsMode::processed_logits ||
              logprobs_mode == LogprobsMode::processed_logprobs) {
            logits_to_return_ptr[l * VEC_SIZE + e] = -INFINITY;
          }
        }
      }
    }

    if (has_last_loop) {
#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
      }

      auto rand4 = dist_func(&state);

#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        float logit = local_data[e];
        float rand = exponential_func(static_cast<acc_scalar_t>((&rand4.x)[e]));

        if (logit >= pivot) {
          if constexpr (logprobs_mode == LogprobsMode::processed_logits) {
            logits_to_return_ptr[loop_times * VEC_SIZE + e] = logit;
          }
          logit = sycl::native::exp(logit - max_softmax_value) / sum_softmax;
          if constexpr (logprobs_mode == LogprobsMode::processed_logprobs) {
            logits_to_return_ptr[loop_times * VEC_SIZE + e] = sycl::log(logit);
          }
          logit /= rand;
          if (logit > max_value_local) {
            max_value_local = logit;
            max_idx_local = local_offset + loop_times * VEC_SIZE + e;
          }
        } else {
          if constexpr (
              logprobs_mode == LogprobsMode::processed_logits ||
              logprobs_mode == LogprobsMode::processed_logprobs) {
            logits_to_return_ptr[loop_times * VEC_SIZE + e] = -INFINITY;
          }
        }
      }
    }

    float max_val_global =
        sycl::reduce_over_group(group, max_value_local, sycl::maximum<>());
    bool is_max = (max_val_global == max_value_local);
    int64_t first_max_id = sycl::reduce_over_group(
        group, is_max ? max_idx_local : (vocab_size - 1), sycl::minimum<>());

    if (0 == local_id) {
      random_sampled_ptr[0] = first_max_id;
    }
  }

 private:
  int64_t* random_sampled;
  float* logits_to_return;
  float* logits;
  const int64_t* top_k;
  const int batch_size;
  const int vocab_size;
  const int64_t seed;
  const int64_t offset;
  const float lambda;
};

template <LogprobsMode logprobs_mode>
struct top_p_only_kernel {
 public:
  static constexpr int sub_group_size = 16;
  static constexpr int group_size = 512;
  static constexpr int VEC_SIZE = 4;

  using scalar_t = float;
  using acc_scalar_t = float;

  top_p_only_kernel(
      int64_t* random_sampled,
      float* logits_to_return,
      float* logits,
      const float* top_p,
      const int batch_size,
      const int vocab_size,
      const int64_t seed,
      const int64_t offset,
      const float lambda)
      : random_sampled(random_sampled),
        logits_to_return(logits_to_return),
        logits(logits),
        top_p(top_p),
        batch_size(batch_size),
        vocab_size(vocab_size),
        seed(seed),
        offset(offset),
        lambda(lambda) {}

  static inline sycl::nd_range<1>
  get_nd_range(const int batch_size, const int vocab_size) {
    int local_size = group_size;
    if (vocab_size < group_size) {
      local_size =
          (vocab_size + sub_group_size - 1) / sub_group_size * sub_group_size;
    }
    sycl::range<1> local(local_size);
    sycl::range<1> global(batch_size);
    return sycl::nd_range<1>(global * local, local);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<1> item) const {
    const int batch_id = item.get_group(0);
    const int local_id = item.get_local_linear_id();
    const int local_range = item.get_local_range(0);

    const int global_id = item.get_global_linear_id();
    uint64_t philox_seed = seed;
    uint64_t philox_offset = offset;
    RAND::randStatePhilox4_32_10_t state;
    RAND::rand_init(philox_seed, global_id, philox_offset, &state);

    RAND::Uniform4DistributionFunctor dist_func;
    RAND::ExponentialFunctor<scalar_t, acc_scalar_t> exponential_func(lambda);

    auto group = item.get_group();

    const float top_p_value = top_p[batch_id];

    const int local_handle_size = (vocab_size + local_range - 1) / local_range;
    const int local_offset = local_id * local_handle_size;
    const int remained_size = vocab_size - local_offset;
    const int handle_size = sycl::min(local_handle_size, remained_size);

    int64_t* random_sampled_ptr = random_sampled + batch_id;
    float* logits_ptr = logits + batch_id * vocab_size + local_offset;
    float* logits_to_return_ptr =
        logits_to_return + batch_id * vocab_size + local_offset;

    double low = INFINITY, high = -INFINITY;
    double pivot = -INFINITY;
    float pivot_count = top_p_value;
    double eps = 1e-9;

    float local_data[VEC_SIZE];
    const int loop_count = (handle_size + VEC_SIZE - 1) / VEC_SIZE;
    const int remained_vec_size = handle_size - (loop_count - 1) * VEC_SIZE;
    const int loop_times =
        (remained_vec_size == VEC_SIZE) ? loop_count : (loop_count - 1);
    const bool has_last_loop = (remained_vec_size == VEC_SIZE) ? false : true;

    float max_softmax_value = -INFINITY;

    // low, high, and max value for softmax
    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];

        if (logit < low) {
          low = logit;
        }

        if (logit > high) {
          high = logit;
        }
      }
    }

    if (has_last_loop) {
#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        float logit = local_data[e];

        if (logit < low) {
          low = logit;
        }

        if (logit > high) {
          high = logit;
        }
      }
    }

    low = sycl::reduce_over_group(group, low, sycl::minimum<>());
    high = sycl::reduce_over_group(group, high, sycl::maximum<>());
    max_softmax_value = high;

    // get sum_softmax after mask without pivot
    float sum_softmax = 0.0f;
    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];
        sum_softmax += sycl::native::exp(logit - max_softmax_value);
      }
    }

    if (has_last_loop) {
#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        float logit = local_data[e];
        sum_softmax += sycl::native::exp(logit - max_softmax_value);
      }
    }

    sum_softmax = sycl::reduce_over_group(group, sum_softmax, sycl::plus<>());
    low = sycl::native::exp(low - max_softmax_value) / sum_softmax;
    high = sycl::native::exp(high - max_softmax_value) / sum_softmax;
    pivot = low;

    // topp
    if (top_p_value != 1.0f) {
      float low_count = 1.0f;
      do {
        float pivot_count_local = 0.0f;

        pivot = (low + high) / 2;

        for (int l = 0; l < loop_times; ++l) {
#pragma unroll
          for (int e = 0; e < VEC_SIZE; ++e) {
            local_data[e] = logits_ptr[l * VEC_SIZE + e];
          }

#pragma unroll
          for (int e = 0; e < VEC_SIZE; ++e) {
            float logit = local_data[e];
            logit = sycl::native::exp(logit - max_softmax_value) / sum_softmax;

            if (logit >= pivot) {
              pivot_count_local += logit;
            }
          }
        }

        if (has_last_loop) {
#pragma unroll
          for (int e = 0; e < remained_vec_size; ++e) {
            local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
          }

#pragma unroll
          for (int e = 0; e < remained_vec_size; ++e) {
            float logit = local_data[e];
            logit = sycl::native::exp(logit - max_softmax_value) / sum_softmax;

            if (logit >= pivot) {
              pivot_count_local += logit;
            }
          }
        }

        pivot_count =
            sycl::reduce_over_group(group, pivot_count_local, sycl::plus<>());

        if (pivot_count == top_p_value) {
          break;
        } else if (pivot_count < top_p_value) {
          high = pivot;
        } else {
          low = pivot;
          low_count = pivot_count;
        }

      } while ((high - low) > eps);

      if (pivot_count < top_p_value) {
        pivot = low;
        pivot_count = low_count;
      }
    }

    float max_value_local = -INFINITY;
    int max_idx_local = 0;

    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

      auto rand4 = dist_func(&state);

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];
        float logit_softmax =
            sycl::native::exp(logit - max_softmax_value) / sum_softmax;
        float rand = exponential_func(static_cast<acc_scalar_t>((&rand4.x)[e]));

        if (logit_softmax >= pivot) {
          if constexpr (logprobs_mode == LogprobsMode::processed_logits) {
            logits_to_return_ptr[l * VEC_SIZE + e] = logit;
          }
          if constexpr (logprobs_mode == LogprobsMode::processed_logprobs) {
            logits_to_return_ptr[l * VEC_SIZE + e] =
                sycl::log(logit_softmax / pivot_count);
          }
          logit_softmax /= rand;
          if (logit_softmax > max_value_local) {
            max_value_local = logit_softmax;
            max_idx_local = local_offset + l * VEC_SIZE + e;
          }
        } else {
          if constexpr (
              logprobs_mode == LogprobsMode::processed_logits ||
              logprobs_mode == LogprobsMode::processed_logprobs) {
            logits_to_return_ptr[l * VEC_SIZE + e] = -INFINITY;
          }
        }
      }
    }

    if (has_last_loop) {
#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
      }

      auto rand4 = dist_func(&state);

#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        float logit = local_data[e];
        float logit_softmax =
            sycl::native::exp(logit - max_softmax_value) / sum_softmax;
        float rand = exponential_func(static_cast<acc_scalar_t>((&rand4.x)[e]));

        if (logit_softmax >= pivot) {
          if constexpr (logprobs_mode == LogprobsMode::processed_logits) {
            logits_to_return_ptr[loop_times * VEC_SIZE + e] = logit;
          }
          if constexpr (logprobs_mode == LogprobsMode::processed_logprobs) {
            logits_to_return_ptr[loop_times * VEC_SIZE + e] =
                sycl::log(logit_softmax / pivot_count);
          }
          logit_softmax /= rand;
          if (logit_softmax > max_value_local) {
            max_value_local = logit_softmax;
            max_idx_local = local_offset + loop_times * VEC_SIZE + e;
          }
        } else {
          if constexpr (
              logprobs_mode == LogprobsMode::processed_logits ||
              logprobs_mode == LogprobsMode::processed_logprobs) {
            logits_to_return_ptr[loop_times * VEC_SIZE + e] = -INFINITY;
          }
        }
      }
    }

    float max_val_global =
        sycl::reduce_over_group(group, max_value_local, sycl::maximum<>());
    bool is_max = (max_val_global == max_value_local);
    int64_t first_max_id = sycl::reduce_over_group(
        group, is_max ? max_idx_local : (vocab_size - 1), sycl::minimum<>());

    if (0 == local_id) {
      random_sampled_ptr[0] = first_max_id;
    }
  }

 private:
  int64_t* random_sampled;
  float* logits_to_return;
  float* logits;
  const float* top_p;
  const int batch_size;
  const int vocab_size;
  const int64_t seed;
  const int64_t offset;
  const float lambda;
};

template <LogprobsMode logprobs_mode>
struct top_k_top_p_kernel {
 public:
  static constexpr int sub_group_size = 16;
  static constexpr int group_size = 512;
  static constexpr int VEC_SIZE = 4;

  using scalar_t = float;
  using acc_scalar_t = float;

  top_k_top_p_kernel(
      int64_t* random_sampled,
      float* logits_to_return,
      float* logits,
      float* buffer,
      const int64_t* top_k,
      const float* top_p,
      const int batch_size,
      const int vocab_size,
      const int64_t seed,
      const int64_t offset,
      const float lambda)
      : random_sampled(random_sampled),
        logits_to_return(logits_to_return),
        logits(logits),
        buffer(buffer),
        top_k(top_k),
        top_p(top_p),
        batch_size(batch_size),
        vocab_size(vocab_size),
        seed(seed),
        offset(offset),
        lambda(lambda) {}

  static inline sycl::nd_range<1>
  get_nd_range(const int batch_size, const int vocab_size) {
    int local_size = group_size;
    if (vocab_size < group_size) {
      local_size =
          (vocab_size + sub_group_size - 1) / sub_group_size * sub_group_size;
    }
    sycl::range<1> local(local_size);
    sycl::range<1> global(batch_size);
    return sycl::nd_range<1>(global * local, local);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<1> item) const {
    const int batch_id = item.get_group(0);
    const int local_id = item.get_local_linear_id();
    const int local_range = item.get_local_range(0);

    const int global_id = item.get_global_linear_id();
    uint64_t philox_seed = seed;
    uint64_t philox_offset = offset;
    RAND::randStatePhilox4_32_10_t state;
    RAND::rand_init(philox_seed, global_id, philox_offset, &state);

    RAND::Uniform4DistributionFunctor dist_func;
    RAND::ExponentialFunctor<scalar_t, acc_scalar_t> exponential_func(lambda);

    auto group = item.get_group();

    const int top_k_value = top_k[batch_id];
    const float top_p_value = top_p[batch_id];

    const int local_handle_size = (vocab_size + local_range - 1) / local_range;
    const int local_offset = local_id * local_handle_size;
    const int remained_size = vocab_size - local_offset;
    const int handle_size = sycl::min(local_handle_size, remained_size);

    int64_t* random_sampled_ptr = random_sampled + batch_id;
    float* logits_ptr = logits + batch_id * vocab_size + local_offset;
    float* buffer_ptr = buffer + batch_id * vocab_size + local_offset;
    float* logits_to_return_ptr =
        logits_to_return + batch_id * vocab_size + local_offset;

    double low_k = INFINITY, high_k = -INFINITY;
    double pivot_k = -INFINITY;
    double eps = 1e-9;

    float local_data[VEC_SIZE];
    const int loop_count = (handle_size + VEC_SIZE - 1) / VEC_SIZE;
    const int remained_vec_size = handle_size - (loop_count - 1) * VEC_SIZE;
    const int loop_times =
        (remained_vec_size == VEC_SIZE) ? loop_count : (loop_count - 1);
    const bool has_last_loop = (remained_vec_size == VEC_SIZE) ? false : true;

    float max_softmax_value = -INFINITY;

    // low, high, and max value for softmax
    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];

        if (!sycl::isfinite(logit)) continue;

        if (logit < low_k) {
          low_k = logit;
        }

        if (logit > high_k) {
          high_k = logit;
        }
      }
    }

    if (has_last_loop) {
#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        float logit = local_data[e];

        if (!sycl::isfinite(logit)) continue;

        if (logit < low_k) {
          low_k = logit;
        }

        if (logit > high_k) {
          high_k = logit;
        }
      }
    }

    low_k = sycl::reduce_over_group(group, low_k, sycl::minimum<>());
    high_k = sycl::reduce_over_group(group, high_k, sycl::maximum<>());
    pivot_k = low_k;
    max_softmax_value = high_k;

    if (!sycl::isfinite(max_softmax_value)) {
      max_softmax_value = INFINITY;
    }

    // topk
    if ((top_k_value != vocab_size) &&
        (sycl::isfinite(low_k) && sycl::isfinite(high_k))) {
      int pivot_count_k = top_k_value;
      do {
        int pivot_count_local = 0;

        pivot_k = (low_k + high_k) / 2;

        for (int l = 0; l < loop_times; ++l) {
#pragma unroll
          for (int e = 0; e < VEC_SIZE; ++e) {
            local_data[e] = logits_ptr[l * VEC_SIZE + e];
          }

#pragma unroll
          for (int e = 0; e < VEC_SIZE; ++e) {
            float logit = local_data[e];

            if (logit >= pivot_k) {
              pivot_count_local += 1;
            }
          }
        }

        if (has_last_loop) {
#pragma unroll
          for (int e = 0; e < remained_vec_size; ++e) {
            local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
          }

#pragma unroll
          for (int e = 0; e < remained_vec_size; ++e) {
            float logit = local_data[e];

            if (logit >= pivot_k) {
              pivot_count_local += 1;
            }
          }
        }

        pivot_count_k =
            sycl::reduce_over_group(group, pivot_count_local, sycl::plus<>());

        if (pivot_count_k == top_k_value) {
          break;
        } else if (pivot_count_k < top_k_value) {
          high_k = pivot_k;
        } else {
          low_k = pivot_k;
        }
      } while (((high_k - low_k) > eps));

      if (pivot_count_k < top_k_value) {
        pivot_k = low_k;
      }
    }

    // get sum_softmax after mask with pivot_k
    float sum_softmax = 0.0f;
    int buffer_size = 0;
    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];

        if (logit >= pivot_k) {
          logit = sycl::native::exp(logit - max_softmax_value);
          sum_softmax += logit;
          buffer_ptr[buffer_size] = logit;
          ++buffer_size;
        }
      }
    }

    if (has_last_loop) {
#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        float logit = local_data[e];

        if (logit >= pivot_k) {
          logit = sycl::native::exp(logit - max_softmax_value);
          sum_softmax += logit;
          buffer_ptr[buffer_size] = logit;
          ++buffer_size;
        }
      }
    }

    const int loop_count_buffer = (buffer_size + VEC_SIZE - 1) / VEC_SIZE;
    const int remained_vec_size_buffer =
        buffer_size - (loop_count_buffer - 1) * VEC_SIZE;
    const int loop_times_buffer = (remained_vec_size_buffer == VEC_SIZE)
                                      ? loop_count_buffer
                                      : (loop_count_buffer - 1);
    const bool has_last_loop_buffer =
        (remained_vec_size_buffer == VEC_SIZE) ? false : true;

    sum_softmax = sycl::reduce_over_group(group, sum_softmax, sycl::plus<>());
    double low_p = sycl::native::exp(low_k - max_softmax_value) / sum_softmax;
    double high_p =
        sycl::native::exp(max_softmax_value - max_softmax_value) / sum_softmax;
    double pivot_p = low_p;
    float pivot_count_p = top_p_value;

    // topp
    if (top_p_value != 1.0f) {
      float low_count = 1.0f;
      do {
        float pivot_count_local = 0.0f;

        pivot_p = (low_p + high_p) / 2;

        for (int l = 0; l < loop_times_buffer; ++l) {
#pragma unroll
          for (int e = 0; e < VEC_SIZE; ++e) {
            local_data[e] = buffer_ptr[l * VEC_SIZE + e];
          }

#pragma unroll
          for (int e = 0; e < VEC_SIZE; ++e) {
            float logit = local_data[e];
            logit /= sum_softmax;

            if (logit >= pivot_p) {
              pivot_count_local += logit;
            }
          }
        }

        if (has_last_loop_buffer) {
#pragma unroll
          for (int e = 0; e < remained_vec_size_buffer; ++e) {
            local_data[e] = buffer_ptr[loop_times_buffer * VEC_SIZE + e];
          }

#pragma unroll
          for (int e = 0; e < remained_vec_size_buffer; ++e) {
            float logit = local_data[e];
            logit /= sum_softmax;

            if (logit >= pivot_p) {
              pivot_count_local += logit;
            }
          }
        }

        pivot_count_p =
            sycl::reduce_over_group(group, pivot_count_local, sycl::plus<>());

        if (pivot_count_p == top_p_value) {
          break;
        } else if (pivot_count_p < top_p_value) {
          high_p = pivot_p;
        } else {
          low_p = pivot_p;
          low_count = pivot_count_p;
        }

      } while ((high_p - low_p) > eps);

      if (pivot_count_p < top_p_value) {
        pivot_p = low_p;
        pivot_count_p = low_count;
      }
    }

    float max_value_local = -INFINITY;
    int max_idx_local = 0;

    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

      auto rand4 = dist_func(&state);

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];
        float logit_softmax =
            sycl::native::exp(logit - max_softmax_value) / sum_softmax;
        float rand = exponential_func(static_cast<acc_scalar_t>((&rand4.x)[e]));

        if (logit >= pivot_k && logit_softmax >= pivot_p) {
          if constexpr (logprobs_mode == LogprobsMode::processed_logits) {
            logits_to_return_ptr[l * VEC_SIZE + e] = logit;
          }
          if constexpr (logprobs_mode == LogprobsMode::processed_logprobs) {
            logits_to_return_ptr[l * VEC_SIZE + e] =
                sycl::log(logit_softmax / pivot_count_p);
          }
          logit_softmax /= rand;
          if (logit_softmax > max_value_local) {
            max_value_local = logit_softmax;
            max_idx_local = local_offset + l * VEC_SIZE + e;
          }
        } else {
          if constexpr (
              logprobs_mode == LogprobsMode::processed_logits ||
              logprobs_mode == LogprobsMode::processed_logprobs) {
            logits_to_return_ptr[l * VEC_SIZE + e] = -INFINITY;
          }
        }
      }
    }

    if (has_last_loop) {
#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
      }

      auto rand4 = dist_func(&state);

#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
        float logit = local_data[e];
        float logit_softmax =
            sycl::native::exp(logit - max_softmax_value) / sum_softmax;
        float rand = exponential_func(static_cast<acc_scalar_t>((&rand4.x)[e]));

        if (logit >= pivot_k && logit_softmax >= pivot_p) {
          if constexpr (logprobs_mode == LogprobsMode::processed_logits) {
            logits_to_return_ptr[loop_times * VEC_SIZE + e] = logit;
          }
          if constexpr (logprobs_mode == LogprobsMode::processed_logprobs) {
            logits_to_return_ptr[loop_times * VEC_SIZE + e] =
                sycl::log(logit_softmax / pivot_count_p);
          }
          logit_softmax /= rand;
          if (logit_softmax > max_value_local) {
            max_value_local = logit_softmax;
            max_idx_local = local_offset + loop_times * VEC_SIZE + e;
          }
        } else {
          if constexpr (
              logprobs_mode == LogprobsMode::processed_logits ||
              logprobs_mode == LogprobsMode::processed_logprobs) {
            logits_to_return_ptr[loop_times * VEC_SIZE + e] = -INFINITY;
          }
        }
      }
    }

    float max_val_global =
        sycl::reduce_over_group(group, max_value_local, sycl::maximum<>());
    bool is_max = (max_val_global == max_value_local);
    int64_t first_max_id = sycl::reduce_over_group(
        group, is_max ? max_idx_local : (vocab_size - 1), sycl::minimum<>());

    if (0 == local_id) {
      random_sampled_ptr[0] = first_max_id;
    }
  }

 private:
  int64_t* random_sampled;
  float* logits_to_return;
  float* logits;
  float* buffer;
  const int64_t* top_k;
  const float* top_p;
  const int batch_size;
  const int vocab_size;
  const int64_t seed;
  const int64_t offset;
  const float lambda;
};

template <LogprobsMode logprobs_mode>
void topk_topp_sampler_kernel_launcher(
    sycl::queue& queue,
    int64_t* random_sampled,
    float* logits_to_return,
    float* logits,
    float* buffer,
    const int64_t* top_k,
    const float* top_p,
    const int batch_size,
    const int vocab_size,
    const int64_t seed,
    const int64_t offset,
    const float lambda) {
  if (top_k != nullptr && top_p == nullptr) {
    // launch top_k_only_kernel
    using KERNEL_TOPK_ONLY = top_k_only_kernel<logprobs_mode>;
    auto range = KERNEL_TOPK_ONLY::get_nd_range(batch_size, vocab_size);
    queue.submit([&](sycl::handler& cgh) {
      KERNEL_TOPK_ONLY task(
          random_sampled,
          logits_to_return,
          logits,
          top_k,
          batch_size,
          vocab_size,
          seed,
          offset,
          lambda);
      cgh.parallel_for(range, task);
    });
  } else if (top_k == nullptr && top_p != nullptr) {
    // launch top_p_only_kernel
    using KERNEL_TOPP_ONLY = top_p_only_kernel<logprobs_mode>;
    auto range = KERNEL_TOPP_ONLY::get_nd_range(batch_size, vocab_size);
    queue.submit([&](sycl::handler& cgh) {
      KERNEL_TOPP_ONLY task(
          random_sampled,
          logits_to_return,
          logits,
          top_p,
          batch_size,
          vocab_size,
          seed,
          offset,
          lambda);
      cgh.parallel_for(range, task);
    });
  } else if (top_k != nullptr && top_p != nullptr) {
    // launch top_k_top_p_kernel
    using KERNEL_TOPK_TOPP = top_k_top_p_kernel<logprobs_mode>;
    auto range = KERNEL_TOPK_TOPP::get_nd_range(batch_size, vocab_size);
    queue.submit([&](sycl::handler& cgh) {
      KERNEL_TOPK_TOPP task(
          random_sampled,
          logits_to_return,
          logits,
          buffer,
          top_k,
          top_p,
          batch_size,
          vocab_size,
          seed,
          offset,
          lambda);
      cgh.parallel_for(range, task);
    });
  } else {
    // launch random_sampler_only_kernel
    using KERNEL_SAMPLER_ONLY = random_sampler_only_kernel<logprobs_mode>;
    auto range = KERNEL_SAMPLER_ONLY::get_nd_range(batch_size, vocab_size);
    queue.submit([&](sycl::handler& cgh) {
      KERNEL_SAMPLER_ONLY task(
          random_sampled,
          logits_to_return,
          logits,
          batch_size,
          vocab_size,
          seed,
          offset,
          lambda);
      cgh.parallel_for(range, task);
    });
  }
}
}  // namespace TopkToppSamplerImpl
