#pragma once
#include <sycl/sycl.hpp>
#include "xpu/rand/heads/DistributionTemplates.h"

namespace TopkToppSamplerImpl {

// Percentile (in units of 0.5%, i.e. index = k/vocab*200) to Gaussian sigma
// multiplier. Used to estimate a top-k pivot from the sampled mean/std so the
// pivot search only has to scan gathered outliers instead of the full vocab.
// Mirrors _PERCENTILE_TO_STD_TABLE in vLLM's topk_topp_triton.py.
static constexpr float kPercentileToStdTable[200] = {
    2.576f,  2.319f,  2.178f,  2.064f,  1.968f,  1.892f,  1.819f,  1.757f,
    1.708f,  1.659f,  1.616f,  1.568f,  1.526f,  1.492f,  1.456f,  1.420f,
    1.382f,  1.342f,  1.309f,  1.280f,  1.249f,  1.221f,  1.193f,  1.169f,
    1.145f,  1.121f,  1.095f,  1.073f,  1.050f,  1.030f,  1.008f,  0.987f,
    0.966f,  0.945f,  0.926f,  0.910f,  0.891f,  0.871f,  0.854f,  0.837f,
    0.819f,  0.803f,  0.784f,  0.767f,  0.753f,  0.734f,  0.719f,  0.702f,
    0.690f,  0.675f,  0.658f,  0.640f,  0.625f,  0.609f,  0.595f,  0.578f,
    0.564f,  0.550f,  0.537f,  0.521f,  0.509f,  0.495f,  0.481f,  0.466f,
    0.453f,  0.439f,  0.424f,  0.410f,  0.397f,  0.383f,  0.370f,  0.356f,
    0.343f,  0.330f,  0.316f,  0.302f,  0.289f,  0.274f,  0.261f,  0.247f,
    0.235f,  0.223f,  0.209f,  0.196f,  0.184f,  0.172f,  0.159f,  0.149f,
    0.137f,  0.124f,  0.112f,  0.100f,  0.086f,  0.074f,  0.062f,  0.050f,
    0.035f,  0.023f,  0.009f,  -0.003f, -0.015f, -0.027f, -0.039f, -0.052f,
    -0.063f, -0.074f, -0.085f, -0.097f, -0.109f, -0.122f, -0.134f, -0.147f,
    -0.158f, -0.171f, -0.184f, -0.196f, -0.210f, -0.223f, -0.235f, -0.248f,
    -0.261f, -0.275f, -0.289f, -0.302f, -0.317f, -0.328f, -0.341f, -0.353f,
    -0.368f, -0.382f, -0.396f, -0.410f, -0.426f, -0.439f, -0.452f, -0.465f,
    -0.480f, -0.493f, -0.507f, -0.521f, -0.537f, -0.551f, -0.568f, -0.582f,
    -0.597f, -0.614f, -0.628f, -0.643f, -0.658f, -0.673f, -0.691f, -0.706f,
    -0.721f, -0.738f, -0.754f, -0.769f, -0.789f, -0.808f, -0.824f, -0.838f,
    -0.857f, -0.877f, -0.893f, -0.912f, -0.929f, -0.947f, -0.965f, -0.983f,
    -1.003f, -1.027f, -1.050f, -1.070f, -1.092f, -1.117f, -1.139f, -1.162f,
    -1.189f, -1.216f, -1.241f, -1.272f, -1.300f, -1.330f, -1.367f, -1.404f,
    -1.441f, -1.485f, -1.523f, -1.564f, -1.607f, -1.658f, -1.710f, -1.778f,
    -1.832f, -1.901f, -1.978f, -2.068f, -2.174f, -2.325f, -2.577f, -3.813f};

static constexpr int kMaxBisectIters = 40;

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
    const int local_offset =
        sycl::min(local_id * local_handle_size, vocab_size);
    const int remained_size = vocab_size - local_offset;
    const int handle_size =
        sycl::max(sycl::min(local_handle_size, remained_size), 0);

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
      float* buffer,
      const int64_t* top_k,
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
    const int local_offset =
        sycl::min(local_id * local_handle_size, vocab_size);
    const int remained_size = vocab_size - local_offset;
    const int handle_size =
        sycl::max(sycl::min(local_handle_size, remained_size), 0);

    int64_t* random_sampled_ptr = random_sampled + batch_id;
    float* logits_ptr = logits + batch_id * vocab_size + local_offset;
    float* buffer_ptr = buffer + batch_id * vocab_size + local_offset;
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

    // low, high, sample mean/std (for outlier pivot estimate) and max value
    // for softmax.
    float sum_logit = 0.0f;
    float sum_sq_logit = 0.0f;
    int finite_count = 0;
    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];

        if (!sycl::isfinite(logit)) continue;

        sum_logit += logit;
        sum_sq_logit += logit * logit;
        ++finite_count;

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

        sum_logit += logit;
        sum_sq_logit += logit * logit;
        ++finite_count;

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
    sum_logit = sycl::reduce_over_group(group, sum_logit, sycl::plus<>());
    sum_sq_logit = sycl::reduce_over_group(group, sum_sq_logit, sycl::plus<>());
    finite_count = sycl::reduce_over_group(group, finite_count, sycl::plus<>());
    pivot = low;
    max_softmax_value = high;

    if (!sycl::isfinite(max_softmax_value)) {
      max_softmax_value = INFINITY;
    }

    // Estimate an outlier pivot from the Gaussian statistics and gather logits
    // above it into this work-item's slice of the scratch buffer. When the
    // estimate captures at least top_k_value elements the pivot search scans
    // only these outliers instead of the full vocab. Falls back to the full
    // scan otherwise.
    bool use_outliers = false;
    int outlier_size = 0;
    float outlier_pivot = -INFINITY;
    if ((top_k_value != vocab_size) &&
        (sycl::isfinite(low) && sycl::isfinite(high)) && finite_count > 0) {
      float mean = sum_logit / finite_count;
      float var = sum_sq_logit / finite_count - mean * mean;
      float std_logit = sycl::sqrt(sycl::fmax(var, 0.0f));

      int percentile = static_cast<int>(
          static_cast<float>(top_k_value) / vocab_size * 200.0f);
      percentile = sycl::min(percentile, 199);
      percentile = sycl::max(percentile, 0);
      float sigma = kPercentileToStdTable[percentile];
      sigma = sigma + sycl::fabs(sigma) * -0.15f;
      outlier_pivot = mean + std_logit * sigma;

      for (int l = 0; l < loop_times; ++l) {
#pragma unroll
        for (int e = 0; e < VEC_SIZE; ++e) {
          local_data[e] = logits_ptr[l * VEC_SIZE + e];
        }

#pragma unroll
        for (int e = 0; e < VEC_SIZE; ++e) {
          float logit = local_data[e];
          if (logit > outlier_pivot) {
            buffer_ptr[outlier_size] = logit;
            ++outlier_size;
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
          if (logit > outlier_pivot) {
            buffer_ptr[outlier_size] = logit;
            ++outlier_size;
          }
        }
      }

      int num_outliers =
          sycl::reduce_over_group(group, outlier_size, sycl::plus<>());
      use_outliers = (num_outliers >= top_k_value);
    }

    const int outlier_loop_count = (outlier_size + VEC_SIZE - 1) / VEC_SIZE;
    const int outlier_remained =
        outlier_size - (outlier_loop_count - 1) * VEC_SIZE;
    const int outlier_loop_times = (outlier_remained == VEC_SIZE)
                                       ? outlier_loop_count
                                       : (outlier_loop_count - 1);
    const bool outlier_has_last =
        (outlier_size > 0) && (outlier_remained != VEC_SIZE);

    // topk
    if ((top_k_value != vocab_size) &&
        (sycl::isfinite(low) && sycl::isfinite(high))) {
      if (use_outliers) {
        low = outlier_pivot;
      }
      int iter = 0;
      do {
        int pivot_count_local = 0;

        pivot = (low + high) / 2;

        if (use_outliers) {
          for (int l = 0; l < outlier_loop_times; ++l) {
#pragma unroll
            for (int e = 0; e < VEC_SIZE; ++e) {
              local_data[e] = buffer_ptr[l * VEC_SIZE + e];
            }

#pragma unroll
            for (int e = 0; e < VEC_SIZE; ++e) {
              if (local_data[e] >= pivot) {
                pivot_count_local += 1;
              }
            }
          }

          if (outlier_has_last) {
#pragma unroll
            for (int e = 0; e < outlier_remained; ++e) {
              local_data[e] = buffer_ptr[outlier_loop_times * VEC_SIZE + e];
            }

#pragma unroll
            for (int e = 0; e < outlier_remained; ++e) {
              if (local_data[e] >= pivot) {
                pivot_count_local += 1;
              }
            }
          }
        } else {
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
        ++iter;
      } while (((high - low) > eps) && iter < kMaxBisectIters);

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
  float* buffer;
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
    const int local_offset =
        sycl::min(local_id * local_handle_size, vocab_size);
    const int remained_size = vocab_size - local_offset;
    const int handle_size =
        sycl::max(sycl::min(local_handle_size, remained_size), 0);

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
      int iter = 0;
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
            float prob = sycl::native::exp(local_data[e] - max_softmax_value) /
                         sum_softmax;

            if (prob >= pivot) {
              pivot_count_local += prob;
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
            float prob = sycl::native::exp(local_data[e] - max_softmax_value) /
                         sum_softmax;

            if (prob >= pivot) {
              pivot_count_local += prob;
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
        ++iter;
      } while (((high - low) > eps) && iter < kMaxBisectIters);

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
    const int local_offset =
        sycl::min(local_id * local_handle_size, vocab_size);
    const int remained_size = vocab_size - local_offset;
    const int handle_size =
        sycl::max(sycl::min(local_handle_size, remained_size), 0);

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

    // low, high, sample mean/std (for outlier pivot estimate) and max value
    // for softmax. mean/std are computed from this work-item's slice and then
    // reduced over the group; this approximates the full-row statistics.
    float sum_logit = 0.0f;
    float sum_sq_logit = 0.0f;
    int finite_count = 0;
    for (int l = 0; l < loop_times; ++l) {
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        local_data[e] = logits_ptr[l * VEC_SIZE + e];
      }

#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float logit = local_data[e];

        if (!sycl::isfinite(logit)) continue;

        sum_logit += logit;
        sum_sq_logit += logit * logit;
        ++finite_count;

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

        sum_logit += logit;
        sum_sq_logit += logit * logit;
        ++finite_count;

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
    sum_logit = sycl::reduce_over_group(group, sum_logit, sycl::plus<>());
    sum_sq_logit = sycl::reduce_over_group(group, sum_sq_logit, sycl::plus<>());
    finite_count = sycl::reduce_over_group(group, finite_count, sycl::plus<>());
    pivot_k = low_k;
    max_softmax_value = high_k;

    if (!sycl::isfinite(max_softmax_value)) {
      max_softmax_value = INFINITY;
    }

    // Estimate an outlier pivot from the Gaussian statistics and gather the
    // logits above it into this work-item's own slice of the scratch buffer.
    // If the estimate captures at least top_k_value elements, the pivot binary
    // search below scans only these gathered outliers instead of the full
    // vocab, which is the main speedup over the naive full-scan bisection.
    // Each work-item compacts into buffer_ptr (offset by its slice), so no
    // cross-item scan is needed; the search then reduces per-item counts.
    bool use_outliers = false;
    int outlier_size = 0;
    float outlier_pivot = -INFINITY;
    if ((top_k_value != vocab_size) &&
        (sycl::isfinite(low_k) && sycl::isfinite(high_k)) && finite_count > 0) {
      float mean = sum_logit / finite_count;
      float var = sum_sq_logit / finite_count - mean * mean;
      float std_logit = sycl::sqrt(sycl::fmax(var, 0.0f));

      int percentile = static_cast<int>(
          static_cast<float>(top_k_value) / vocab_size * 200.0f);
      percentile = sycl::min(percentile, 199);
      percentile = sycl::max(percentile, 0);
      float sigma = kPercentileToStdTable[percentile];
      sigma = sigma + sycl::fabs(sigma) * -0.15f;
      outlier_pivot = mean + std_logit * sigma;

      for (int l = 0; l < loop_times; ++l) {
#pragma unroll
        for (int e = 0; e < VEC_SIZE; ++e) {
          local_data[e] = logits_ptr[l * VEC_SIZE + e];
        }

#pragma unroll
        for (int e = 0; e < VEC_SIZE; ++e) {
          float logit = local_data[e];
          if (logit > outlier_pivot) {
            buffer_ptr[outlier_size] = logit;
            ++outlier_size;
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
          if (logit > outlier_pivot) {
            buffer_ptr[outlier_size] = logit;
            ++outlier_size;
          }
        }
      }

      int num_outliers =
          sycl::reduce_over_group(group, outlier_size, sycl::plus<>());
      use_outliers = (num_outliers >= top_k_value);
    }

    const int outlier_loop_count = (outlier_size + VEC_SIZE - 1) / VEC_SIZE;
    const int outlier_remained =
        outlier_size - (outlier_loop_count - 1) * VEC_SIZE;
    const int outlier_loop_times = (outlier_remained == VEC_SIZE)
                                       ? outlier_loop_count
                                       : (outlier_loop_count - 1);
    const bool outlier_has_last =
        (outlier_size > 0) && (outlier_remained != VEC_SIZE);

    // topk
    if ((top_k_value != vocab_size) &&
        (sycl::isfinite(low_k) && sycl::isfinite(high_k))) {
      int pivot_count_k = top_k_value;
      if (use_outliers) {
        low_k = outlier_pivot;
      }
      int iter = 0;
      do {
        int pivot_count_local = 0;

        pivot_k = (low_k + high_k) / 2;

        if (use_outliers) {
          // Scan only the gathered outliers in the per-item buffer slice.
          for (int l = 0; l < outlier_loop_times; ++l) {
#pragma unroll
            for (int e = 0; e < VEC_SIZE; ++e) {
              local_data[e] = buffer_ptr[l * VEC_SIZE + e];
            }

#pragma unroll
            for (int e = 0; e < VEC_SIZE; ++e) {
              if (local_data[e] >= pivot_k) {
                pivot_count_local += 1;
              }
            }
          }

          if (outlier_has_last) {
#pragma unroll
            for (int e = 0; e < outlier_remained; ++e) {
              local_data[e] = buffer_ptr[outlier_loop_times * VEC_SIZE + e];
            }

#pragma unroll
            for (int e = 0; e < outlier_remained; ++e) {
              if (local_data[e] >= pivot_k) {
                pivot_count_local += 1;
              }
            }
          }
        } else {
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
        ++iter;
      } while (((high_k - low_k) > eps) && iter < kMaxBisectIters);

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
      int iter_p = 0;
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
        ++iter_p;
      } while (((high_p - low_p) > eps) && iter_p < kMaxBisectIters);

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
          buffer,
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
