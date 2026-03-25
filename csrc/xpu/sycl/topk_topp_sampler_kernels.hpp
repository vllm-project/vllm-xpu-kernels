#include <sycl/sycl.hpp>
#include "Philox4x32.h"

namespace TopkToppSamplerImpl {

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

enum class LogprobsMode{
    default_mode,
    raw_logits,
    raw_logprobs,
    processed_logits,
    processed_logprobs
};

struct top_k_mask_logits_kernel {
 public:
  static constexpr int sub_group_size = 16;
  static constexpr int group_size = 512;
  static constexpr int VEC_SIZE = 4;

  top_k_mask_logits_kernel(
      float* logits,
      const int64_t* top_k,
      const int batch_size,
      const int vocal_size)
      : logits(logits),
        top_k(top_k),
        batch_size(batch_size),
        vocal_size(vocal_size)
        {}

  static inline sycl::nd_range<1>
  get_nd_range(const int batch_size, const int vocal_size) {
    int local_size = group_size;
    if(vocal_size < group_size){
        local_size = (vocal_size + sub_group_size - 1) / sub_group_size * sub_group_size;
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

    auto group = item.get_group();

    const int top_k_value = top_k[batch_id];

    const int local_handle_size = (vocal_size + local_range - 1) / local_range;
    const int local_offset = local_id * local_handle_size;
    const int remained_size = vocal_size - local_offset;
    const int handle_size = sycl::min(local_handle_size, remained_size);

    float* logits_ptr = logits + batch_id * vocal_size + local_offset;

    double low = 0.0f, high = 1.f / static_cast<double>(top_k_value);
    double pivot = low;
    int pivot_count = top_k_value;
    double eps = 1e-9;

    float local_data[VEC_SIZE];
    const int loop_count = (handle_size + VEC_SIZE - 1) / VEC_SIZE;
    const int remained_vec_size = handle_size - (loop_count - 1) * VEC_SIZE;
    const int loop_times = (remained_vec_size == VEC_SIZE) ? loop_count : (loop_count - 1);
    const bool has_last_loop = (remained_vec_size == VEC_SIZE) ? false : true;
    
    if(top_k_value != vocal_size){
        do{
            int pivot_count_local = 0;

            for(int l = 0; l < loop_times; ++l){
                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    local_data[e] = logits_ptr[l * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    float logit = local_data[e] ;
                    
                    if(logit >= pivot){
                        pivot_count_local += 1;
                    }
                }
            }

            if(has_last_loop){
                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    float logit = local_data[e] ;
                    
                    if(logit >= pivot){
                        pivot_count_local += 1;
                    }
                }
            }

            pivot_count = sycl::reduce_over_group(group, pivot_count_local, sycl::plus<>());

            if(pivot_count == top_k_value){
                break;
            } else if(pivot_count < top_k_value){
                high = pivot;
            } else {
                low = pivot;
            }

            pivot = (low + high) / 2;
        } while (((high - low) > eps));
    }

    if(pivot_count != top_k_value){
        if(pivot_count > top_k_value){
            pivot = high;
        } else {
            pivot = low;
        }
    }

    for(int l = 0; l < loop_times; ++l){
        #pragma unroll
        for(int e = 0; e < VEC_SIZE; ++e){
            local_data[e] = logits_ptr[l * VEC_SIZE + e];
        }

        #pragma unroll
        for(int e = 0; e < VEC_SIZE; ++e){
            float logit = local_data[e];
            
            if(logit < pivot){
                logits_ptr[l * VEC_SIZE + e] = -INFINITY;
            }
        }
    }

    if(has_last_loop){
        #pragma unroll
        for(int e = 0; e < remained_vec_size; ++e){
            local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
        }

        #pragma unroll
        for(int e = 0; e < remained_vec_size; ++e){
            float logit = local_data[e];
            
            if(logit < pivot){
                logits_ptr[loop_times * VEC_SIZE + e] = -INFINITY;
            }
        }
    }
  }

 private:
    float* logits;
    const int64_t* top_k;
    const int batch_size;
    const int vocal_size;
};

void top_k_mask_logits_kernel_launcher(
    sycl::queue& queue,
    float* logits,
    const int64_t* top_k,
    const int batch_size,
    const int vocal_size) {

    // launch top_k_mask_logits_kernel
    std::cout << "Launch top_k_mask_logits_kernel" << std::endl;
    using KERNEL = top_k_mask_logits_kernel;
    auto range = KERNEL::get_nd_range(batch_size, vocal_size);
    queue.submit([&](sycl::handler& cgh) {
        KERNEL task(
            logits,
            top_k,
            batch_size,
            vocal_size);
        cgh.parallel_for(range, task);
    });
}

template <LogprobsMode logprobs_mode>
struct random_sampler_only_kernel {
 public:
  static constexpr int sub_group_size = 16;
  static constexpr int group_size = 512;
  static constexpr int VEC_SIZE = 4;

  using scalar_t = float;
  using accscalar_t = float;

  random_sampler_only_kernel(
      int64_t* random_sampled,
      float* logits_to_return,
      float* logits,
      const int batch_size,
      const int vocal_size,
      const int64_t seed,
      const int64_t offset,
      const float lambda)
      : random_sampled(random_sampled),
        logits_to_return(logits_to_return),
        logits(logits),
        batch_size(batch_size),
        vocal_size(vocal_size),
        seed(seed),
        offset(offset),
        lambda(lambda)
        {}

  static inline sycl::nd_range<1>
  get_nd_range(const int batch_size, const int vocal_size) {
    int local_size = group_size;
    if(vocal_size < group_size){
        local_size = (vocal_size + sub_group_size - 1) / sub_group_size * sub_group_size;
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
    randStatePhilox4_32_10_t state;
    rand_init(philox_seed, global_id, philox_offset, &state);

    Uniform4DistributionFunctor dist_func;
    ExponentialFunctor<scalar_t, accscalar_t> exponential_func(lambda);

    auto group = item.get_group();

    const int local_handle_size = (vocal_size + local_range - 1) / local_range;
    const int local_offset = local_id * local_handle_size;
    const int remained_size = vocal_size - local_offset;
    const int handle_size = sycl::min(local_handle_size, remained_size);

    int64_t* random_sampled_ptr = random_sampled + batch_id;
    float* logits_to_return_ptr = logits_to_return + batch_id * vocal_size + local_offset;
    float* logits_ptr = logits + batch_id * vocal_size + local_offset;

    float local_data[VEC_SIZE];
    const int loop_count = (handle_size + VEC_SIZE - 1) / VEC_SIZE;
    const int remained_vec_size = handle_size - (loop_count - 1) * VEC_SIZE;
    const int loop_times = (remained_vec_size == VEC_SIZE) ? loop_count : (loop_count - 1);
    const bool has_last_loop = (remained_vec_size == VEC_SIZE) ? false : true;

    float max_value_local = -INFINITY;
    int max_idx_local = 0;

    for(int l = 0; l < loop_times; ++l){
        #pragma unroll
        for(int e = 0; e < VEC_SIZE; ++e){
            local_data[e] = logits_ptr[l * VEC_SIZE + e];
        }

        auto rand4 = dist_func(&state);

        #pragma unroll
        for(int e = 0; e < VEC_SIZE; ++e){
            float logit = local_data[e];
            float rand = exponential_func(static_cast<accscalar_t>((&rand4.x)[e]));

            if constexpr (logprobs_mode == LogprobsMode::processed_logprobs){
                logits_to_return_ptr[l * VEC_SIZE + e] = sycl::log(logit);
            }
            
            logit /= rand;
            if(logit > max_value_local){
                max_value_local = logit;
                max_idx_local = local_offset + l * VEC_SIZE + e;
            }
        }
    }

    if(has_last_loop){
        #pragma unroll
        for(int e = 0; e < remained_vec_size; ++e){
            local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
        }

        auto rand4 = dist_func(&state);

        #pragma unroll
        for(int e = 0; e < remained_vec_size; ++e){
            float logit = local_data[e];
            float rand = exponential_func(static_cast<accscalar_t>((&rand4.x)[e]));

            if constexpr (logprobs_mode == LogprobsMode::processed_logprobs){
                logits_to_return_ptr[loop_times * VEC_SIZE + e] = sycl::log(logit);
            }
            
            logit /= rand;
            if(logit > max_value_local){
                max_value_local = logit;
                max_idx_local = local_offset + loop_times * VEC_SIZE + e;
            }
        }
    }


    float max_val_global =
        sycl::reduce_over_group(group, max_value_local, sycl::maximum<>());
    bool is_max = (max_val_global == max_value_local);
    int64_t first_max_id = sycl::reduce_over_group(
        group, is_max ? max_idx_local : (vocal_size - 1), sycl::minimum<>());

    if (0 == local_id) {
      random_sampled_ptr[0] = first_max_id;
    }

  }

 private:
    int64_t* random_sampled;
    float* logits_to_return;
    float* logits;
    const int batch_size;
    const int vocal_size;
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
  using accscalar_t = float;

  top_k_only_kernel(
      int64_t* random_sampled,
      float* logits_to_return,
      float* logits,
      const int64_t* top_k,
      const int batch_size,
      const int vocal_size,
      const int64_t seed,
      const int64_t offset,
      const float lambda)
      : random_sampled(random_sampled),
        logits_to_return(logits_to_return),
        logits(logits),
        top_k(top_k),
        batch_size(batch_size),
        vocal_size(vocal_size),
        seed(seed),
        offset(offset),
        lambda(lambda)
        {}

  static inline sycl::nd_range<1>
  get_nd_range(const int batch_size, const int vocal_size) {
    int local_size = group_size;
    if(vocal_size < group_size){
        local_size = (vocal_size + sub_group_size - 1) / sub_group_size * sub_group_size;
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
    randStatePhilox4_32_10_t state;
    rand_init(philox_seed, global_id, philox_offset, &state);

    Uniform4DistributionFunctor dist_func;
    ExponentialFunctor<scalar_t, accscalar_t> exponential_func(lambda);

    auto group = item.get_group();

    const int top_k_value = top_k[batch_id];

    const int local_handle_size = (vocal_size + local_range - 1) / local_range;
    const int local_offset = local_id * local_handle_size;
    const int remained_size = vocal_size - local_offset;
    const int handle_size = sycl::min(local_handle_size, remained_size);

    int64_t* random_sampled_ptr = random_sampled + batch_id;
    float* logits_ptr = logits + batch_id * vocal_size + local_offset;
    float* logits_to_return_ptr = logits_to_return + batch_id * vocal_size + local_offset;

    double low = INFINITY, high = -INFINITY;
    double pivot = -INFINITY;
    int pivot_count = top_k_value;
    double eps = 1e-9;

    float local_data[VEC_SIZE];
    const int loop_count = (handle_size + VEC_SIZE - 1) / VEC_SIZE;
    const int remained_vec_size = handle_size - (loop_count - 1) * VEC_SIZE;
    const int loop_times = (remained_vec_size == VEC_SIZE) ? loop_count : (loop_count - 1);
    const bool has_last_loop = (remained_vec_size == VEC_SIZE) ? false : true;
    
    if(top_k_value != vocal_size){

        // low, high
        for(int l = 0; l < loop_times; ++l){
            #pragma unroll
            for(int e = 0; e < VEC_SIZE; ++e){
                local_data[e] = logits_ptr[l * VEC_SIZE + e];
            }

            #pragma unroll
            for(int e = 0; e < VEC_SIZE; ++e){
                float logit = local_data[e] ;
                
                if(logit < low){
                    low = logit;
                }

                if(logit > high){
                    high = logit;
                }
            }
        }

        if(has_last_loop){
            #pragma unroll
            for(int e = 0; e < remained_vec_size; ++e){
                local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
            }

            #pragma unroll
            for(int e = 0; e < remained_vec_size; ++e){
                float logit = local_data[e] ;
                
                if(logit < low){
                    low = logit;
                }

                if(logit > high){
                    high = logit;
                }
            }
        }

        low = sycl::reduce_over_group(group, low, sycl::minimum<>());
        high = sycl::reduce_over_group(group, high, sycl::maximum<>());

        int iter = 0;

        do{
            int pivot_count_local = 0;

            pivot = (low + high) / 2;

            for(int l = 0; l < loop_times; ++l){
                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    local_data[e] = logits_ptr[l * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    float logit = local_data[e] ;
                    
                    if(logit >= pivot){
                        pivot_count_local += 1;
                    }
                }
            }

            if(has_last_loop){
                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    float logit = local_data[e] ;
                    
                    if(logit >= pivot){
                        pivot_count_local += 1;
                    }
                }
            }

            pivot_count = sycl::reduce_over_group(group, pivot_count_local, sycl::plus<>());

            if(pivot_count == top_k_value){
                break;
            } else if(pivot_count < top_k_value){
                high = pivot;
            } else {
                low = pivot;
            }
        } while (((high - low) > eps));

        if(pivot_count < top_k_value){
            pivot = low;
        }
    }

    float pivot_sum = 1.0f;

    if constexpr (logprobs_mode == LogprobsMode::processed_logprobs){
        if(top_k_value != vocal_size){
            float pivot_sum_local = 0.0f;
            pivot_sum = 0.0f;
            for(int l = 0; l < loop_times; ++l){
                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    local_data[e] = logits_ptr[l * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    float logit = local_data[e];
                    
                    if(logit >= pivot){
                        pivot_sum_local += logit;
                    }
                }
            }

            if(has_last_loop){
                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    float logit = local_data[e];
                    
                    if(logit >= pivot){
                        pivot_sum_local += logit;
                    }
                }
            }
            pivot_sum = sycl::reduce_over_group(group, pivot_sum_local, sycl::plus<>());
        }
    }

    float max_value_local = -INFINITY;
    int max_idx_local = 0;

    for(int l = 0; l < loop_times; ++l){
        #pragma unroll
        for(int e = 0; e < VEC_SIZE; ++e){
            local_data[e] = logits_ptr[l * VEC_SIZE + e];
        }

        auto rand4 = dist_func(&state);

        #pragma unroll
        for(int e = 0; e < VEC_SIZE; ++e){
            float logit = local_data[e];
            float rand = exponential_func(static_cast<accscalar_t>((&rand4.x)[e]));
            
            if(logit >= pivot){
                if constexpr (logprobs_mode == LogprobsMode::processed_logprobs){
                    logits_to_return_ptr[l * VEC_SIZE + e] = sycl::log(logit / pivot_sum);
                }
                logit /= rand;
                if(logit > max_value_local){
                    max_value_local = logit;
                    max_idx_local = local_offset + l * VEC_SIZE + e;
                }
            } else {
                if constexpr (logprobs_mode == LogprobsMode::processed_logits || logprobs_mode == LogprobsMode::processed_logprobs){
                    logits_to_return_ptr[l * VEC_SIZE + e] = -INFINITY;
                }
            }
        }
    }

    if(has_last_loop){
        #pragma unroll
        for(int e = 0; e < remained_vec_size; ++e){
            local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
        }

        auto rand4 = dist_func(&state);

        #pragma unroll
        for(int e = 0; e < remained_vec_size; ++e){
            float logit = local_data[e];
            float rand = exponential_func(static_cast<accscalar_t>((&rand4.x)[e]));
            
            if(logit >= pivot){
                if constexpr (logprobs_mode == LogprobsMode::processed_logprobs){
                    logits_to_return_ptr[loop_times * VEC_SIZE + e] = sycl::log(logit / pivot_sum);
                }
                logit /= rand;
                if(logit > max_value_local){
                    max_value_local = logit;
                    max_idx_local = local_offset + loop_times * VEC_SIZE + e;
                }
            } else {
                if constexpr (logprobs_mode == LogprobsMode::processed_logits || logprobs_mode == LogprobsMode::processed_logprobs){
                    logits_to_return_ptr[loop_times * VEC_SIZE + e] = -INFINITY;
                }
            }
        }
    }


    float max_val_global =
        sycl::reduce_over_group(group, max_value_local, sycl::maximum<>());
    bool is_max = (max_val_global == max_value_local);
    int64_t first_max_id = sycl::reduce_over_group(
        group, is_max ? max_idx_local : (vocal_size - 1), sycl::minimum<>());

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
    const int vocal_size;
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
  using accscalar_t = float;

  top_p_only_kernel(
      int64_t* random_sampled,
      float* logits_to_return,
      float* logits,
      const float* top_p,
      const int batch_size,
      const int vocal_size,
      const int64_t seed,
      const int64_t offset,
      const float lambda)
      : random_sampled(random_sampled),
        logits_to_return(logits_to_return),
        logits(logits),
        top_p(top_p),
        batch_size(batch_size),
        vocal_size(vocal_size),
        seed(seed),
        offset(offset),
        lambda(lambda)
        {}

  static inline sycl::nd_range<1>
  get_nd_range(const int batch_size, const int vocal_size) {
    int local_size = group_size;
    if(vocal_size < group_size){
        local_size = (vocal_size + sub_group_size - 1) / sub_group_size * sub_group_size;
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
    randStatePhilox4_32_10_t state;
    rand_init(philox_seed, global_id, philox_offset, &state);

    Uniform4DistributionFunctor dist_func;
    ExponentialFunctor<scalar_t, accscalar_t> exponential_func(lambda);

    auto group = item.get_group();

    const float top_p_value = top_p[batch_id];

    const int local_handle_size = (vocal_size + local_range - 1) / local_range;
    const int local_offset = local_id * local_handle_size;
    const int remained_size = vocal_size - local_offset;
    const int handle_size = sycl::min(local_handle_size, remained_size);

    int64_t* random_sampled_ptr = random_sampled + batch_id;
    float* logits_ptr = logits + batch_id * vocal_size + local_offset;
    float* logits_to_return_ptr = logits_to_return + batch_id * vocal_size + local_offset;

    double low = 0.0f, high = 1.f;
    double pivot = low;
    float pivot_count = 1.f;
    double eps = 1e-9;

    float local_data[VEC_SIZE];
    const int loop_count = (handle_size + VEC_SIZE - 1) / VEC_SIZE;
    const int remained_vec_size = handle_size - (loop_count - 1) * VEC_SIZE;
    const int loop_times = (remained_vec_size == VEC_SIZE) ? loop_count : (loop_count - 1);
    const bool has_last_loop = (remained_vec_size == VEC_SIZE) ? false : true;
    
    if(top_p_value != 1.0f){
        do{
            float pivot_count_local = 0.0f;

            for(int l = 0; l < loop_times; ++l){
                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    local_data[e] = logits_ptr[l * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    float logit = local_data[e] ;
                    
                    if(logit >= pivot){
                        pivot_count_local += logit;
                    }
                }
            }

            if(has_last_loop){
                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    float logit = local_data[e] ;
                    
                    if(logit >= pivot){
                        pivot_count_local += logit;
                    }
                }
            }

            pivot_count = sycl::reduce_over_group(group, pivot_count_local, sycl::plus<>());

            if(pivot_count < top_p_value){
                high = pivot;
            } else {
                low = pivot;
            }

            pivot = (low + high) / 2;
        } while (((high - low) > eps));

        pivot = low;
    }

    float pivot_sum = 1.0f;

    if constexpr (logprobs_mode == LogprobsMode::processed_logprobs){
        if(top_p_value != vocal_size){
            float pivot_sum_local = 0.0f;
            pivot_sum = 0.0f;
            for(int l = 0; l < loop_times; ++l){
                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    local_data[e] = logits_ptr[l * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    float logit = local_data[e];
                    
                    if(logit >= pivot){
                        pivot_sum_local += logit;
                    }
                }
            }

            if(has_last_loop){
                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    float logit = local_data[e];
                    
                    if(logit >= pivot){
                        pivot_sum_local += logit;
                    }
                }
            }
            pivot_sum = sycl::reduce_over_group(group, pivot_sum_local, sycl::plus<>());
        }
    }

    float max_value_local = -INFINITY;
    int max_idx_local = 0;

    for(int l = 0; l < loop_times; ++l){
        #pragma unroll
        for(int e = 0; e < VEC_SIZE; ++e){
            local_data[e] = logits_ptr[l * VEC_SIZE + e];
        }

        auto rand4 = dist_func(&state);

        #pragma unroll
        for(int e = 0; e < VEC_SIZE; ++e){
            float logit = local_data[e];
            float rand = exponential_func(static_cast<accscalar_t>((&rand4.x)[e]));
            
            if(logit >= pivot){
                if constexpr (logprobs_mode == LogprobsMode::processed_logprobs){
                    logits_to_return_ptr[l * VEC_SIZE + e] = sycl::log(logit / pivot_sum);
                }
                logit /= rand;
                if(logit > max_value_local){
                    max_value_local = logit;
                    max_idx_local = local_offset + l * VEC_SIZE + e;
                }
            } else {
                if constexpr (logprobs_mode == LogprobsMode::processed_logits || logprobs_mode == LogprobsMode::processed_logprobs){
                    logits_to_return_ptr[l * VEC_SIZE + e] = -INFINITY;
                }
            }
        }
    }

    if(has_last_loop){
        #pragma unroll
        for(int e = 0; e < remained_vec_size; ++e){
            local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
        }

        auto rand4 = dist_func(&state);

        #pragma unroll
        for(int e = 0; e < remained_vec_size; ++e){
            float logit = local_data[e];
            float rand = exponential_func(static_cast<accscalar_t>((&rand4.x)[e]));
            
            if(logit >= pivot){
                if constexpr (logprobs_mode == LogprobsMode::processed_logprobs){
                    logits_to_return_ptr[loop_times * VEC_SIZE + e] = sycl::log(logit / pivot_sum);
                }
                logit /= rand;
                if(logit > max_value_local){
                    max_value_local = logit;
                    max_idx_local = local_offset + loop_times * VEC_SIZE + e;
                }
            } else {
                if constexpr (logprobs_mode == LogprobsMode::processed_logits || logprobs_mode == LogprobsMode::processed_logprobs){
                    logits_to_return_ptr[loop_times * VEC_SIZE + e] = -INFINITY;
                }
            }
        }
    }


    float max_val_global =
        sycl::reduce_over_group(group, max_value_local, sycl::maximum<>());
    bool is_max = (max_val_global == max_value_local);
    int64_t first_max_id = sycl::reduce_over_group(
        group, is_max ? max_idx_local : (vocal_size - 1), sycl::minimum<>());

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
    const int vocal_size;
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
  using accscalar_t = float;

  top_k_top_p_kernel(
      int64_t* random_sampled,
      float* logits_to_return,
      float* logits,
      const int64_t* top_k,
      const float* top_p,
      const int batch_size,
      const int vocal_size,
      const int64_t seed,
      const int64_t offset,
      const float lambda)
      : random_sampled(random_sampled),
        logits_to_return(logits_to_return),
        logits(logits),
        top_k(top_k),
        top_p(top_p),
        batch_size(batch_size),
        vocal_size(vocal_size),
        seed(seed),
        offset(offset),
        lambda(lambda)
        {}

  static inline sycl::nd_range<1>
  get_nd_range(const int batch_size, const int vocal_size) {
    int local_size = group_size;
    if(vocal_size < group_size){
        local_size = (vocal_size + sub_group_size - 1) / sub_group_size * sub_group_size;
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
    randStatePhilox4_32_10_t state;
    rand_init(philox_seed, global_id, philox_offset, &state);

    Uniform4DistributionFunctor dist_func;
    ExponentialFunctor<scalar_t, accscalar_t> exponential_func(lambda);

    auto group = item.get_group();

    const int top_k_value = top_k[batch_id];
    float top_p_value = top_p[batch_id];

    const int local_handle_size = (vocal_size + local_range - 1) / local_range;
    const int local_offset = local_id * local_handle_size;
    const int remained_size = vocal_size - local_offset;
    const int handle_size = sycl::min(local_handle_size, remained_size);

    int64_t* random_sampled_ptr = random_sampled + batch_id;
    float* logits_ptr = logits + batch_id * vocal_size + local_offset;
    float* logits_to_return_ptr = logits_to_return + batch_id * vocal_size + local_offset;

    double low = 0.0f, high = 1.f / static_cast<double>(top_k_value);
    double pivot = low;
    double eps = 1e-9;

    float local_data[VEC_SIZE];
    const int loop_count = (handle_size + VEC_SIZE - 1) / VEC_SIZE;
    const int remained_vec_size = handle_size - (loop_count - 1) * VEC_SIZE;
    const int loop_times = (remained_vec_size == VEC_SIZE) ? loop_count : (loop_count - 1);
    const bool has_last_loop = (remained_vec_size == VEC_SIZE) ? false : true;
    
    double last_pivot_k_count = 1.0f;
    if(top_k_value != vocal_size){
        int pivot_count = top_k_value;
        do{
            int pivot_count_local = 0;

            for(int l = 0; l < loop_times; ++l){
                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    local_data[e] = logits_ptr[l * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    float logit = local_data[e] ;
                    
                    if(logit >= pivot){
                        pivot_count_local += 1;
                    }
                }
            }

            if(has_last_loop){
                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    float logit = local_data[e] ;
                    
                    if(logit >= pivot){
                        pivot_count_local += 1;
                    }
                }
            }

            pivot_count = sycl::reduce_over_group(group, pivot_count_local, sycl::plus<>());

            if(pivot_count == top_k_value){
                break;
            } else if(pivot_count < top_k_value){
                high = pivot;
            } else {
                low = pivot;
            }

            pivot = (low + high) / 2;
        } while (((high - low) > eps));

        if(pivot_count != top_k_value){
            if(pivot_count > top_k_value){
                pivot = high;
            } else {
                pivot = low;
            }
        }

        double last_pivot_count_local = 0.0f;

        for(int l = 0; l < loop_times; ++l){
            #pragma unroll
            for(int e = 0; e < VEC_SIZE; ++e){
                local_data[e] = logits_ptr[l * VEC_SIZE + e];
            }

            #pragma unroll
            for(int e = 0; e < VEC_SIZE; ++e){
                float logit = local_data[e] ;
                
                if(logit >= pivot){
                    last_pivot_count_local += logit;
                }
            }
        }

        if(has_last_loop){
            #pragma unroll
            for(int e = 0; e < remained_vec_size; ++e){
                local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
            }

            #pragma unroll
            for(int e = 0; e < remained_vec_size; ++e){
                float logit = local_data[e] ;
                
                if(logit >= pivot){
                    last_pivot_count_local += logit;
                }
            }
        }

        last_pivot_k_count = sycl::reduce_over_group(group, last_pivot_count_local, sycl::plus<>());
    }

    if(top_p_value != 1.0f){
        low = pivot;
        high = last_pivot_k_count;
        top_p_value *= last_pivot_k_count;
        double pivot_count = top_p_value;
        do{
            double pivot_count_local = 0.0f;

            for(int l = 0; l < loop_times; ++l){
                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    local_data[e] = logits_ptr[l * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    float logit = local_data[e] ;
                    
                    if(logit >= pivot){
                        pivot_count_local += logit;
                    }
                }
            }

            if(has_last_loop){
                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    float logit = local_data[e] ;
                    
                    if(logit >= pivot){
                        pivot_count_local += logit;
                    }
                }
            }

            pivot_count = sycl::reduce_over_group(group, pivot_count_local, sycl::plus<>());

            if(pivot_count < top_p_value){
                high = pivot;
            } else {
                low = pivot;
            }
            pivot = (low + high) / 2;
        } while (((high - low) > eps));

        pivot = low;
    }

    float pivot_sum = 1.0f;

    if constexpr (logprobs_mode == LogprobsMode::processed_logprobs){
        if(top_p_value != vocal_size || top_k_value != vocal_size){
            float pivot_sum_local = 0.0f;
            pivot_sum = 0.0f;
            for(int l = 0; l < loop_times; ++l){
                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    local_data[e] = logits_ptr[l * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < VEC_SIZE; ++e){
                    float logit = local_data[e];
                    
                    if(logit >= pivot){
                        pivot_sum_local += logit;
                    }
                }
            }

            if(has_last_loop){
                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
                }

                #pragma unroll
                for(int e = 0; e < remained_vec_size; ++e){
                    float logit = local_data[e];
                    
                    if(logit >= pivot){
                        pivot_sum_local += logit;
                    }
                }
            }
            pivot_sum = sycl::reduce_over_group(group, pivot_sum_local, sycl::plus<>());
        }
    }

    float max_value_local = -INFINITY;
    int max_idx_local = 0;

    for(int l = 0; l < loop_times; ++l){
        #pragma unroll
        for(int e = 0; e < VEC_SIZE; ++e){
            local_data[e] = logits_ptr[l * VEC_SIZE + e];
        }

        auto rand4 = dist_func(&state);

        #pragma unroll
        for(int e = 0; e < VEC_SIZE; ++e){
            float logit = local_data[e];
            float rand = exponential_func(static_cast<accscalar_t>((&rand4.x)[e]));
            
            if(logit >= pivot){
                if constexpr (logprobs_mode == LogprobsMode::processed_logprobs){
                    logits_to_return_ptr[l * VEC_SIZE + e] = sycl::log(logit / pivot_sum);
                }
                logit /= rand;
                if(logit > max_value_local){
                    max_value_local = logit;
                    max_idx_local = local_offset + l * VEC_SIZE + e;
                }
            } else {
                if constexpr (logprobs_mode == LogprobsMode::processed_logits || logprobs_mode == LogprobsMode::processed_logprobs){
                    logits_to_return_ptr[l * VEC_SIZE + e] = -INFINITY;
                }
            }
        }
    }

    if(has_last_loop){
        #pragma unroll
        for(int e = 0; e < remained_vec_size; ++e){
            local_data[e] = logits_ptr[loop_times * VEC_SIZE + e];
        }

        auto rand4 = dist_func(&state);

        #pragma unroll
        for(int e = 0; e < remained_vec_size; ++e){
            float logit = local_data[e];
            float rand = exponential_func(static_cast<accscalar_t>((&rand4.x)[e]));
            
            if(logit >= pivot){
                if constexpr (logprobs_mode == LogprobsMode::processed_logprobs){
                    logits_to_return_ptr[loop_times * VEC_SIZE + e] = sycl::log(logit / pivot_sum);
                }
                logit /= rand;
                if(logit > max_value_local){
                    max_value_local = logit;
                    max_idx_local = local_offset + loop_times * VEC_SIZE + e;
                }
            } else {
                if constexpr (logprobs_mode == LogprobsMode::processed_logits || logprobs_mode == LogprobsMode::processed_logprobs){
                    logits_to_return_ptr[loop_times * VEC_SIZE + e] = -INFINITY;
                }
            }
        }
    }


    float max_val_global =
        sycl::reduce_over_group(group, max_value_local, sycl::maximum<>());
    bool is_max = (max_val_global == max_value_local);
    int64_t first_max_id = sycl::reduce_over_group(
        group, is_max ? max_idx_local : (vocal_size - 1), sycl::minimum<>());

    if (0 == local_id) {
      random_sampled_ptr[0] = first_max_id;
    }

  }
 private:
    int64_t* random_sampled;
    float* logits_to_return;
    float* logits;
    const int64_t* top_k;
    const float* top_p;
    const int batch_size;
    const int vocal_size;
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
    const int64_t* top_k,
    const float* top_p,
    const int batch_size,
    const int vocal_size,
    const int64_t seed,
    const int64_t offset,
    const float lambda) {

    if(top_k != nullptr && top_p == nullptr){
        // launch top_k_only_kernel
        using KERNEL_TOPK_ONLY = top_k_only_kernel<logprobs_mode>;
        auto range = KERNEL_TOPK_ONLY::get_nd_range(batch_size, vocal_size);
        queue.submit([&](sycl::handler& cgh) {
            KERNEL_TOPK_ONLY task(
                random_sampled,
                logits_to_return,
                logits,
                top_k,
                batch_size,
                vocal_size,
                seed,
                offset,
                lambda);
            cgh.parallel_for(range, task);
        });
    } else if(top_k == nullptr && top_p != nullptr){
        // launch top_p_only_kernel
        using KERNEL_TOPP_ONLY = top_p_only_kernel<logprobs_mode>;
        auto range = KERNEL_TOPP_ONLY::get_nd_range(batch_size, vocal_size);
        queue.submit([&](sycl::handler& cgh) {
            KERNEL_TOPP_ONLY task(
                random_sampled,
                logits_to_return,
                logits,
                top_p,
                batch_size,
                vocal_size,
                seed,
                offset,
                lambda);
            cgh.parallel_for(range, task);
        });
    } else if(top_k != nullptr && top_p != nullptr){
        // launch top_k_top_p_kernel
        using KERNEL_TOPK_TOPP = top_k_top_p_kernel<logprobs_mode>;
        auto range = KERNEL_TOPK_TOPP::get_nd_range(batch_size, vocal_size);
        queue.submit([&](sycl::handler& cgh) {
            KERNEL_TOPK_TOPP task(
                random_sampled,
                logits_to_return,
                logits,
                top_k,
                top_p,
                batch_size,
                vocal_size,
                seed,
                offset,
                lambda);
            cgh.parallel_for(range, task);
        });
    } else {
        // launch random_sampler_only_kernel
        using KERNEL_SAMPLER_ONLY = random_sampler_only_kernel<logprobs_mode>;
        auto range = KERNEL_SAMPLER_ONLY::get_nd_range(batch_size, vocal_size);
        queue.submit([&](sycl::handler& cgh) {
            KERNEL_SAMPLER_ONLY task(
                random_sampled,
                logits_to_return,
                logits,
                batch_size,
                vocal_size,
                seed,
                offset,
                lambda);
            cgh.parallel_for(range, task);
        });
    }
}

template <typename T>
struct exponential_2d_kernel {
 public:
  static constexpr int sub_group_size = 16;
  static constexpr int group_size = 512;
  static constexpr int VEC_SIZE = 4; // align with rand_uniform4

  using scalar_t = float;
  using accscalar_t = float;

  exponential_2d_kernel(
      T* tensor_ptr,
      const int batch_size,
      const int vocal_size,
      const int64_t seed,
      const int64_t offset,
      const float lambda)
      : tensor_ptr(tensor_ptr),
        batch_size(batch_size),
        vocal_size(vocal_size),
        seed(seed),
        offset(offset),
        lambda(lambda)
        {}

  static inline sycl::nd_range<1>
  get_nd_range(const int batch_size, const int vocal_size) {
    int local_size = group_size;
    if(vocal_size < group_size){
        local_size = (vocal_size + sub_group_size - 1) / sub_group_size * sub_group_size;
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

    auto global_id = item.get_global_linear_id();
    uint64_t philox_seed = seed;
    uint64_t philox_offset = offset;
    randStatePhilox4_32_10_t state;
    rand_init(philox_seed, global_id, philox_offset, &state);

    Uniform4DistributionFunctor dist_func;
    ExponentialFunctor<scalar_t, accscalar_t> exponential_func(lambda);

    const int local_handle_size = (vocal_size + local_range - 1) / local_range;
    const int local_offset = local_id * local_handle_size;
    const int remained_size = vocal_size - local_offset;
    const int handle_size = sycl::min(local_handle_size, remained_size);

    T* tensor_local_ptr = tensor_ptr + batch_id * vocal_size + local_offset;

    const int loop_count = (handle_size + VEC_SIZE - 1) / VEC_SIZE;
    const int remained_vec_size = handle_size - (loop_count - 1) * VEC_SIZE;
    const int loop_times = (remained_vec_size == VEC_SIZE) ? loop_count : (loop_count - 1);
    const bool has_last_loop = (remained_vec_size == VEC_SIZE) ? false : true;

    for(int l = 0; l < loop_times; ++l){
        auto rand4 = dist_func(&state);
        #pragma unroll
        for(int e = 0; e < VEC_SIZE; ++e){
            auto rand = exponential_func(static_cast<accscalar_t>((&rand4.x)[e]));
            tensor_local_ptr[l * VEC_SIZE + e] = static_cast<T>(rand);
        }
    }

    if(has_last_loop){
        auto rand4 = dist_func(&state);
        #pragma unroll
        for(int e = 0; e < remained_vec_size; ++e){
            auto rand = exponential_func(static_cast<accscalar_t>((&rand4.x)[e]));
            tensor_local_ptr[loop_times * VEC_SIZE + e] = static_cast<T>(rand);
        }
    }
  }

 private:
    T* tensor_ptr;
    const int batch_size;
    const int vocal_size;
    const int64_t seed;
    const int64_t offset;
    const float lambda;
};

template <typename T>
void exponential_2d_kernel_launcher(
    sycl::queue& queue,
    T* tensor_ptr,
    const int batch_size,
    const int vocal_size,
    const int64_t seed,
    const int64_t offset,
    const float lambda) {

    using KERNEL = exponential_2d_kernel<T>;
    auto range = KERNEL::get_nd_range(batch_size, vocal_size);
    queue.submit([&](sycl::handler& cgh) {
        KERNEL task(
            tensor_ptr,
            batch_size,
            vocal_size,
            seed,
            offset,
            lambda);
        cgh.parallel_for(range, task);
    });

}
}
