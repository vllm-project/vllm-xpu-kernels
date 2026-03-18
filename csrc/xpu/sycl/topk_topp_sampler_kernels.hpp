#include <sycl/sycl.hpp>
#include <oneapi/dpl/random>

namespace TopkToppSamplerImpl {

enum class LogprobsMode{
    raw_logits,
    raw_logprobs,
    processed_logits,
    processed_logprobs
};

struct top_k_only_kernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;

  top_k_only_kernel(
      int64_t* random_sampled,
      float* processed_logprobs,
      const float* logits,
      const int64_t* top_k,
      const int batch_size,
      const int vocal_size,
      const LogprobsMode logprobs_mode)
      : random_sampled(random_sampled),
        processed_logprobs(processed_logprobs),
        logits(logits),
        top_k(top_k),
        batch_size(batch_size),
        vocal_size(vocal_size),
        logprobs_mode(logprobs_mode)
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
    int64_t* random_sampled_ptr = random_sampled + batch_id;
    float* processed_logprobs_ptr = processed_logprobs + batch_id * vocal_size;
    const float* logits_ptr = logits + batch_id * vocal_size;

    const int local_handle_size = (vocal_size + local_range - 1) / local_range;
    const int local_offset = local_id * local_handle_size;
    const int remained_size = vocal_size - local_offset;
    const int handle_size = sycl::min(local_handle_size, remained_size);

    double low = 1.f / static_cast<double>(vocal_size), high = 1.f / static_cast<double>(top_k_value);
    double pivot = low;
    double eps = 1e-9;
    
    if(top_k_value != vocal_size){
        do{
            int pivot_count_local = 0;

            for(int e = 0; e < handle_size; ++e){
                int idx = local_offset + e;
                float logit = logits_ptr[idx];
                
                if(logit >= pivot){
                    pivot_count_local += 1;
                }
            }

            int pivot_count = sycl::reduce_over_group(group, pivot_count_local, sycl::plus<>());

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

    oneapi::dpl::minstd_rand engine(0);
    oneapi::dpl::exponential_distribution<float> exp_dist(1.0f);
    engine.discard(local_offset);

    float max_value_local = INFINITY * -1;
    int max_idx_local = 0;

    for(int e = 0; e < handle_size; ++e){
        int idx = local_offset + e;
        float logit = logits_ptr[idx];
        float rnd_value = exp_dist(engine);
        
        if(logit > pivot){
            // logit /= rnd_value;
            if(logit > max_value_local){
                max_value_local = logit;
                max_idx_local = idx;
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
    float* processed_logprobs;
    const float* logits;
    const int64_t* top_k;
    const int batch_size;
    const int vocal_size;
    const LogprobsMode logprobs_mode;
};


// struct random_sample_only_kernel {
//  public:
//   static constexpr int sub_group_size = 32;
//   static constexpr int group_size = 256;

//   random_sample_only_kernel(
//       int64_t* random_sampled,
//       float* processed_logprobs,
//       const float* logits,
//       const int batch_size,
//       const int vocal_size,
//       const LogprobsMode logprobs_mode)
//       : random_sampled(random_sampled),
//         processed_logprobs(processed_logprobs),
//         logits(logits),
//         batch_size(batch_size),
//         vocal_size(vocal_size),
//         logprobs_mode(logprobs_mode)
//         {}

//   static inline sycl::nd_range<1>
//   get_nd_range(const int batch_size, const int vocal_size) {
//     int local_size = group_size;
//     if(vocal_size < group_size){
//         local_size = (vocal_size + sub_group_size - 1) / sub_group_size * sub_group_size;
//     }
//     sycl::range<1> local(local_size);
//     sycl::range<1> global(batch_size);
//     return sycl::nd_range<1>(global * local, local);
//   }

//   [[sycl::reqd_sub_group_size(sub_group_size)]] void
//   operator()(sycl::nd_item<1> item) const {
//     const int batch_id = item.get_group(0);
//     const int local_id = item.get_local_linear_id();
//     const int local_range = item.get_local_range(0);

//     auto group = item.get_group();

//     const int64_t* random_sampled_ptr = random_sampled + batch_id;
//     const float* processed_logprobs_ptr = processed_logprobs + batch_id * vocal_size;

//     const float* logits_ptr = logits + batch_id * vocal_size;

//     const int local_handle_size = (vocal_size + local_range - 1) / local_range;
//     const int local_offset = local_id * local_handle_size;
//     const int remained_size = vocal_size - local_offset;
//     const int handle_size = sycl::min(local_handle_size, remained_size);

//     int max_value_local = -inf;
//     int max_idx = 0;

//     oneapi::dpl::minstd_rand engine(0);
//     oneapi::dpl::exponential_distribution<float> exp_dist(1.0f);
//     engine.discard(local_offset);

//     for(int e = 0; e < handle_size; ++e){
//         int idx = local_offset + e;
//         float logit = logits_ptr[idx];
//         float rnd_value = exp_dist(engine);

//         logit /= rnd_value;

//         if(logit > max_value_local){
//             max_value_local = logit;
//             max_idx = idx;
//         }
//     }

//     float max_val_global =
//         sycl::reduce_over_group(group, max_value_local, sycl::maximum<>());
//     bool is_max = (max_val_global == max_value_local);
//     int64_t first_max_local_id = sycl::reduce_over_group(
//         group, is_max ? max_score_local_id : INFINITY, sycl::minimum<>());

//     if(logprobs_mode == LogprobsMode::processed_logits){
//         for(int e = 0; e < handle_size; ++e){
//             int idx = local_offset + e;
//             float processed_logprobs_ptr[idx] = logits_ptr[idx];
//         }
//     } else if(logprobs_mode == LogprobsMode::processed_logprobs){
//         for(int e = 0; e < handle_size; ++e){
//             int idx = local_offset + e;
//             processed_logprobs_ptr[idx] = logits_ptr[idx];
//         }
//     }

//     if (0 == local_id) {
//       random_sampled_ptr[0] = first_max_local_id;
//     }

//   }

//  private:
//     int64_t* random_sampled;
//     float* processed_logprobs;
//     const float* logits;
//     const int batch_size;
//     const int vocal_size;
//     const LogprobsMode logprobs_mode;
// };

void kernel_launcher(
    sycl::queue& queue,
    int64_t* random_sampled,
    float* processed_logprobs,
    const float* logits,
    const int64_t* top_k,
    const float* top_p,
    const int batch_size,
    const int vocal_size,
    const LogprobsMode logprobs_mode){

    if(top_k != nullptr && top_p == nullptr){
        // launch top_k_only_kernel
        std::cout << "Launch top_k_only_kernel" << std::endl;
        using KERNEL_TOPK_ONLY = top_k_only_kernel;
        auto range = KERNEL_TOPK_ONLY::get_nd_range(batch_size, vocal_size);
        queue.submit([&](sycl::handler& cgh) {
            KERNEL_TOPK_ONLY task(
                random_sampled,
                processed_logprobs,
                logits,
                top_k,
                batch_size,
                vocal_size,
                logprobs_mode);
            cgh.parallel_for(range, task);
        });
    } else if(top_k == nullptr && top_p != nullptr){
        // launch top_p_only_kernel
    } else if(top_k != nullptr && top_p != nullptr){
        // launch top_k_top_p_kernel
    } else {
        // launch random_sample_only_kernel
    }

}
}