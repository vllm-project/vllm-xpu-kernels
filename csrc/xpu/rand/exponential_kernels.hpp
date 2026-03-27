#include <sycl/sycl.hpp>
#include "heads/DistributionTemplates.h"

namespace RAND {

template <typename T>
struct exponential_2d_kernel {
 public:
  static constexpr int sub_group_size = 16;
  static constexpr int group_size = 512;
  static constexpr int VEC_SIZE = 4;  // align with rand_uniform4

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
        lambda(lambda) {}

  static inline sycl::nd_range<1>
  get_nd_range(const int batch_size, const int vocal_size) {
    int local_size = group_size;
    if (vocal_size < group_size) {
      local_size =
          (vocal_size + sub_group_size - 1) / sub_group_size * sub_group_size;
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
    const int loop_times =
        (remained_vec_size == VEC_SIZE) ? loop_count : (loop_count - 1);
    const bool has_last_loop = (remained_vec_size == VEC_SIZE) ? false : true;

    for (int l = 0; l < loop_times; ++l) {
      auto rand4 = dist_func(&state);
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        auto rand = exponential_func(static_cast<accscalar_t>((&rand4.x)[e]));
        tensor_local_ptr[l * VEC_SIZE + e] = static_cast<T>(rand);
      }
    }

    if (has_last_loop) {
      auto rand4 = dist_func(&state);
#pragma unroll
      for (int e = 0; e < remained_vec_size; ++e) {
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
    KERNEL task(tensor_ptr, batch_size, vocal_size, seed, offset, lambda);
    cgh.parallel_for(range, task);
  });
}
}  // namespace RAND
