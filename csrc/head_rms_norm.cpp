#include <sycl/sycl.hpp>

#include <algorithm>
#include <limits>
#include <type_traits>
#include <ATen/DeviceGuard.h>
#include "utils.h"
#include "dispatch_utils.h"

#if defined(__SYCL_DEVICE_ONLY__)
#define VLLM_REQD_SG_16 [[sycl::reqd_sub_group_size(16)]]
#else
#define VLLM_REQD_SG_16
#endif

namespace vllm {

constexpr int HEAD_RMS_NORM_SIMD = 16;

struct SyclKerConfigBase {};

#define __SYCL_KER_CONFIG_CONVENTION__ SyclKerConfigBase
#define SYCL_REQD_SUB_GROUP_SIZE(SIZE) [[sycl::reqd_sub_group_size(SIZE)]]

template <typename T>
using sycl_local_acc_t = sycl::local_accessor<T, 1>;

template <typename T, int N>
struct alignas(16) aligned_vector {
  T val[N];
};

struct HeadSelection {
  int use_head_selection;
  int64_t total_heads;
  int64_t head_dim;
  int64_t norm_head_start;
  int64_t norm_head_num;
};

inline int64_t head_row_offset(int64_t row_idx, const HeadSelection& selection) {
  if (!selection.use_head_selection) {
    return row_idx * selection.head_dim;
  }

  const int64_t token_idx = row_idx / selection.norm_head_num;
  const int64_t head_idx = row_idx - token_idx * selection.norm_head_num;
  return ((token_idx * selection.total_heads) +
          selection.norm_head_start +
          head_idx) * selection.head_dim;
}

template <typename T>
bool head_can_vectorize(const T* ptr, int alignment) {
  uint64_t addr = reinterpret_cast<uint64_t>(ptr);
  return addr % alignment == 0;
}

template <typename KernelFn>
inline void sycl_kernel_submit(
    const sycl::range<1>& global_range,
    const sycl::range<1>& local_range,
    sycl::queue& queue,
    KernelFn kfn) {
  queue.submit([&](sycl::handler& cgh) {
    kfn.sycl_ker_config_convention(cgh);
    cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
  });
}

template <typename KernelFn>
inline void sycl_kernel_submit(
    const sycl::range<2>& global_range,
    const sycl::range<2>& local_range,
    sycl::queue& queue,
    KernelFn kfn) {
  queue.submit([&](sycl::handler& cgh) {
    kfn.sycl_ker_config_convention(cgh);
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
  });
}

namespace head_rms_norm_impl_detail {

constexpr int granularity = 16;

inline int next_pow2(int val) {
  int result = 1;
  while (result < val) {
    result <<= 1;
  }
  return result;
}

}  // namespace head_rms_norm_impl_detail

template <typename T, int UNROLL, int threadsPerGroup, int maxThreads>
struct HeadRMSNormKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int T_per_load =
      head_rms_norm_impl_detail::granularity / sizeof(T);

  SYCL_REQD_SUB_GROUP_SIZE(HEAD_RMS_NORM_SIMD)
  void operator()(sycl::nd_item<2> item_id) const {
    constexpr int groups_per_block = maxThreads / threadsPerGroup;
    const int row_in_block = item_id.get_local_id(0);
    const int tid = item_id.get_local_id(1);
    const int row_idx = item_id.get_group(1) * groups_per_block + row_in_block;

    if (row_idx >= M_) {
      return;
    }

    const int thread_offset = tid * T_per_load;
    const int stride = threadsPerGroup * T_per_load;

    float var_sum = 0.f;
    T* row_data = X_ + head_row_offset(static_cast<int64_t>(row_idx), selection_);

    T local_buffer[UNROLL * T_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + i * T_per_load;
      const int iter_offset = i * stride + thread_offset;

      if (aligned_mode_) {
        const bool do_loads = (iter_offset < N_);
        if (do_loads) {
          using vec_t = aligned_vector<T, T_per_load>;
          *reinterpret_cast<vec_t*>(iteration_buffer) =
              *reinterpret_cast<const vec_t*>(row_data + iter_offset);
        } else {
#pragma unroll
          for (int j = 0; j < T_per_load; j++) {
            iteration_buffer[j] = T(0);
          }
        }
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
          float up_cast = static_cast<float>(iteration_buffer[j]);
          var_sum += up_cast * up_cast;
        }
      } else {
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
          const int idx = iter_offset + j;
          T v = (idx < N_) ? row_data[idx] : T(0);
          iteration_buffer[j] = v;
          float up_cast = static_cast<float>(v);
          var_sum += up_cast * up_cast;
        }
      }
    }

    auto sg = item_id.get_sub_group();

    if constexpr (threadsPerGroup <= HEAD_RMS_NORM_SIMD) {
      int sg_local_id = sg.get_local_linear_id();
      int local_tid = sg_local_id % threadsPerGroup;

#pragma unroll
      for (int offset = threadsPerGroup / 2; offset > 0; offset >>= 1) {
        float shifted = sycl::shift_group_left(sg, var_sum, offset);
        if (local_tid < offset) {
          var_sum += shifted;
        }
      }

      int partition_in_sg = sg_local_id / threadsPerGroup;
      var_sum = sycl::select_from_group(
          sg, var_sum, partition_in_sg * threadsPerGroup);
    } else {
#pragma unroll
      for (int offset = HEAD_RMS_NORM_SIMD / 2; offset > 0; offset >>= 1) {
        var_sum += sycl::shift_group_left(sg, var_sum, offset);
      }

      constexpr int num_warps = threadsPerGroup / HEAD_RMS_NORM_SIMD;
      int sg_local_id = sg.get_local_linear_id();
      int warp_id = tid / HEAD_RMS_NORM_SIMD;

      if (sg_local_id == 0) {
        shared_[warp_id] = var_sum;
      }
      sycl::group_barrier(item_id.get_group());

      if (warp_id == 0) {
        var_sum = (sg_local_id < num_warps) ? shared_[sg_local_id] : 0.f;
#pragma unroll
        for (int offset = HEAD_RMS_NORM_SIMD / 2; offset > 0; offset >>= 1) {
          var_sum += sycl::shift_group_left(sg, var_sum, offset);
        }
        if (sg_local_id == 0) {
          shared_[0] = var_sum;
        }
      }
      sycl::group_barrier(item_id.get_group());
      var_sum = shared_[0];
    }

    const float var = var_sum / static_cast<float>(N_);
    const float denom = sycl::rsqrt(var + epsilon_);

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + i * T_per_load;
      const int iter_offset = i * stride + thread_offset;

      T gamma_local[T_per_load];
      if (aligned_mode_) {
        if (gamma_ != nullptr && iter_offset < N_) {
          using vec_t = aligned_vector<T, T_per_load>;
          *reinterpret_cast<vec_t*>(gamma_local) =
              *reinterpret_cast<const vec_t*>(gamma_ + iter_offset);
        } else {
#pragma unroll
          for (int j = 0; j < T_per_load; j++) {
            gamma_local[j] = T(0);
          }
        }
      } else {
        if (gamma_ != nullptr) {
#pragma unroll
          for (int j = 0; j < T_per_load; j++) {
            const int idx = iter_offset + j;
            gamma_local[j] = (idx < N_) ? gamma_[idx] : T(0);
          }
        }
      }

#pragma unroll
      for (int j = 0; j < T_per_load; j++) {
        float val = static_cast<float>(iteration_buffer[j]) * denom;
        if (gamma_ != nullptr) {
          val *= static_cast<float>(gamma_local[j]);
        }
        iteration_buffer[j] = static_cast<T>(val);
      }

      if (aligned_mode_) {
        if (iter_offset < N_) {
          using vec_t = aligned_vector<T, T_per_load>;
          *reinterpret_cast<vec_t*>(row_data + iter_offset) =
              *reinterpret_cast<const vec_t*>(iteration_buffer);
        }
      } else {
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
          const int idx = iter_offset + j;
          if (idx < N_) {
            row_data[idx] = iteration_buffer[j];
          }
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    constexpr int shared_size =
        (threadsPerGroup > HEAD_RMS_NORM_SIMD) ? (threadsPerGroup / HEAD_RMS_NORM_SIMD) : 1;
    shared_ = sycl_local_acc_t<float>(shared_size, cgh);
  }

  HeadRMSNormKernelFunctor(
      int N,
      int M,
      float epsilon,
      T* X,
      const T* gamma,
      HeadSelection selection,
      bool aligned_mode)
      : N_(N),
        M_(M),
        epsilon_(epsilon),
        X_(X),
        gamma_(gamma),
        selection_(selection),
        aligned_mode_(aligned_mode) {}

 private:
  int N_;
  int M_;
  float epsilon_;
  T* X_;
  const T* gamma_;
  HeadSelection selection_;
  bool aligned_mode_;
  sycl_local_acc_t<float> shared_;
};

template <typename T, int TPB>
struct HeadRMSNormLargeNKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int T_per_load =
      head_rms_norm_impl_detail::granularity / sizeof(T);
  static constexpr int NWARPS = TPB / HEAD_RMS_NORM_SIMD;

  SYCL_REQD_SUB_GROUP_SIZE(HEAD_RMS_NORM_SIMD)
  void operator()(sycl::nd_item<1> item_id) const {
    const int row = static_cast<int>(item_id.get_group(0));
    if (row >= M_) {
      return;
    }
    const int tid = static_cast<int>(item_id.get_local_id(0));

    T* row_data = X_ + head_row_offset(static_cast<int64_t>(row), selection_);

    float var_sum = 0.f;
    if (can_vec_) {
      using vec_t = aligned_vector<T, T_per_load>;
      const int n_vec = N_ / T_per_load;
      const int tail_start = n_vec * T_per_load;
      const vec_t* X_vec = reinterpret_cast<const vec_t*>(row_data);
      for (int i = tid; i < n_vec; i += TPB) {
        vec_t v = X_vec[i];
#pragma unroll
        for (int j = 0; j < T_per_load; ++j) {
          float f = static_cast<float>(v.val[j]);
          var_sum += f * f;
        }
      }
      for (int i = tail_start + tid; i < N_; i += TPB) {
        float f = static_cast<float>(row_data[i]);
        var_sum += f * f;
      }
    } else {
      for (int i = tid; i < N_; i += TPB) {
        float f = static_cast<float>(row_data[i]);
        var_sum += f * f;
      }
    }

    auto sg = item_id.get_sub_group();
#pragma unroll
    for (int off = HEAD_RMS_NORM_SIMD / 2; off > 0; off >>= 1) {
      var_sum += sycl::shift_group_left(sg, var_sum, off);
    }
    const int sg_lane = static_cast<int>(sg.get_local_linear_id());
    const int wid = tid / HEAD_RMS_NORM_SIMD;
    if (sg_lane == 0) {
      shared_[wid] = var_sum;
    }
    sycl::group_barrier(item_id.get_group());
    if (wid == 0) {
      float v = (sg_lane < NWARPS) ? shared_[sg_lane] : 0.f;
#pragma unroll
      for (int off = HEAD_RMS_NORM_SIMD / 2; off > 0; off >>= 1) {
        v += sycl::shift_group_left(sg, v, off);
      }
      if (sg_lane == 0) {
        shared_[0] = v;
      }
    }
    sycl::group_barrier(item_id.get_group());
    const float total = shared_[0];
    const float denom = sycl::rsqrt(total / static_cast<float>(N_) + epsilon_);

    if (can_vec_) {
      using vec_t = aligned_vector<T, T_per_load>;
      const int n_vec = N_ / T_per_load;
      const int tail_start = n_vec * T_per_load;
      const vec_t* G_vec = (gamma_ != nullptr)
          ? reinterpret_cast<const vec_t*>(gamma_)
          : nullptr;
      vec_t* X_vec = reinterpret_cast<vec_t*>(row_data);
      for (int i = tid; i < n_vec; i += TPB) {
        vec_t v = X_vec[i];
        vec_t g;
        if (G_vec != nullptr) {
          g = G_vec[i];
        }
        vec_t out;
#pragma unroll
        for (int j = 0; j < T_per_load; ++j) {
          float val = static_cast<float>(v.val[j]) * denom;
          if (G_vec != nullptr) {
            val *= static_cast<float>(g.val[j]);
          }
          out.val[j] = static_cast<T>(val);
        }
        X_vec[i] = out;
      }
      for (int i = tail_start + tid; i < N_; i += TPB) {
        float val = static_cast<float>(row_data[i]) * denom;
        if (gamma_ != nullptr) {
          val *= static_cast<float>(gamma_[i]);
        }
        row_data[i] = static_cast<T>(val);
      }
    } else {
      for (int i = tid; i < N_; i += TPB) {
        float val = static_cast<float>(row_data[i]) * denom;
        if (gamma_ != nullptr) {
          val *= static_cast<float>(gamma_[i]);
        }
        row_data[i] = static_cast<T>(val);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<float>(NWARPS, cgh);
  }

  HeadRMSNormLargeNKernelFunctor(
      int N,
      int M,
      float epsilon,
      T* X,
      const T* gamma,
      HeadSelection selection,
      bool can_vec)
      : N_(N),
        M_(M),
        epsilon_(epsilon),
        X_(X),
        gamma_(gamma),
        selection_(selection),
        can_vec_(can_vec) {}

 private:
  int N_;
  int M_;
  float epsilon_;
  T* X_;
  const T* gamma_;
  HeadSelection selection_;
  bool can_vec_;
  sycl_local_acc_t<float> shared_;
};

template <typename T>
void launch_head_rms_norm_large_n_kernel(
    int N,
    int M,
    float eps,
    T* X,
    const T* gamma,
    HeadSelection selection,
    sycl::queue& queue) {
  constexpr int TPB = 256;
  constexpr int T_per_load =
      head_rms_norm_impl_detail::granularity / sizeof(T);
  constexpr int alignment = T_per_load * sizeof(T);

  T* row0_data = X + head_row_offset(0, selection);
  const bool can_vec = (N % T_per_load == 0) && (N >= T_per_load) &&
      head_can_vectorize(row0_data, alignment) &&
      (gamma == nullptr || head_can_vectorize(gamma, alignment));

  using KernelClass = HeadRMSNormLargeNKernelFunctor<T, TPB>;
  KernelClass kfn(N, M, eps, X, gamma, selection, can_vec);
  sycl::range<1> local_range(static_cast<size_t>(TPB));
  sycl::range<1> global_range(static_cast<size_t>(M) * TPB);
  sycl_kernel_submit(global_range, local_range, queue, kfn);
}

#define LAUNCH_HEAD_RMS_NORM_IPEX(UNROLL_VAL, TPG, MAXT)                    \
  do {                                                                       \
    using KernelClass =                                                      \
        HeadRMSNormKernelFunctor<T, UNROLL_VAL, TPG, MAXT>;                 \
    KernelClass kfn(                                                         \
        N_int,                                                               \
        M_int,                                                               \
        eps_f,                                                               \
        X_data,                                                              \
        gamma_data,                                                          \
        selection,                                                           \
        aligned_mode);                                                       \
    sycl::range<2> local_range{                                              \
        static_cast<size_t>(groups_per_block), static_cast<size_t>(TPG)};    \
    sycl::range<2> global_range{                                             \
        static_cast<size_t>(groups_per_block),                               \
        static_cast<size_t>(groups_launch) * static_cast<size_t>(TPG)};      \
    sycl_kernel_submit(global_range, local_range, queue, kfn);               \
  } while (0)

template <typename T, typename T_ACC>
void head_rms_norm_kernel_impl(
    torch::Tensor& X,
    const at::Tensor& gamma,
    int64_t M,
    int64_t N,
    T_ACC eps,
    HeadSelection selection,
    sycl::queue& queue) {
  constexpr int T_per_load =
      head_rms_norm_impl_detail::granularity / sizeof(T);

  T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;

  if (M == 0) {
    return;
  }

  constexpr int kIpexMaxN = 16384;

  if (N > kIpexMaxN) {
    launch_head_rms_norm_large_n_kernel<T>(
        static_cast<int>(N),
        static_cast<int>(M),
        static_cast<float>(eps),
        X_data,
        gamma_data,
        selection,
        queue);
    return;
  }

  const int N_int = static_cast<int>(N);
  const int M_int = static_cast<int>(M);
  const float eps_f = static_cast<float>(eps);

  constexpr int maxThreads = 256;
  constexpr int internalUnroll = sizeof(T) == 4 ? 4 : 2;

  const bool is_subblock_schedule = (N_int <= 128);
  const int h_per_step =
      is_subblock_schedule ? T_per_load : T_per_load * internalUnroll;

  const int one_step_threads =
      head_rms_norm_impl_detail::next_pow2((N_int + h_per_step - 1) / h_per_step);
  const int threads_per_group =
      (one_step_threads < maxThreads) ? one_step_threads : maxThreads;

  const int groups_per_block_max = is_subblock_schedule
      ? (maxThreads + threads_per_group - 1) / threads_per_group
      : 1;
  const int groups_per_block =
      (M_int < groups_per_block_max) ? M_int : groups_per_block_max;
  const int groups_launch = (M_int + groups_per_block - 1) / groups_per_block;

  const int elems_per_step = threads_per_group * h_per_step;
  const int external_unroll = (N_int + elems_per_step - 1) / elems_per_step;

  constexpr int kAlignBytes = head_rms_norm_impl_detail::granularity;
  T* row0_data = X_data + head_row_offset(0, selection);
  const bool aligned_mode = (N_int % T_per_load == 0) &&
      head_can_vectorize(row0_data, kAlignBytes) &&
      (gamma_data == nullptr || head_can_vectorize(gamma_data, kAlignBytes));

  if (is_subblock_schedule) {
    if (threads_per_group == 1) {
      LAUNCH_HEAD_RMS_NORM_IPEX(1, 1, maxThreads);
    } else if (threads_per_group == 2) {
      LAUNCH_HEAD_RMS_NORM_IPEX(1, 2, maxThreads);
    } else if (threads_per_group == 4) {
      LAUNCH_HEAD_RMS_NORM_IPEX(1, 4, maxThreads);
    } else if (threads_per_group == 8) {
      LAUNCH_HEAD_RMS_NORM_IPEX(1, 8, maxThreads);
    } else if (threads_per_group == 16) {
      LAUNCH_HEAD_RMS_NORM_IPEX(1, 16, maxThreads);
    } else if (threads_per_group == 32) {
      LAUNCH_HEAD_RMS_NORM_IPEX(1, 32, maxThreads);
    }
  } else if (external_unroll == 1) {
    LAUNCH_HEAD_RMS_NORM_IPEX(1 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unroll == 2) {
    LAUNCH_HEAD_RMS_NORM_IPEX(2 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unroll == 3) {
    LAUNCH_HEAD_RMS_NORM_IPEX(3 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unroll == 4) {
    LAUNCH_HEAD_RMS_NORM_IPEX(4 * internalUnroll, maxThreads, maxThreads);
  }
}

#undef LAUNCH_HEAD_RMS_NORM_IPEX

void head_rms_norm(
    torch::Tensor& input,
    torch::Tensor& weight,
    int64_t norm_head_start,
    int64_t norm_head_num,
    double epsilon) {
  TORCH_CHECK(input.is_xpu(), "input must be XPU tensor");
  TORCH_CHECK(weight.is_xpu(), "weight must be XPU tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(input.dim() == 3, "input must be 3D [B, H, N]");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D [N]");
  TORCH_CHECK(norm_head_start >= 0, "norm_head_start must be non-negative");
  TORCH_CHECK(norm_head_num >= 0, "norm_head_num must be non-negative");

  const torch::Tensor weight_cast =
      (weight.scalar_type() == input.scalar_type())
          ? weight
          : weight.to(input.scalar_type());

  HeadSelection selection{0, 1, 0, 0, 1};
  int64_t M = 0;
  int64_t N = 0;

  const int64_t total_heads = input.size(1);
  N = input.size(2);
  TORCH_CHECK(
      norm_head_start <= total_heads,
      "norm_head_start must be within total head count");
  const int64_t max_norm_head_num = total_heads - norm_head_start;
  const int64_t effective_norm_head_num =
      (norm_head_num < max_norm_head_num) ? norm_head_num : max_norm_head_num;

  if (effective_norm_head_num == 0) {
    return;
  }

  M = input.size(0) * effective_norm_head_num;
  selection.use_head_selection = 1;
  selection.total_heads = total_heads;
  selection.head_dim = N;
  selection.norm_head_start = norm_head_start;
  selection.norm_head_num = effective_norm_head_num;

  TORCH_CHECK(weight_cast.size(0) == N, "weight size must match head_dim of input");
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "head_rms_norm", [&]() {
    using acc_t = typename std::conditional<
        std::is_same<scalar_t, double>::value,
        double,
        float>::type;
    head_rms_norm_kernel_impl<scalar_t, acc_t>(
        input,
        weight_cast,
        M,
        N,
        static_cast<acc_t>(epsilon),
        selection,
        queue);
  });
}

}  // namespace vllm

void head_rms_norm(
    torch::Tensor& input,
    torch::Tensor& weight,
    int64_t norm_head_start,
    int64_t norm_head_num,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  vllm::head_rms_norm(
      input, weight, norm_head_start, norm_head_num, epsilon);
}
