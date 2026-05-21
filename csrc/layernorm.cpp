#include <sycl/sycl.hpp>

#include <ATen/DeviceGuard.h>
#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {
namespace rms_ipex_detail {

constexpr int granularity = 16;

inline int next_pow2(int val) {
  int rounded_val = val - 1;
  rounded_val |= rounded_val >> 1;
  rounded_val |= rounded_val >> 2;
  rounded_val |= rounded_val >> 4;
  rounded_val |= rounded_val >> 8;
  return rounded_val + 1;
}

namespace conversion {
template <typename TO, typename FROM>
inline TO to(FROM val) {
  return static_cast<TO>(val);
}
}  // namespace conversion

namespace mem_access {
template <int AccessSize>
inline void load_global(void* dst, const void* src, bool do_access);

template <>
inline void load_global<16>(void* dst, const void* src, bool do_access) {
  sycl::uint4* data = reinterpret_cast<sycl::uint4*>(dst);
  const sycl::uint4* src_cast = reinterpret_cast<const sycl::uint4*>(src);
  if (do_access) {
    data[0] = src_cast[0];
  } else {
    data[0].x() = 0;
    data[0].y() = 0;
    data[0].z() = 0;
    data[0].w() = 0;
  }
}

template <int AccessSize>
inline void store_global(void* dst, const void* src);

template <>
inline void store_global<16>(void* dst, const void* src) {
  const sycl::uint4* data = reinterpret_cast<const sycl::uint4*>(src);
  sycl::uint4* dst_cast = reinterpret_cast<sycl::uint4*>(dst);
  dst_cast[0] = data[0];
}
}  // namespace mem_access

namespace reduce {
enum class ROpType {
  Add,
};

template <ROpType Op, typename T>
inline T init();

template <>
inline float init<ROpType::Add, float>() {
  return 0.0f;
}

template <ROpType Op, typename T>
inline T element(const T lhs, const T rhs);

template <>
inline float element<ROpType::Add, float>(const float lhs, const float rhs) {
  return lhs + rhs;
}

template <ROpType Op, int num_threads>
inline void partitioned_block(
    sycl::group<3>& tb,
    sycl::sub_group& warp,
    float& val) {
  if constexpr (num_threads <= 32) {
    const int sg_local_id = static_cast<int>(warp.get_local_linear_id());
    const int local_tid = sg_local_id % num_threads;
#pragma unroll
    for (int offset = num_threads / 2; offset > 0; offset >>= 1) {
      float shifted = sycl::shift_group_left(warp, val, offset);
      if (local_tid < offset) {
        val = element<Op, float>(val, shifted);
      }
    }
    const int partition_in_sg = sg_local_id / num_threads;
    val = sycl::select_from_group(warp, val, partition_in_sg * num_threads);
  } else {
    val = sycl::reduce_over_group(tb, val, sycl::plus<>());
  }
}
}  // namespace reduce

using rop = reduce::ROpType;

template <typename T, int UNROLL, int threadsPerGroup, int maxThreads>
class rms_norm {
 private:
  T* output;
  const T* vals;
  const T* gamma;
  float epsilon;
  int elems_per_row;

 public:
  rms_norm(
      T* output,
      const T* vals,
      const T* gamma,
      float epsilon,
      int elems_per_row)
      : output(output),
        vals(vals),
        gamma(gamma),
        epsilon(epsilon),
        elems_per_row(elems_per_row) {}

  void operator()(sycl::nd_item<3> item_ct1) const {
    constexpr int T_per_load = granularity / sizeof(T);

    sycl::group<3> tb = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();

    const int block_offset = (tb.get_group_id()[2] *
                              (maxThreads / threadsPerGroup) * elems_per_row) +
        (tb.get_local_id()[1] * elems_per_row);
    const int thread_offset = tb.get_local_id()[2] * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = item_ct1.get_local_range(2) * T_per_load;

    float var_sum = reduce::init<rop::Add, float>();

    const T* input_base = vals + base_offset;

    T local_buffer[UNROLL * T_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + (i * T_per_load);

      mem_access::load_global<granularity>(
          iteration_buffer,
          input_base + (i * stride),
          thread_offset + (i * stride) < elems_per_row);

#pragma unroll
      for (int j = 0; j < T_per_load; j++) {
        int iter_idx = thread_offset + (i * stride) + j;
        if (iter_idx < elems_per_row) {
          float up_cast = conversion::to<float>(iteration_buffer[j]);
          float sq_val = up_cast * up_cast;
          var_sum = reduce::element<rop::Add, float>(var_sum, sq_val);
        }
      }
    }

    reduce::partitioned_block<rop::Add, threadsPerGroup>(tb, warp, var_sum);
    const float var = var_sum / elems_per_row;
    const float denom = sycl::rsqrt(var + epsilon);

    T* block_output = output + block_offset;

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + (i * T_per_load);
      const int iter_idx = i * stride + thread_offset;
      const bool do_loads = (iter_idx < elems_per_row);

      T gamma_local[T_per_load];

      mem_access::load_global<granularity>(
          gamma_local, gamma + iter_idx, do_loads);

#pragma unroll
      for (int j = 0; j < T_per_load; j++) {
        float v = conversion::to<float>(iteration_buffer[j]) * denom *
            conversion::to<float>(gamma_local[j]);
        iteration_buffer[j] = conversion::to<T>(v);
      }

      if (do_loads) {
        mem_access::store_global<granularity>(
            block_output + iter_idx, iteration_buffer);
      }
    }
  }
};

template <typename T, int UNROLL, int threadsPerGroup, int maxThreads>
class pre_rms_norm {
 private:
  T* output;
  const T* vals;
  const T* residual;
  const T* gamma;
  float epsilon;
  int elems_per_row;

 public:
  pre_rms_norm(
      T* output,
      const T* vals,
      const T* residual,
      const T* gamma,
      float epsilon,
      int elems_per_row)
      : output(output),
        vals(vals),
        residual(residual),
        gamma(gamma),
        epsilon(epsilon),
        elems_per_row(elems_per_row) {}

  void operator()(sycl::nd_item<3> item_ct1) const {
    constexpr int T_per_load = granularity / sizeof(T);

    sycl::group<3> tb = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();

    const int block_offset = (tb.get_group_id()[2] *
                              (maxThreads / threadsPerGroup) * elems_per_row) +
        (tb.get_local_id()[1] * elems_per_row);
    const int thread_offset = tb.get_local_id()[2] * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = item_ct1.get_local_range(2) * T_per_load;

    float var_sum = reduce::init<rop::Add, float>();

    const T* input_base = vals + base_offset;
    const T* residual_base = residual + base_offset;

    T local_buffer[UNROLL * T_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + (i * T_per_load);
      T residual_buffer[T_per_load];

      const int iter_offset = i * stride + thread_offset;
      const bool do_loads = (iter_offset < elems_per_row);

      mem_access::load_global<granularity>(
          iteration_buffer, input_base + (i * stride), do_loads);
      mem_access::load_global<granularity>(
          residual_buffer, residual_base + (i * stride), do_loads);

#pragma unroll
      for (int j = 0; j < T_per_load; j++) {
        int iter_idx = thread_offset + (i * stride) + j;
        if (iter_idx < elems_per_row) {
          iteration_buffer[j] += residual_buffer[j];
          float vals_up_cast = conversion::to<float>(iteration_buffer[j]);

          var_sum = reduce::element<rop::Add, float>(
              var_sum, vals_up_cast * vals_up_cast);
        }
      }
    }

    reduce::partitioned_block<rop::Add, threadsPerGroup>(tb, warp, var_sum);
    const float var = var_sum / elems_per_row;
    const float denom = sycl::rsqrt(var + epsilon);

    T* block_output = output + block_offset;

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + (i * T_per_load);
      const int iter_idx = i * stride + thread_offset;
      const bool do_loads = (iter_idx < elems_per_row);

      T gamma_local[T_per_load];

      mem_access::load_global<granularity>(
          gamma_local, gamma + iter_idx, do_loads);

#pragma unroll
      for (int j = 0; j < T_per_load; j++) {
        float v = conversion::to<float>(iteration_buffer[j]) * denom *
            conversion::to<float>(gamma_local[j]);
        iteration_buffer[j] = conversion::to<T>(v);
      }

      if (do_loads) {
        mem_access::store_global<granularity>(
            block_output + iter_idx, iteration_buffer);
      }
    }
  }
};

template <typename T, int UNROLL, int threadsPerGroup, int maxThreads>
class rms_norm_big {
 private:
  T* output;
  const T* vals;
  const T* gamma;
  float epsilon;
  int elems_per_row;
  int outer_iters;

 public:
  rms_norm_big(
      T* output,
      const T* vals,
      const T* gamma,
      float epsilon,
      int elems_per_row,
      int outer_iters)
      : output(output),
        vals(vals),
        gamma(gamma),
        epsilon(epsilon),
        elems_per_row(elems_per_row),
        outer_iters(outer_iters) {}

  void operator()(sycl::nd_item<3> item_ct1) const {
    constexpr int T_per_load = granularity / sizeof(T);

    sycl::group<3> tb = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();

    const int block_offset = (tb.get_group_id()[2] *
                              (maxThreads / threadsPerGroup) * elems_per_row) +
        (tb.get_local_id()[1] * elems_per_row);
    const int thread_offset = tb.get_local_id()[2] * T_per_load;
    const int stride = item_ct1.get_local_range(2) * T_per_load;
    const int chunk_stride = UNROLL * stride;

    float var_sum = reduce::init<rop::Add, float>();

    for (int outer = 0; outer < outer_iters; outer++) {
      const int outer_off = outer * chunk_stride;
      T iter_buf[UNROLL * T_per_load];
#pragma unroll
      for (int i = 0; i < UNROLL; i++) {
        T* ib = iter_buf + i * T_per_load;
        const int total_off = thread_offset + outer_off + i * stride;
        mem_access::load_global<granularity>(
            ib, vals + block_offset + outer_off + thread_offset + i * stride,
            total_off < elems_per_row);
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
          if (total_off + j < elems_per_row) {
            float up = conversion::to<float>(ib[j]);
            var_sum = reduce::element<rop::Add, float>(var_sum, up * up);
          }
        }
      }
    }

    reduce::partitioned_block<rop::Add, threadsPerGroup>(tb, warp, var_sum);
    const float denom = sycl::rsqrt(var_sum / elems_per_row + epsilon);

    for (int outer = 0; outer < outer_iters; outer++) {
      const int outer_off = outer * chunk_stride;
      T iter_buf[UNROLL * T_per_load];
      T gamma_buf[UNROLL * T_per_load];
#pragma unroll
      for (int i = 0; i < UNROLL; i++) {
        T* ib = iter_buf + i * T_per_load;
        T* gb = gamma_buf + i * T_per_load;
        const int total_off = thread_offset + outer_off + i * stride;
        const bool do_loads = total_off < elems_per_row;
        mem_access::load_global<granularity>(
            ib, vals + block_offset + outer_off + thread_offset + i * stride,
            do_loads);
        mem_access::load_global<granularity>(
            gb, gamma + outer_off + thread_offset + i * stride, do_loads);
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
          float v = conversion::to<float>(ib[j]) * denom *
              conversion::to<float>(gb[j]);
          ib[j] = conversion::to<T>(v);
        }
        if (do_loads) {
          mem_access::store_global<granularity>(
              output + block_offset + outer_off + thread_offset + i * stride,
              ib);
        }
      }
    }
  }
};

template <typename T, int UNROLL, int threadsPerGroup, int maxThreads>
class pre_rms_norm_big {
 private:
  T* output;
  const T* vals;
  const T* residual;
  const T* gamma;
  float epsilon;
  int elems_per_row;
  int outer_iters;

 public:
  pre_rms_norm_big(
      T* output,
      const T* vals,
      const T* residual,
      const T* gamma,
      float epsilon,
      int elems_per_row,
      int outer_iters)
      : output(output),
        vals(vals),
        residual(residual),
        gamma(gamma),
        epsilon(epsilon),
        elems_per_row(elems_per_row),
        outer_iters(outer_iters) {}

  void operator()(sycl::nd_item<3> item_ct1) const {
    constexpr int T_per_load = granularity / sizeof(T);

    sycl::group<3> tb = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();

    const int block_offset = (tb.get_group_id()[2] *
                              (maxThreads / threadsPerGroup) * elems_per_row) +
        (tb.get_local_id()[1] * elems_per_row);
    const int thread_offset = tb.get_local_id()[2] * T_per_load;
    const int stride = item_ct1.get_local_range(2) * T_per_load;
    const int chunk_stride = UNROLL * stride;

    float var_sum = reduce::init<rop::Add, float>();

    for (int outer = 0; outer < outer_iters; outer++) {
      const int outer_off = outer * chunk_stride;
      T iter_buf[UNROLL * T_per_load];
      T res_buf[UNROLL * T_per_load];
#pragma unroll
      for (int i = 0; i < UNROLL; i++) {
        T* ib = iter_buf + i * T_per_load;
        T* rb = res_buf + i * T_per_load;
        const int total_off = thread_offset + outer_off + i * stride;
        const bool do_loads = total_off < elems_per_row;
        mem_access::load_global<granularity>(
            ib, vals + block_offset + outer_off + thread_offset + i * stride,
            do_loads);
        mem_access::load_global<granularity>(
            rb,
            residual + block_offset + outer_off + thread_offset + i * stride,
            do_loads);
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
          if (total_off + j < elems_per_row) {
            ib[j] += rb[j];
            float up = conversion::to<float>(ib[j]);
            var_sum = reduce::element<rop::Add, float>(var_sum, up * up);
          }
        }
      }
    }

    reduce::partitioned_block<rop::Add, threadsPerGroup>(tb, warp, var_sum);
    const float denom = sycl::rsqrt(var_sum / elems_per_row + epsilon);

    for (int outer = 0; outer < outer_iters; outer++) {
      const int outer_off = outer * chunk_stride;
      T iter_buf[UNROLL * T_per_load];
      T res_buf[UNROLL * T_per_load];
      T gamma_buf[UNROLL * T_per_load];
#pragma unroll
      for (int i = 0; i < UNROLL; i++) {
        T* ib = iter_buf + i * T_per_load;
        T* rb = res_buf + i * T_per_load;
        T* gb = gamma_buf + i * T_per_load;
        const int total_off = thread_offset + outer_off + i * stride;
        const bool do_loads = total_off < elems_per_row;
        mem_access::load_global<granularity>(
            ib, vals + block_offset + outer_off + thread_offset + i * stride,
            do_loads);
        mem_access::load_global<granularity>(
            rb,
            residual + block_offset + outer_off + thread_offset + i * stride,
            do_loads);
        mem_access::load_global<granularity>(
            gb, gamma + outer_off + thread_offset + i * stride, do_loads);
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
          ib[j] += rb[j];
          float v = conversion::to<float>(ib[j]) * denom *
              conversion::to<float>(gb[j]);
          ib[j] = conversion::to<T>(v);
        }
        if (do_loads) {
          mem_access::store_global<granularity>(
              output + block_offset + outer_off + thread_offset + i * stride,
              ib);
        }
      }
    }
  }
};

template <typename T>
void launch_rms_norm_big(
    T* norm_output,
    const T* vals,
    const T* residual,
    const T* gamma,
    float epsilon,
    int rows,
    int elems_per_row,
    sycl::queue* stream) {
  constexpr int T_per_load = granularity / sizeof(T);
  constexpr int internalUnroll = sizeof(T) == 4 ? 4 : 2;
  constexpr int UNROLL = internalUnroll;
  constexpr int maxThreads = 1024;

  const int h_per_step = T_per_load * UNROLL;
  const int chunk = maxThreads * h_per_step;
  const int outer_iters = (elems_per_row + chunk - 1) / chunk;

  sycl::range<3> block(1, 1, maxThreads);
  sycl::range<3> grid(1, 1, static_cast<size_t>(rows));

  bool pre_norm = (residual != nullptr);
  if (pre_norm) {
    stream->submit([&](sycl::handler& cgh) {
      pre_rms_norm_big<T, UNROLL, maxThreads, maxThreads> fn(
          norm_output, vals, residual, gamma, epsilon,
          elems_per_row, outer_iters);
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block), fn);
    });
  } else {
    stream->submit([&](sycl::handler& cgh) {
      rms_norm_big<T, UNROLL, maxThreads, maxThreads> fn(
          norm_output, vals, gamma, epsilon, elems_per_row, outer_iters);
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block), fn);
    });
  }
}

#define LAUNCH_RMS_NORM(UNROLL, threadsPerGroup, maxThreads)           \
  stream->submit([&](sycl::handler& cgh) {                             \
    rms_norm<T, UNROLL, threadsPerGroup, maxThreads> fn(               \
        norm_output, vals, gamma, epsilon, elems_per_row);             \
    cgh.parallel_for(sycl::nd_range<3>(grid * block, block), fn);      \
  });

#define LAUNCH_PRE_RMS_NORM(UNROLL, threadsPerGroup, maxThreads)       \
  stream->submit([&](sycl::handler& cgh) {                             \
    pre_rms_norm<T, UNROLL, threadsPerGroup, maxThreads> fn(           \
        norm_output,                                                    \
        vals,                                                           \
        residual,                                                       \
        gamma,                                                          \
        epsilon,                                                        \
        elems_per_row);                                                 \
    cgh.parallel_for(sycl::nd_range<3>(grid * block, block), fn);      \
  });

#define LAUNCH_ALL_RMS_NORM(UNROLL, threadsPerGroup, maxThreads) \
  if (pre_norm) {                                                \
    LAUNCH_PRE_RMS_NORM(UNROLL, threadsPerGroup, maxThreads)     \
  } else {                                                       \
    LAUNCH_RMS_NORM(UNROLL, threadsPerGroup, maxThreads)         \
  }

template <typename T>
void launch_rms_norm(
    T* norm_output,
    const T* vals,
    const T* residual,
    const T* gamma,
    float epsilon,
    int rows,
    int elems_per_row,
    sycl::queue* stream) {
  constexpr int T_per_load = granularity / sizeof(T);
  constexpr int maxThreads = 256;
  constexpr int internalUnroll = sizeof(T) == 4 ? 4 : 2;

  const bool is_subblock_schedule = (elems_per_row <= 128) ? true : false;
  const int h_per_step = T_per_load * internalUnroll;

  const int one_step_threads =
      next_pow2((elems_per_row + h_per_step - 1) / h_per_step);
  const int threads_per_group =
      (one_step_threads < maxThreads) ? one_step_threads : maxThreads;

  const int groups_per_block_max = is_subblock_schedule
      ? (maxThreads + threads_per_group - 1) / threads_per_group
      : 1;
  const int groups_per_block =
      (rows < groups_per_block_max) ? rows : groups_per_block_max;
  const int groups_launch = (groups_per_block + rows - 1) / groups_per_block;

  sycl::range<3> block(1, groups_per_block, threads_per_group);
  sycl::range<3> grid(1, 1, groups_launch);

  const int elems_per_step = threads_per_group * h_per_step;
  const int external_unRoll =
      (elems_per_row + elems_per_step - 1) / elems_per_step;

  bool pre_norm = (residual == nullptr) ? false : true;

  if (is_subblock_schedule) {
    if (threads_per_group == 1) {
      LAUNCH_ALL_RMS_NORM(internalUnroll, 1, maxThreads);
    } else if (threads_per_group == 2) {
      LAUNCH_ALL_RMS_NORM(internalUnroll, 2, maxThreads);
    } else if (threads_per_group == 4) {
      LAUNCH_ALL_RMS_NORM(internalUnroll, 4, maxThreads);
    } else if (threads_per_group == 8) {
      LAUNCH_ALL_RMS_NORM(internalUnroll, 8, maxThreads);
    } else if (threads_per_group == 16) {
      LAUNCH_ALL_RMS_NORM(internalUnroll, 16, maxThreads);
    }
  } else if (external_unRoll == 1) {
    LAUNCH_ALL_RMS_NORM(1 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unRoll == 2) {
    LAUNCH_ALL_RMS_NORM(2 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unRoll == 3) {
    LAUNCH_ALL_RMS_NORM(3 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unRoll == 4) {
    LAUNCH_ALL_RMS_NORM(4 * internalUnroll, maxThreads, maxThreads);
  }
}

#undef LAUNCH_RMS_NORM
#undef LAUNCH_PRE_RMS_NORM
#undef LAUNCH_ALL_RMS_NORM

template <typename scalar_t>
void call_rms_norm_kernel(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  constexpr int kVecElems = 16 / sizeof(scalar_t);
  TORCH_CHECK(
      hidden_size % kVecElems == 0,
      "rms_norm: hidden_size (",
      hidden_size,
      ") must be a multiple of ",
      kVecElems,
      " for dtype with sizeof=",
      sizeof(scalar_t),
      " (kernel uses 16-byte vectorized load/store)");

  auto out_ptr = out.data_ptr<scalar_t>();
  auto input_ptr = input.data_ptr<scalar_t>();
  auto weight_ptr = weight.data_ptr<scalar_t>();

  if (hidden_size > 16384) {
    auto& queue = vllm::xpu::vllmGetQueue();
    launch_rms_norm_big<sycl_t>(
        reinterpret_cast<sycl_t*>(out_ptr),
        reinterpret_cast<const sycl_t*>(input_ptr),
        nullptr,
        reinterpret_cast<const sycl_t*>(weight_ptr),
        epsilon,
        num_tokens,
        hidden_size,
        &queue);
    return;
  }

  auto& queue = vllm::xpu::vllmGetQueue();
  launch_rms_norm<sycl_t>(
      reinterpret_cast<sycl_t*>(out_ptr),
      reinterpret_cast<const sycl_t*>(input_ptr),
      nullptr,
      reinterpret_cast<const sycl_t*>(weight_ptr),
      epsilon,
      num_tokens,
      hidden_size,
      &queue);
}

template <typename scalar_t>
void call_fused_add_rms_norm_kernel(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  constexpr int kVecElems = 16 / sizeof(scalar_t);
  TORCH_CHECK(
      hidden_size % kVecElems == 0,
      "fused_add_rms_norm: hidden_size (",
      hidden_size,
      ") must be a multiple of ",
      kVecElems,
      " for dtype with sizeof=",
      sizeof(scalar_t),
      " (kernel uses 16-byte vectorized load/store)");

  auto input_ptr = input.data_ptr<scalar_t>();
  auto residual_ptr = residual.data_ptr<scalar_t>();
  auto weight_ptr = weight.data_ptr<scalar_t>();

  if (hidden_size > 16384) {
    auto& queue = vllm::xpu::vllmGetQueue();
    launch_rms_norm_big<sycl_t>(
        reinterpret_cast<sycl_t*>(input_ptr),
        reinterpret_cast<const sycl_t*>(input_ptr),
        reinterpret_cast<const sycl_t*>(residual_ptr),
        reinterpret_cast<const sycl_t*>(weight_ptr),
        epsilon,
        num_tokens,
        hidden_size,
        &queue);
    return;
  }

  auto& queue = vllm::xpu::vllmGetQueue();
  launch_rms_norm<sycl_t>(
      reinterpret_cast<sycl_t*>(input_ptr),
      reinterpret_cast<const sycl_t*>(input_ptr),
      reinterpret_cast<const sycl_t*>(residual_ptr),
      reinterpret_cast<const sycl_t*>(weight_ptr),
      epsilon,
      num_tokens,
      hidden_size,
      &queue);
}

}  // namespace rms_ipex_detail
}  // namespace vllm

void rms_norm(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(out.is_contiguous());
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }
  TORCH_CHECK(weight.is_contiguous());

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_rms_norm_kernel", [&] {
        vllm::rms_ipex_detail::call_rms_norm_kernel<scalar_t>(
            out, input, weight, static_cast<float>(epsilon));
      });
}

void fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_fused_add_rms_norm_kernel", [&] {
        vllm::rms_ipex_detail::call_fused_add_rms_norm_kernel<scalar_t>(
            input, residual, weight, static_cast<float>(epsilon));
      });
}
