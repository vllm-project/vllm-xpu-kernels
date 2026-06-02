#pragma once

#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "gdn_attn_utils.h"
#include "csrc/utils.h"

namespace gdn {

static constexpr int l2norm_elem_per_item = 2;
static constexpr int l2norm_sub_group_size = 16;
static constexpr int L2NormVecSize = 8;

template <typename T>
SYCL_EXTERNAL void l2norm_kernel(
    const T* q,
    const T* k,
    const int total_virtual_seqlen,
    const int num_k_heads,
    const int head_k_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

  int group_id = item.get_group(1);
  auto sg = item.get_sub_group();
  int sg_id = sg.get_group_linear_id();
  int sg_range = sg.get_group_linear_range();
  int sg_local_id = sg.get_local_linear_id();

  int total_sg_id = group_id * sg_range + sg_id;
  const int total_qk_heads = total_virtual_seqlen * num_k_heads;
  if (total_sg_id >= total_qk_heads) {
    return;
  }
  float q_scale = sycl::rsqrt(static_cast<float>(head_k_dim));

  auto q_ptr = const_cast<T*>(q) + total_sg_id * head_k_dim;
  auto k_ptr = const_cast<T*>(k) + total_sg_id * head_k_dim;
  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int k_dim_idx = sg_local_id * l2norm_elem_per_item;
       k_dim_idx < head_k_dim;
       k_dim_idx += l2norm_sub_group_size * l2norm_elem_per_item) {
    for (int e = 0; e < l2norm_elem_per_item; ++e) {
      float q_value = q_ptr[k_dim_idx + e];
      float k_value = k_ptr[k_dim_idx + e];
      q_sum += q_value * q_value;
      k_sum += k_value * k_value;
    }
  }
  q_sum = sycl::reduce_over_group(sg, q_sum, sycl::plus<>());
  k_sum = sycl::reduce_over_group(sg, k_sum, sycl::plus<>());
  float q_inv = sycl::rsqrt(q_sum + l2norm_eps) * q_scale;
  float k_inv = sycl::rsqrt(k_sum + l2norm_eps);
  for (int k_dim_idx = sg_local_id * l2norm_elem_per_item;
       k_dim_idx < head_k_dim;
       k_dim_idx += l2norm_sub_group_size * l2norm_elem_per_item) {
    for (int e = 0; e < l2norm_elem_per_item; ++e) {
      q_ptr[k_dim_idx + e] =
          static_cast<T>(static_cast<float>(q_ptr[k_dim_idx + e]) * q_inv);
      k_ptr[k_dim_idx + e] =
          static_cast<T>(static_cast<float>(k_ptr[k_dim_idx + e]) * k_inv);
    }
  }
}

template <typename T, int VEC_SIZE>
SYCL_EXTERNAL void l2norm_vectorized_kernel(
    const T* q,
    const T* k,
    const int total_virtual_seqlen,
    const int num_k_heads,
    const int head_k_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

  using vec_t = vllm::xpu::aligned_vec<T, VEC_SIZE>;

  int group_id = item.get_group(1);
  auto sg = item.get_sub_group();
  int sg_id = sg.get_group_linear_id();
  int sg_range = sg.get_group_linear_range();
  int sg_local_id = sg.get_local_linear_id();

  int total_sg_id = group_id * sg_range + sg_id;
  const int total_qk_heads = total_virtual_seqlen * num_k_heads;
  if (total_sg_id >= total_qk_heads) {
    return;
  }

  float q_scale = sycl::rsqrt(static_cast<float>(head_k_dim));
  auto q_ptr = const_cast<T*>(q) + total_sg_id * head_k_dim;
  auto k_ptr = const_cast<T*>(k) + total_sg_id * head_k_dim;

  auto q_vec_ptr = reinterpret_cast<vec_t*>(q_ptr);
  auto k_vec_ptr = reinterpret_cast<vec_t*>(k_ptr);
  const int num_vec = head_k_dim / VEC_SIZE;

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int vec_idx = sg_local_id; vec_idx < num_vec;
       vec_idx += l2norm_sub_group_size) {
    auto q_vec = q_vec_ptr[vec_idx];
    auto k_vec = k_vec_ptr[vec_idx];
    for (int e = 0; e < VEC_SIZE; ++e) {
      float q_value = static_cast<float>(q_vec.val[e]);
      float k_value = static_cast<float>(k_vec.val[e]);
      q_sum += q_value * q_value;
      k_sum += k_value * k_value;
    }
  }

  q_sum = sycl::reduce_over_group(sg, q_sum, sycl::plus<>());
  k_sum = sycl::reduce_over_group(sg, k_sum, sycl::plus<>());
  float q_inv = sycl::rsqrt(q_sum + l2norm_eps) * q_scale;
  float k_inv = sycl::rsqrt(k_sum + l2norm_eps);

  for (int vec_idx = sg_local_id; vec_idx < num_vec;
       vec_idx += l2norm_sub_group_size) {
    auto q_vec = q_vec_ptr[vec_idx];
    auto k_vec = k_vec_ptr[vec_idx];
    for (int e = 0; e < VEC_SIZE; ++e) {
      q_vec.val[e] = static_cast<T>(static_cast<float>(q_vec.val[e]) * q_inv);
      k_vec.val[e] = static_cast<T>(static_cast<float>(k_vec.val[e]) * k_inv);
    }
    q_vec_ptr[vec_idx] = q_vec;
    k_vec_ptr[vec_idx] = k_vec;
  }
}

template <typename T>
class L2NormKernelTag;

template <typename T>
class L2NormVecKernelTag;

template <typename T>
void l2norm_launch(
    sycl::queue& queue,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const int total_virtual_seqlen,
    const int num_k_heads,
    const int head_k_dim) {
  static constexpr int MaxThreadsPerSM = 512;
  static constexpr int MaxSubgroupsPerWorkgroup =
      MaxThreadsPerSM / l2norm_sub_group_size;
  const int total_qk_heads = total_virtual_seqlen * num_k_heads;
  const int wg_count = (total_qk_heads + MaxSubgroupsPerWorkgroup - 1) /
                       MaxSubgroupsPerWorkgroup;

  sycl::range<3> local(1, 1, MaxThreadsPerSM);
  sycl::range<3> global(1, wg_count, 1);

  auto q_ptr = reinterpret_cast<T*>(q.data_ptr());
  auto k_ptr = reinterpret_cast<T*>(k.data_ptr());

  constexpr int l2norm_vec_width = sizeof(T) * L2NormVecSize;
  const bool vec_enabled =
      ((reinterpret_cast<uintptr_t>(q.data_ptr()) & (l2norm_vec_width - 1)) ==
       0) &&
      ((reinterpret_cast<uintptr_t>(k.data_ptr()) & (l2norm_vec_width - 1)) ==
       0) &&
      ((head_k_dim & (L2NormVecSize - 1)) == 0);
  queue.submit([&](sycl::handler& cgh) {
    if (vec_enabled) {
      cgh.parallel_for<L2NormVecKernelTag<T>>(
          sycl::nd_range<3>{global * local, local},
          [=](auto) [[sycl::reqd_sub_group_size(l2norm_sub_group_size)]] {
            l2norm_vectorized_kernel<T, L2NormVecSize>(
                q_ptr, k_ptr, total_virtual_seqlen, num_k_heads, head_k_dim);
          });
    } else {
      cgh.parallel_for<L2NormKernelTag<T>>(
          sycl::nd_range<3>{global * local, local},
          [=](auto) [[sycl::reqd_sub_group_size(l2norm_sub_group_size)]] {
            l2norm_kernel<T>(
                q_ptr, k_ptr, total_virtual_seqlen, num_k_heads, head_k_dim);
          });
    }
  });
}

void l2norm_impl(
    sycl::queue& queue, const torch::Tensor& q, const torch::Tensor& k) {
  const int total_virtual_seqlen = q.size(0);
  const int num_k_heads = q.size(1);
  const int head_k_dim = q.size(2);

  if (q.scalar_type() == at::kBFloat16) {
    l2norm_launch<sycl::ext::oneapi::bfloat16>(
        queue, q, k, total_virtual_seqlen, num_k_heads, head_k_dim);
  } else if (q.scalar_type() == at::kHalf) {
    l2norm_launch<sycl::half>(
        queue, q, k, total_virtual_seqlen, num_k_heads, head_k_dim);
  }
}

}  // namespace gdn
