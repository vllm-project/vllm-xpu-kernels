#pragma once

#include <algorithm>
#include <functional>
#include <type_traits>

#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "chunk_kda_kernels_xe2.hpp"

namespace kda_xe2 {

template <typename T, typename StateT>
class ChunkKdaPrepareKernel;
template <typename T, typename StateT, int V>
class ChunkKdaPrepareVecKernel;
template <typename T, typename StateT>
class ChunkKdaComputeAKernel;
template <typename T, typename StateT>
class ChunkKdaInverseKernel;
template <typename T, typename StateT>
class ChunkKdaInverseOptKernel;
template <typename T, typename StateT>
class ChunkKdaComputeWUKernel;
template <typename T, typename StateT>
class ChunkKdaFwdOKernel;

// Returns false when `abort_after_prepare` reports that stage 1 produced a
// cumulative log-decay the later stages cannot represent. In that case only the
// scratch buffers have been written, so the caller is free to fall back to the
// recurrent backend with the inputs and the state cache untouched.
template <typename T, typename StateT>
bool chunk_kda_launcher(
    sycl::queue& queue,
    T* core_attn_out,
    const T* q,
    const T* k,
    const T* v,
    const T* raw_gate,
    const float* raw_beta,
    const float* a_log,
    const float* dt_bias,
    const float lower_bound,
    int* saturated,
    T* Ka,
    T* Kb,
    T* Qt,
    T* Vp,
    T* A,
    T* W,
    T* U,
    float* Tl,
    StateT* recurrent_state,
    int64_t recurrent_state_stride_0,
    const int* query_start_loc,
    const int* state_indices,
    const bool* has_initial_state,
    const int* token_indx,
    const int batch_size,
    const int total_virtual_seqlen,
    const int num_heads,
    const int head_dim,
    const std::function<bool()>* abort_after_prepare = nullptr) {
  using Element_non_CV = cutlass::platform::remove_cv_t<T>;
  auto op = XE_DPAS_TT<8, float, Element_non_CV>{};

  const int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  // Every stage is decomposed as one work-group per (chunk, head). Sizing the
  // launch from the actual chunk count (rather than a fixed occupancy target)
  // is what keeps all Xe-cores busy: with a chunk-only grid the per-work-group
  // head loop would serialise `num_heads` chunks' worth of work.
  const int max_chunks = total_virtual_seqlen / chunk_size;
  const int chunk_wgs = std::min(max_chunks, 2048);
  const int chunk_head_wgs = chunk_wgs * num_heads;

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{
      syclex::sub_group_size<cute::detail::subgroup_size>,
      intelex::grf_size<256>};
  syclex::properties prepare_props{
      syclex::sub_group_size<prepare_sub_group_size>};

  // --- stage 1: prepare -----------------------------------------------------
  // The vectorized variant needs `head_dim == prepare_sub_group_size * V` with
  // a power-of-two V so the per-lane pack is naturally aligned. Everything else
  // falls back to the scalar one-thread-per-channel kernel.
  const int prepare_vec_width = head_dim / prepare_sub_group_size;
  const bool prepare_vectorizable =
      head_dim == prepare_vec_width * prepare_sub_group_size &&
      (prepare_vec_width == 2 || prepare_vec_width == 4 ||
       prepare_vec_width == 8);

  if (prepare_vectorizable) {
    // One sub-group owns a whole (chunk, head), so the launch is sized in
    // sub-groups; the kernel drops any tail sub-group that maps past num_heads.
    const int sgs_per_wg = prepare_work_group_size / prepare_sub_group_size;
    const int wg_count = (chunk_head_wgs + sgs_per_wg - 1) / sgs_per_wg;
    sycl::range<3> local(1, 1, prepare_work_group_size);
    sycl::range<3> global(1, wg_count, 1);

    auto submit = [&](auto width) {
      constexpr int V = decltype(width)::value;
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<ChunkKdaPrepareVecKernel<T, StateT, V>>(
            sycl::nd_range<3>{global * local, local}, prepare_props, [=](auto) {
              chunk_kda_prepare_vec_kernel<T, V>(
                  Ka,
                  Kb,
                  Qt,
                  Vp,
                  Tl,
                  q,
                  k,
                  v,
                  raw_gate,
                  raw_beta,
                  a_log,
                  dt_bias,
                  lower_bound,
                  saturated,
                  query_start_loc,
                  token_indx,
                  total_virtual_seqlen,
                  batch_size,
                  num_heads,
                  head_dim);
            });
      });
    };

    if (prepare_vec_width == 2) {
      submit(std::integral_constant<int, 2>{});
    } else if (prepare_vec_width == 4) {
      submit(std::integral_constant<int, 4>{});
    } else {
      submit(std::integral_constant<int, 8>{});
    }
  } else {
    // Phase B of `prepare` assigns one thread per key channel, so sizing the
    // work-group to head_dim keeps every lane busy through the serial cumsum
    // instead of idling half of a fixed 256-wide group at head_dim == 128.
    const int prepare_wg =
        std::max(prepare_sub_group_size * 2, std::min(512, (int)head_dim));
    sycl::range<3> local(1, 1, prepare_wg);
    sycl::range<3> global(1, chunk_head_wgs, 1);
    const int slm_size = chunk_size * 4;
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> local_mem(sycl::range<1>(slm_size), cgh);
      cgh.parallel_for<ChunkKdaPrepareKernel<T, StateT>>(
          sycl::nd_range<3>{global * local, local}, prepare_props, [=](auto) {
            chunk_kda_prepare_kernel<T>(
                local_mem,
                Ka,
                Kb,
                Qt,
                Vp,
                Tl,
                q,
                k,
                v,
                raw_gate,
                raw_beta,
                a_log,
                dt_bias,
                lower_bound,
                saturated,
                query_start_loc,
                token_indx,
                total_virtual_seqlen,
                batch_size,
                num_heads,
                head_dim);
          });
    });
  }

  if (abort_after_prepare != nullptr && (*abort_after_prepare)()) {
    return false;
  }

  // --- stage 2: A = I + tril_strict(Ka @ Kb^T) ------------------------------
  using WGTileA = chunk_gemm_policy_compute_A::WGTile;
  using SGLayoutA = chunk_gemm_policy_compute_A::SGLayout;
  using MMAComputeA = typename cute::TiledMMAHelper<
      cute::MMA_Atom<decltype(op)>,
      cute::Layout<WGTileA>,
      SGLayoutA>::TiledMMA;
  {
    auto mma = MMAComputeA{};
    const int wg_size = cute::size(mma);
    sycl::range<3> local(1, 1, wg_size);
    sycl::range<3> global(1, chunk_head_wgs, 1);
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<ChunkKdaComputeAKernel<T, StateT>>(
          sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
            chunk_kda_compute_A_kernel<T, MMAComputeA>(
                A,
                Ka,
                Kb,
                query_start_loc,
                total_virtual_seqlen,
                batch_size,
                num_heads,
                head_dim);
          });
    });
  }

  // --- stage 3: A := (I + A)^-1 ---------------------------------------------
  using WGTileInv = chunk_gemm_policy_inverse::WGTile;
  using SGLayoutInv = chunk_gemm_policy_inverse::SGLayout;
  using MMAInverse = typename cute::TiledMMAHelper<
      cute::MMA_Atom<decltype(op)>,
      cute::Layout<WGTileInv>,
      SGLayoutInv>::TiledMMA;
  if (vllm::xpu::is_bmg()) {
    auto mma = MMAInverse{};
    const int wg_size = cute::size(mma);
    sycl::range<3> local(1, 1, wg_size);
    // The kernel derives (chunk, head) from the group index, so the group
    // count has to stay a whole multiple of num_heads.
    sycl::range<3> global(1, chunk_head_wgs, 1);
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<ChunkKdaInverseOptKernel<T, StateT>>(
          sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
            chunk_kda_inverse_opt_kernel<T, MMAInverse>(
                A,
                query_start_loc,
                total_virtual_seqlen,
                batch_size,
                num_heads);
          });
    });
  } else {
    // PVC hits an accumulator issue with the blocked DPAS inverse, so keep the
    // scalar forward substitution available there.
    const int wg_size = 64;
    sycl::range<3> local(1, 1, wg_size);
    sycl::range<3> global(1, sm_count * MaxThreadsPerSM / wg_size, 1);
    const int slm_size = chunk_size * chunk_size * 2;
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> local_mem(sycl::range<1>(slm_size), cgh);
      cgh.parallel_for<ChunkKdaInverseKernel<T, StateT>>(
          sycl::nd_range<3>{global * local, local}, [=](auto) {
            chunk_kda_inverse_kernel<T>(
                local_mem,
                A,
                query_start_loc,
                total_virtual_seqlen,
                batch_size,
                num_heads);
          });
    });
  }

  // --- stage 4: W = A^-1 @ Ka, U = A^-1 @ Vp --------------------------------
  using WGTileWU = chunk_gemm_policy_compute_wu::WGTile;
  using SGLayoutWU = chunk_gemm_policy_compute_wu::SGLayout;
  using MMAComputeWU = typename cute::TiledMMAHelper<
      cute::MMA_Atom<decltype(op)>,
      cute::Layout<WGTileWU>,
      SGLayoutWU>::TiledMMA;
  {
    auto mma = MMAComputeWU{};
    const int wg_size = cute::size(mma);
    sycl::range<3> local(1, 1, wg_size);
    sycl::range<3> global(1, chunk_head_wgs, 1);
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<ChunkKdaComputeWUKernel<T, StateT>>(
          sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
            chunk_kda_compute_wu_kernel<T, MMAComputeWU>(
                A,
                W,
                U,
                Ka,
                Vp,
                query_start_loc,
                total_virtual_seqlen,
                batch_size,
                num_heads,
                head_dim);
          });
    });
  }

  // --- stage 5: sequential inter-chunk carry --------------------------------
  using WGTileO = chunk_gemm_policy_fwd_o::WGTile;
  using SGLayoutO = chunk_gemm_policy_fwd_o::SGLayout;
  using MMAFwdO = typename cute::TiledMMAHelper<
      cute::MMA_Atom<decltype(op)>,
      cute::Layout<WGTileO>,
      SGLayoutO>::TiledMMA;
  {
    auto mma = MMAFwdO{};
    const int wg_size = cute::size(mma);
    const int dv_groups =
        chunk_kda_fwd_o_dv_groups(batch_size, num_heads, head_dim);
    sycl::range<3> local(1, 1, wg_size);
    sycl::range<3> global(batch_size, num_heads * dv_groups, 1);
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> local_mem(sycl::range<1>(head_dim), cgh);
      cgh.parallel_for<ChunkKdaFwdOKernel<T, StateT>>(
          sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
            chunk_kda_fwd_o_kernel<T, StateT, MMAFwdO>(
                local_mem,
                core_attn_out,
                A,
                W,
                U,
                Qt,
                Kb,
                Tl,
                recurrent_state,
                recurrent_state_stride_0,
                query_start_loc,
                state_indices,
                has_initial_state,
                token_indx,
                batch_size,
                total_virtual_seqlen,
                num_heads,
                head_dim,
                dv_groups);
          });
    });
  }

  return true;
}

}  // namespace kda_xe2
