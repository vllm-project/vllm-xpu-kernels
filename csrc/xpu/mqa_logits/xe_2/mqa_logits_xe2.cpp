#include <limits>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <c10/xpu/XPUStream.h>

#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/platform/platform.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/device_kernel.h"
#include "cutlass/util/packed_stride.hpp"
#include "cute/atom/copy_traits_xe_2d.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cute/util/compat/device.hpp"
#include "cute/util/compat/dims.hpp"
#include "cute/util/compat/launch_policy.hpp"

#include "mqa_logits_xe2.h"

using namespace cute;

class mma_policy_base {
 public:
  using WGTile = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyD = void;
};

class w8a8_policy_m_32 : public mma_policy_base {
 public:
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class fp8_mqa_logits_kernel_t {
 public:
  using MMAOperation = XE_DPAS_TT<8, float, bfloat16_t>;

  using MqaPolicy = w8a8_policy_m_32;
  using WGTile = typename MqaPolicy::WGTile;
  using SGLayout = typename MqaPolicy::SGLayout;

  using MMA = typename TiledMMAHelper<
      MMA_Atom<MMAOperation>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;

  static constexpr int64_t kBlockHeads = get<0>(typename MqaPolicy::WGTile{});
  static constexpr int64_t kBlockKV = get<1>(typename MqaPolicy::WGTile{});
  static constexpr int64_t mma_k_tile = get<2>(typename MqaPolicy::WGTile{});
  static constexpr int64_t threads_per_wg = size(MMA{});

  class Params {
   public:
    const float_e4m3_t* q_ptr;
    const float_e4m3_t* kv_ptr;
    const float* scales_ptr;
    const float* weights_ptr;
    const int32_t* ks_ptr;
    const int32_t* ke_ptr;
    float* out_ptr;
    int64_t seq_len;
    int64_t num_heads;
    int64_t head_dim;
    int64_t seq_len_kv;
  };

  CUTLASS_DEVICE
  void operator()(const Params& p, char* smem_buf) const {
    auto q_tensor = make_tensor(
        p.q_ptr,
        make_layout(
            make_shape(p.seq_len, p.num_heads, p.head_dim),
            make_stride(p.num_heads * p.head_dim, p.head_dim, _1{})));
    auto kv_tensor = make_tensor(
        p.kv_ptr,
        make_layout(
            make_shape(p.seq_len_kv, p.head_dim),
            make_stride(p.head_dim, _1{})));
    auto scales_tensor = make_tensor(
        p.scales_ptr, make_layout(make_shape(p.seq_len_kv), make_stride(_1{})));
    auto weights_tensor = make_tensor(
        p.weights_ptr,
        make_layout(
            make_shape(p.seq_len, p.num_heads),
            make_stride(p.num_heads, _1{})));
    auto ks_tensor = make_tensor(
        p.ks_ptr, make_layout(make_shape(p.seq_len), make_stride(_1{})));
    auto ke_tensor = make_tensor(
        p.ke_ptr, make_layout(make_shape(p.seq_len), make_stride(_1{})));
    auto out_tensor = make_tensor(
        p.out_ptr,
        make_layout(
            make_shape(p.seq_len, p.seq_len_kv),
            make_stride(p.seq_len_kv, _1{})));

    auto num_heads = p.num_heads;
    auto head_dim = p.head_dim;
    auto seq_len = p.seq_len;
    auto seq_len_kv = p.seq_len_kv;

    const float neg_inf = -std::numeric_limits<float>::infinity();

    const auto q_token_idx = int(BlockIdxX());
    auto kv_block_idx = int(BlockIdxY());
    auto local_id = int(ThreadIdxY());

    Tensor curr_q_tensor = q_tensor(q_token_idx, _, _);
    Tensor curr_weights_tensor = weights_tensor(q_token_idx, _);
    Tensor curr_out_tensor = out_tensor(q_token_idx, _);
    int32_t ks = ks_tensor(q_token_idx);
    int32_t ke = ke_tensor(q_token_idx);

    const int64_t kv_block_start = kv_block_idx * kBlockKV;
    if (kv_block_start >= seq_len_kv) {
      return;
    }
    const int64_t kv_block_end =
        cute::min(kv_block_start + kBlockKV, seq_len_kv);

    if (ke <= kv_block_start || ks >= kv_block_end) {
      const int64_t kv_index = kv_block_start + local_id;
      if (kv_index < seq_len_kv) {
        curr_out_tensor(kv_index) = neg_inf;
      }
      return;
    }

    Tensor cQ = make_identity_tensor(curr_q_tensor.shape());
    Tensor cKV = make_identity_tensor(kv_tensor.shape());
    Tensor cScales = make_identity_tensor(scales_tensor.shape());
    Tensor cWeights = make_identity_tensor(curr_weights_tensor.shape());
    Tensor cOut = make_identity_tensor(curr_out_tensor.shape());

    auto mma = MMA{};
    auto wg_tile = mma.tile_mnk();

    Tensor gQ = local_tile(cQ, select<0, 2>(wg_tile), make_coord(_, _));
    Tensor gKV =
        local_tile(cKV, select<1, 2>(wg_tile), make_coord(kv_block_idx, _));
    Tensor gScales =
        local_tile(cScales, select<1>(wg_tile), make_coord(kv_block_idx));
    Tensor gWeights = local_tile(cWeights, select<0>(wg_tile), make_coord(_));
    Tensor gOut =
        local_tile(cOut, select<1>(wg_tile), make_coord(kv_block_idx));

    auto copy_q = make_block_2d_copy_A(mma, curr_q_tensor);
    auto copy_kv = make_block_2d_copy_B(mma, kv_tensor);

    auto thr_mma = mma.get_slice(local_id);
    auto thr_copy_q = copy_q.get_slice(local_id);
    auto thr_copy_kv = copy_kv.get_slice(local_id);

    auto tCrA = thr_mma.partition_sg_fragment_A(gQ(_, _, 0, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gKV(_, _, 0));

    auto tArA = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0, 0));
    auto tBrB = thr_copy_kv.partition_sg_fragment_D(gKV(_, _, 0));

    Tensor tAgA = thr_copy_q.partition_S(gQ);
    Tensor tBgB = thr_copy_kv.partition_S(gKV);

    Tensor tCrC = partition_fragment_C(mma, select<0, 1>(wg_tile));

    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_kv = make_block_2d_prefetch(copy_kv);

    auto thr_prefetch_q = prefetch_q.get_slice(local_id);
    auto thr_prefetch_kv = prefetch_kv.get_slice(local_id);

    auto pAgA = thr_prefetch_q.partition_S(gQ);
    auto pBgB = thr_prefetch_kv.partition_S(gKV);

    const int prefetch_dist = 3;

    int head_tile_count = ceil_div(num_heads, kBlockHeads);
    int k_tile_count = ceil_div(head_dim, mma_k_tile);

    const int barrier_scope = 2;

    float output = 0;
    const int64_t kv_index = kv_block_start + local_id;
    const float scale = kv_index < seq_len_kv ? scales_tensor(kv_index) : 0.0f;
    float* weights_smem = reinterpret_cast<float*>(smem_buf);

    for (int64_t head_tile = 0; head_tile < head_tile_count; head_tile++) {
      int k_tile_prefetch = 0;
      clear(tCrC);

      barrier_arrive(barrier_scope);
      barrier_wait(barrier_scope);

      if (local_id < kBlockHeads) {
        weights_smem[local_id] =
            curr_weights_tensor(head_tile * kBlockHeads + local_id);
      }

      barrier_arrive(barrier_scope);
      barrier_wait(barrier_scope);

      CUTE_UNROLL
      for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
        prefetch(prefetch_q, pAgA(_, _, _, head_tile, k_tile_prefetch));
        prefetch(prefetch_kv, pBgB(_, _, _, k_tile_prefetch));
      }

      for (int64_t k_tile = 0; k_tile < k_tile_count;
           k_tile++, k_tile_prefetch++) {
        barrier_arrive(barrier_scope);

        copy(copy_q, tAgA(_, _, _, head_tile, k_tile), tArA);
        copy(copy_kv, tBgB(_, _, _, k_tile), tBrB);

        if (k_tile_prefetch < k_tile_count) {
          prefetch(prefetch_q, pAgA(_, _, _, head_tile, k_tile_prefetch));
          prefetch(prefetch_kv, pBgB(_, _, _, k_tile_prefetch));
        }

        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);

        gemm(mma, tCrA, tCrB, tCrC);
        barrier_wait(barrier_scope);
      }

      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); i++) {
        tCrC[i] = tCrC[i] * scale;
      }

      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); i++) {
        tCrC[i] = tCrC[i] > 0 ? tCrC[i] : 0;
      }

      CUTE_UNROLL
      for (int i = 0; i < size<0>(tCrC); i++) {
        for (int j = 0; j < size<1>(tCrC); j++) {
          int wei_index = j * size<0>(tCrC) + i;
          tCrC(i, j, 0) = tCrC(i, j, 0) * weights_smem[wei_index];
        }
      }

      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); i++) {
        output += tCrC[i];
      }
    }

    if (kv_index >= seq_len_kv) {
      return;
    }

    curr_out_tensor(kv_index) =
        (kv_index >= ks && kv_index < ke) ? output : neg_inf;
  }
};

class fp8_paged_mqa_logits_kernel_t {
 public:
  using MMAOperation = XE_DPAS_TT<8, float, bfloat16_t>;

  using MqaPolicy = w8a8_policy_m_32;
  using WGTile = typename MqaPolicy::WGTile;
  using SGLayout = typename MqaPolicy::SGLayout;

  using MMA = typename TiledMMAHelper<
      MMA_Atom<MMAOperation>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;

  static constexpr int64_t kBlockHeads = get<0>(typename MqaPolicy::WGTile{});
  static constexpr int64_t kBlockKV = get<1>(typename MqaPolicy::WGTile{});
  static constexpr int64_t mma_k_tile = get<2>(typename MqaPolicy::WGTile{});
  static constexpr int64_t threads_per_wg = size(MMA{});

  class Params {
   public:
    const float_e4m3_t* q_ptr;
    const float_e4m3_t* kv_ptr;
    const float* scales_ptr;
    const float* weights_ptr;
    const int32_t* context_ptr;
    const int32_t* block_tables_ptr;
    float* out_ptr;
    int64_t batch_size;
    int64_t next_n;
    int64_t heads;
    int64_t index_dim;
    int64_t num_blocks;
    int64_t block_size;
    int64_t max_blocks;
    int64_t max_model_len;
    int64_t kv_stride0;
    int64_t scale_stride0;
  };

  CUTLASS_DEVICE
  void operator()(const Params& p, char* smem_buf) const {
    auto q_tensor = make_tensor(
        p.q_ptr,
        make_layout(
            make_shape(p.batch_size, p.next_n, p.heads, p.index_dim),
            make_stride(
                p.next_n * p.heads * p.index_dim,
                p.heads * p.index_dim,
                p.index_dim,
                _1{})));
    auto kv_tensor = make_tensor(
        p.kv_ptr,
        make_layout(
            make_shape(p.num_blocks, p.block_size, p.index_dim),
            make_stride(p.kv_stride0, p.index_dim, _1{})));
    auto scales_tensor = make_tensor(
        p.scales_ptr,
        make_layout(
            make_shape(p.num_blocks, p.block_size),
            make_stride(p.scale_stride0, _1{})));
    auto weights_tensor = make_tensor(
        p.weights_ptr,
        make_layout(
            make_shape(p.batch_size, p.next_n, p.heads),
            make_stride(p.next_n * p.heads, p.heads, _1{})));
    auto context_tensor = make_tensor(
        p.context_ptr,
        make_layout(make_shape(p.batch_size), make_stride(_1{})));
    auto block_tables_tensor = make_tensor(
        p.block_tables_ptr,
        make_layout(
            make_shape(p.batch_size, p.max_blocks),
            make_stride(p.max_blocks, _1{})));
    auto out_tensor = make_tensor(
        p.out_ptr,
        make_layout(
            make_shape(p.batch_size, p.next_n, p.max_model_len),
            make_stride(p.next_n * p.max_model_len, p.max_model_len, _1{})));

    auto batch_idx = int(BlockIdxX());
    auto logical_block_idx = int(BlockIdxY());
    auto local_id = int(ThreadIdxY());

    if (logical_block_idx >= p.max_blocks) {
      return;
    }

    int32_t context_len = context_tensor(batch_idx);
    int64_t block_start =
        static_cast<int64_t>(logical_block_idx) * p.block_size;
    if (block_start >= context_len) {
      return;
    }

    int32_t physical_block = block_tables_tensor(batch_idx, logical_block_idx);
    if (physical_block < 0 || physical_block >= p.num_blocks) {
      return;
    }

    int64_t block_end = cute::min(
        block_start + p.block_size, static_cast<int64_t>(context_len));
    int64_t actual_block_size = block_end - block_start;

    auto mma = MMA{};
    auto wg_tile = mma.tile_mnk();

    Tensor curr_kv_tensor = kv_tensor(physical_block, _, _);
    Tensor curr_scales_tensor = scales_tensor(physical_block, _);

    Tensor cKV = make_identity_tensor(curr_kv_tensor.shape());
    Tensor gKV = local_tile(cKV, select<1, 2>(wg_tile), make_coord(0, _));

    auto copy_kv = make_block_2d_copy_B(mma, curr_kv_tensor);
    auto prefetch_kv = make_block_2d_prefetch(copy_kv);

    auto thr_mma = mma.get_slice(local_id);
    auto thr_copy_kv = copy_kv.get_slice(local_id);
    auto thr_prefetch_kv = prefetch_kv.get_slice(local_id);

    auto tCrB = thr_mma.partition_sg_fragment_B(gKV(_, _, 0));
    auto tBrB = thr_copy_kv.partition_sg_fragment_D(gKV(_, _, 0));
    Tensor tBgB = thr_copy_kv.partition_S(gKV);
    auto pBgB = thr_prefetch_kv.partition_S(gKV);

    int head_tile_count = ceil_div(p.heads, kBlockHeads);
    int k_tile_count = ceil_div(p.index_dim, mma_k_tile);
    const int prefetch_dist = 3;
    const int barrier_scope = 2;

    for (int64_t q_token_id = 0; q_token_id < p.next_n; q_token_id++) {
      int64_t q_offset =
          static_cast<int64_t>(context_len) - p.next_n + q_token_id;
      if (q_offset < block_start) {
        continue;
      }

      Tensor curr_q_tensor = q_tensor(batch_idx, q_token_id, _, _);
      Tensor curr_weights_tensor = weights_tensor(batch_idx, q_token_id, _);
      Tensor curr_out_tensor = out_tensor(batch_idx, q_token_id, _);

      Tensor cQ = make_identity_tensor(curr_q_tensor.shape());
      Tensor gQ = local_tile(cQ, select<0, 2>(wg_tile), make_coord(_, _));

      auto copy_q = make_block_2d_copy_A(mma, curr_q_tensor);
      auto prefetch_q = make_block_2d_prefetch(copy_q);

      auto thr_copy_q = copy_q.get_slice(local_id);
      auto thr_prefetch_q = prefetch_q.get_slice(local_id);

      auto tCrA = thr_mma.partition_sg_fragment_A(gQ(_, _, 0, 0));
      auto tArA = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0, 0));
      Tensor tAgA = thr_copy_q.partition_S(gQ);
      auto pAgA = thr_prefetch_q.partition_S(gQ);

      Tensor tCrC = partition_fragment_C(mma, select<0, 1>(wg_tile));

      float output = 0.0f;

      for (int64_t head_tile = 0; head_tile < head_tile_count; head_tile++) {
        int k_tile_prefetch = 0;
        clear(tCrC);

        CUTE_UNROLL
        for (;
             k_tile_prefetch < prefetch_dist && k_tile_prefetch < k_tile_count;
             k_tile_prefetch++) {
          prefetch(prefetch_q, pAgA(_, _, _, head_tile, k_tile_prefetch));
          prefetch(prefetch_kv, pBgB(_, _, _, k_tile_prefetch));
        }

        for (int64_t k_tile = 0; k_tile < k_tile_count;
             k_tile++, k_tile_prefetch++) {
          barrier_arrive(barrier_scope);

          copy(copy_q, tAgA(_, _, _, head_tile, k_tile), tArA);
          copy(copy_kv, tBgB(_, _, _, k_tile), tBrB);

          if (k_tile_prefetch < k_tile_count) {
            prefetch(prefetch_q, pAgA(_, _, _, head_tile, k_tile_prefetch));
            prefetch(prefetch_kv, pBgB(_, _, _, k_tile_prefetch));
          }

          reorder(tArA, tCrA);
          reorder(tBrB, tCrB);

          gemm(mma, tCrA, tCrB, tCrC);
          barrier_wait(barrier_scope);
        }

        int64_t block_offset = local_id;
        float scale = block_offset < actual_block_size
                          ? curr_scales_tensor(block_offset)
                          : 0.0f;

        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); i++) {
          tCrC[i] = tCrC[i] * scale;
        }

        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); i++) {
          tCrC[i] = tCrC[i] > 0 ? tCrC[i] : 0;
        }

        float weight[kBlockHeads];
        CUTE_UNROLL
        for (int i = 0; i < kBlockHeads; i++) {
          int64_t head_idx = head_tile * kBlockHeads + i;
          weight[i] = head_idx < p.heads ? curr_weights_tensor(head_idx) : 0.0f;
        }

        CUTE_UNROLL
        for (int i = 0; i < size<0>(tCrC); i++) {
          for (int j = 0; j < size<1>(tCrC); j++) {
            int wei_index = j * size<0>(tCrC) + i;
            tCrC(i, j, 0) = tCrC(i, j, 0) * weight[wei_index];
          }
        }

        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); i++) {
          output += tCrC[i];
        }
      }

      int64_t kv_index = block_start + local_id;
      if (local_id >= actual_block_size || kv_index >= p.max_model_len) {
        continue;
      }
      if (kv_index <= q_offset) {
        curr_out_tensor(kv_index) = output;
      }
    }
  }
};

torch::Tensor fp8_mqa_logits_xe2(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seqlen_ks,
    const torch::Tensor& cu_seqlen_ke,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t seq_len_kv) {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  auto logits = torch::full(
      {seq_len, seq_len_kv},
      neg_inf,
      torch::dtype(torch::kFloat).device(q.device()).requires_grad(false));

  const auto* q_ptr = reinterpret_cast<const float_e4m3_t*>(q.data_ptr());
  const auto* kv_ptr = reinterpret_cast<const float_e4m3_t*>(kv.data_ptr());
  const float* scales_ptr = kv_scales.data_ptr<float>();
  const float* weights_ptr = weights.data_ptr<float>();
  const int32_t* ks_ptr = cu_seqlen_ks.data_ptr<int32_t>();
  const int32_t* ke_ptr = cu_seqlen_ke.data_ptr<int32_t>();
  float* out_ptr = logits.data_ptr<float>();

  auto queue = c10::xpu::getCurrentXPUStream().queue();

  using MqaPolicy = w8a8_policy_m_32;
  constexpr int64_t kBlockHeads = get<0>(typename MqaPolicy::WGTile{});
  constexpr int64_t kBlockKV = get<1>(typename MqaPolicy::WGTile{});

  using Kernel = fp8_mqa_logits_kernel_t;

  dim3 block(1, Kernel::threads_per_wg, 1);
  dim3 grid(
      seq_len, cute::ceil_div(seq_len_kv, static_cast<int64_t>(kBlockKV)), 1);

  const auto sycl_block = compat::dim3(block.x, block.y, block.z);
  const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

  const int smem_size = static_cast<int>(kBlockHeads * sizeof(float));

  typename Kernel::Params params{
      q_ptr,
      kv_ptr,
      scales_ptr,
      weights_ptr,
      ks_ptr,
      ke_ptr,
      out_ptr,
      seq_len,
      num_heads,
      head_dim,
      seq_len_kv};

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
  using namespace compat::experimental;
  auto event = launch<cutlass::device_kernel<Kernel>>(
      launch_policy{
          sycl_grid,
          sycl_block,
          local_mem_size{static_cast<std::size_t>(smem_size)},
          kernel_properties{sycl_exp::sub_group_size<16>}},
      queue,
      params);
#else
  compat::experimental::launch_properties launch_props{
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
  };
  compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<16>};
  compat::experimental::launch_policy policy{
      sycl_grid, sycl_block, launch_props, kernel_props};
  auto event =
      compat::experimental::launch<cutlass::device_kernel<Kernel>, Kernel>(
          policy, queue, params);
#endif

  return logits;
}

torch::Tensor fp8_paged_mqa_logits_xe2(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_tables,
    int64_t batch_size,
    int64_t next_n,
    int64_t heads,
    int64_t index_dim,
    int64_t num_blocks,
    int64_t block_size,
    int64_t max_blocks,
    int64_t max_model_len) {
  const float neg_inf = -std::numeric_limits<float>::infinity();

  auto logits = torch::full(
      {batch_size * next_n, max_model_len},
      neg_inf,
      torch::dtype(torch::kFloat).device(q.device()).requires_grad(false));

  const auto* q_ptr = reinterpret_cast<const float_e4m3_t*>(q.data_ptr());
  const auto* kv_ptr = reinterpret_cast<const float_e4m3_t*>(kv.data_ptr());
  const float* scale_ptr = kv_scales.data_ptr<float>();
  const float* weights_ptr = weights.data_ptr<float>();
  const int32_t* context_ptr = context_lens.data_ptr<int32_t>();
  const int32_t* block_tables_ptr = block_tables.data_ptr<int32_t>();
  float* out_ptr = logits.data_ptr<float>();

  const int64_t kv_stride0 = kv.stride(0);
  const int64_t kv_stride1 = kv.stride(1);
  const int64_t kv_stride3 = kv.stride(3);
  const int64_t scale_stride0 = kv_scales.stride(0);
  const int64_t scale_stride1 = kv_scales.stride(1);

  TORCH_CHECK(kv_stride1 == index_dim, "kv index_dim stride mismatch");
  TORCH_CHECK(kv_stride3 == 1, "kv last dim stride mismatch");
  TORCH_CHECK(scale_stride1 == 1, "kv_scales last dim stride mismatch");

  auto queue = c10::xpu::getCurrentXPUStream().queue();

  using Kernel = fp8_paged_mqa_logits_kernel_t;

  TORCH_CHECK(
      block_size == Kernel::kBlockKV,
      "fp8_paged_mqa_logits_xe2 currently only supports block_size == ",
      Kernel::kBlockKV,
      ", but got ",
      block_size);

  const int64_t block_count = cute::ceil_div(max_model_len, block_size);

  dim3 block(1, Kernel::threads_per_wg, 1);
  dim3 grid(batch_size, block_count, 1);

  const auto sycl_block = compat::dim3(block.x, block.y, block.z);
  const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

  const int smem_size = 0;

  typename Kernel::Params params{
      q_ptr,
      kv_ptr,
      scale_ptr,
      weights_ptr,
      context_ptr,
      block_tables_ptr,
      out_ptr,
      batch_size,
      next_n,
      heads,
      index_dim,
      num_blocks,
      block_size,
      max_blocks,
      max_model_len,
      kv_stride0,
      scale_stride0};

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
  using namespace compat::experimental;
  auto event = launch<cutlass::device_kernel<Kernel>>(
      launch_policy{
          sycl_grid,
          sycl_block,
          local_mem_size{static_cast<std::size_t>(smem_size)},
          kernel_properties{sycl_exp::sub_group_size<16>}},
      queue,
      params);
#else
  compat::experimental::launch_properties launch_props{
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
  };
  compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<16>};
  compat::experimental::launch_policy policy{
      sycl_grid, sycl_block, launch_props, kernel_props};
  auto event =
      compat::experimental::launch<cutlass::device_kernel<Kernel>, Kernel>(
          policy, queue, params);
#endif

  return logits;
}
