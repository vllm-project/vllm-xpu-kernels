// Absorbed-MLA (FA3 ``q_v``) varlen attention over a paged KV cache.
//
// SYCL counterpart of the FA3 path vLLM's sparse MLA backends use on Hopper
// (flash_attn_varlen_func(q=q_pe, k=k_pe, v=kv_c, q_v=q_nope, ...)): the
// score of key n is q . k[n] + q_v . v[n], and the output is the softmax
// weighted sum of v. Unlike the cutlass paged kernels it supports any page
// size (sparse MLA passes page_size == 1 with per-token "block" ids) and
// arbitrary row strides (e.g. views into a padded MLA cache row).
//
// Work decomposition: one work-group per (query token, kv head, block of 16
// query heads). Each of the 16 sub-groups owns one query head; its 16 lanes
// hold the head's q / q_v / accumulator slices (lane holds dims lane + 16*j).
// KV rows are staged through SLM in tiles of 16 rows ([v | k] per row), one
// row loaded per sub-group, and consumed with an online softmax.

#include <sycl/sycl.hpp>

#include <cmath>

#include "mla_qv_varlen.h"
#include "utils.h"

namespace vllm::mla_qv {

namespace {

constexpr int kSgSize = 16;
constexpr int kHeadsPerWg = 16;  // one sub-group per query head
constexpr int kTileN = 16;       // KV rows per SLM tile, one per sub-group
constexpr int kWgSize = kSgSize * kHeadsPerWg;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;

static_assert(kTileN == kHeadsPerWg, "one KV row is loaded per sub-group");

template <typename T, int DQK, int DV>
class MlaQvVarlenKernel {
  static_assert(DQK % kSgSize == 0 && DV % kSgSize == 0);
  static constexpr int kQkPerLane = DQK / kSgSize;
  static constexpr int kVPerLane = DV / kSgSize;
  static constexpr int kRow = DV + DQK;  // SLM row layout: [v | k]

 public:
  MlaQvVarlenKernel(
      const T* q,
      const T* q_v,
      const T* k,
      const T* v,
      T* out,
      float* lse,
      const int32_t* cu_seqlens_q,
      const int32_t* seqused_k,
      const int32_t* block_table,
      sycl::local_accessor<T, 1> slm_kv,
      sycl::local_accessor<int32_t, 1> slm_valid,
      int64_t q_s0,
      int64_t q_s1,
      int64_t qv_s0,
      int64_t qv_s1,
      int64_t k_s0,
      int64_t k_s1,
      int64_t k_s2,
      int64_t v_s0,
      int64_t v_s1,
      int64_t v_s2,
      int64_t o_s0,
      int64_t o_s1,
      int64_t bt_s0,
      int32_t batch_size,
      int32_t total_q,
      int32_t page_size,
      int32_t num_heads_kv,
      int32_t heads_per_kv,
      int32_t num_head_blocks,
      float softmax_scale,
      bool is_causal)
      : q_(q),
        q_v_(q_v),
        k_(k),
        v_(v),
        out_(out),
        lse_(lse),
        cu_seqlens_q_(cu_seqlens_q),
        seqused_k_(seqused_k),
        block_table_(block_table),
        slm_kv_(slm_kv),
        slm_valid_(slm_valid),
        q_s0_(q_s0),
        q_s1_(q_s1),
        qv_s0_(qv_s0),
        qv_s1_(qv_s1),
        k_s0_(k_s0),
        k_s1_(k_s1),
        k_s2_(k_s2),
        v_s0_(v_s0),
        v_s1_(v_s1),
        v_s2_(v_s2),
        o_s0_(o_s0),
        o_s1_(o_s1),
        bt_s0_(bt_s0),
        batch_size_(batch_size),
        total_q_(total_q),
        page_size_(page_size),
        num_heads_kv_(num_heads_kv),
        heads_per_kv_(heads_per_kv),
        num_head_blocks_(num_head_blocks),
        softmax_scale_(softmax_scale),
        is_causal_(is_causal) {}

  [[sycl::reqd_sub_group_size(kSgSize)]] void
  operator()(sycl::nd_item<1> item) const {
    const auto sg = item.get_sub_group();
    const int sg_id = static_cast<int>(sg.get_group_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());

    int wg = static_cast<int>(item.get_group(0));
    const int hb = wg % num_head_blocks_;
    wg /= num_head_blocks_;
    const int g = wg % num_heads_kv_;
    const int t = wg / num_heads_kv_;

    // Sequence owning token t: the largest b with cu_seqlens_q[b] <= t.
    int lo = 0;
    int hi = batch_size_;
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      if (cu_seqlens_q_[mid] <= t) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const int b = lo;
    const int q_begin = cu_seqlens_q_[b];
    const int q_len = cu_seqlens_q_[b + 1] - q_begin;
    const int qi = t - q_begin;
    const int kv_len = seqused_k_[b];
    // Causal masking is aligned to the bottom-right corner, as in FA.
    int n_keys = kv_len;
    if (is_causal_) {
      n_keys = sycl::min(kv_len, kv_len - (q_len - 1 - qi));
    }
    n_keys = sycl::max(n_keys, 0);

    const int h_local = hb * kHeadsPerWg + sg_id;
    const bool head_active = h_local < heads_per_kv_;
    const int64_t h = static_cast<int64_t>(g) * heads_per_kv_ + h_local;

    // Fold softmax_scale and log2(e) into the query so scores are in the
    // exp2 domain.
    const float q_mult = softmax_scale_ * kLog2e;
    float qk[kQkPerLane];
    float qv[kVPerLane];
    float acc[kVPerLane];
#pragma unroll
    for (int j = 0; j < kVPerLane; ++j) {
      acc[j] = 0.f;
      qv[j] = 0.f;
    }
#pragma unroll
    for (int j = 0; j < kQkPerLane; ++j) {
      qk[j] = 0.f;
    }
    if (head_active) {
      const T* q_row = q_ + t * q_s0_ + h * q_s1_;
      const T* qv_row = q_v_ + t * qv_s0_ + h * qv_s1_;
#pragma unroll
      for (int j = 0; j < kQkPerLane; ++j) {
        qk[j] = static_cast<float>(q_row[lane + j * kSgSize]) * q_mult;
      }
#pragma unroll
      for (int j = 0; j < kVPerLane; ++j) {
        qv[j] = static_cast<float>(qv_row[lane + j * kSgSize]) * q_mult;
      }
    }
    float m = -INFINITY;
    float l = 0.f;

    const int32_t* bt = block_table_ + b * bt_s0_;
    for (int n0 = 0; n0 < n_keys; n0 += kTileN) {
      // Stage KV rows n0 .. n0 + kTileN - 1; sub-group sg_id loads one row.
      {
        const int n = n0 + sg_id;
        bool valid = n < n_keys;
        int64_t blk = -1;
        if (valid) {
          blk = bt[n / page_size_];
          valid = blk >= 0;
        }
        if (valid) {
          const int64_t off = n % page_size_;
          const T* v_row = v_ + blk * v_s0_ + off * v_s1_ + g * v_s2_;
          const T* k_row = k_ + blk * k_s0_ + off * k_s1_ + g * k_s2_;
          T* dst = &slm_kv_[sg_id * kRow];
          for (int d = lane; d < DV; d += kSgSize) {
            dst[d] = v_row[d];
          }
          for (int d = lane; d < DQK; d += kSgSize) {
            dst[DV + d] = k_row[d];
          }
        }
        if (lane == 0) {
          slm_valid_[sg_id] = valid ? 1 : 0;
        }
      }
      sycl::group_barrier(item.get_group());

      if (head_active) {
        float s[kTileN];
        float tile_max = -INFINITY;
#pragma unroll
        for (int r = 0; r < kTileN; ++r) {
          s[r] = -INFINITY;
          // slm_valid_ is uniform across the sub-group, so the group
          // reduction below is reached by all lanes or none.
          if (slm_valid_[r]) {
            const T* row = &slm_kv_[r * kRow];
            float partial = 0.f;
#pragma unroll
            for (int j = 0; j < kVPerLane; ++j) {
              partial += qv[j] * static_cast<float>(row[lane + j * kSgSize]);
            }
#pragma unroll
            for (int j = 0; j < kQkPerLane; ++j) {
              partial +=
                  qk[j] * static_cast<float>(row[DV + lane + j * kSgSize]);
            }
            s[r] = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
            tile_max = sycl::fmax(tile_max, s[r]);
          }
        }
        if (tile_max != -INFINITY) {
          const float m_new = sycl::fmax(m, tile_max);
          const float alpha = sycl::exp2(m - m_new);  // 0 when m == -inf
          l *= alpha;
#pragma unroll
          for (int j = 0; j < kVPerLane; ++j) {
            acc[j] *= alpha;
          }
#pragma unroll
          for (int r = 0; r < kTileN; ++r) {
            if (s[r] != -INFINITY) {
              const float p = sycl::exp2(s[r] - m_new);
              l += p;
              const T* row = &slm_kv_[r * kRow];
#pragma unroll
              for (int j = 0; j < kVPerLane; ++j) {
                acc[j] += p * static_cast<float>(row[lane + j * kSgSize]);
              }
            }
          }
          m = m_new;
        }
      }
      sycl::group_barrier(item.get_group());
    }

    if (!head_active) {
      return;
    }
    const float inv_l = l > 0.f ? 1.f / l : 0.f;
    T* o_row = out_ + t * o_s0_ + h * o_s1_;
#pragma unroll
    for (int j = 0; j < kVPerLane; ++j) {
      o_row[lane + j * kSgSize] = static_cast<T>(acc[j] * inv_l);
    }
    if (lse_ != nullptr && lane == 0) {
      // Same convention as FA (non-split): +inf for rows without keys.
      lse_[h * total_q_ + t] =
          (l == 0.f || l != l) ? INFINITY : m * kLn2 + sycl::log(l);
    }
  }

 private:
  const T* q_;
  const T* q_v_;
  const T* k_;
  const T* v_;
  T* out_;
  float* lse_;
  const int32_t* cu_seqlens_q_;
  const int32_t* seqused_k_;
  const int32_t* block_table_;
  sycl::local_accessor<T, 1> slm_kv_;
  sycl::local_accessor<int32_t, 1> slm_valid_;
  int64_t q_s0_;
  int64_t q_s1_;
  int64_t qv_s0_;
  int64_t qv_s1_;
  int64_t k_s0_;
  int64_t k_s1_;
  int64_t k_s2_;
  int64_t v_s0_;
  int64_t v_s1_;
  int64_t v_s2_;
  int64_t o_s0_;
  int64_t o_s1_;
  int64_t bt_s0_;
  int32_t batch_size_;
  int32_t total_q_;
  int32_t page_size_;
  int32_t num_heads_kv_;
  int32_t heads_per_kv_;
  int32_t num_head_blocks_;
  float softmax_scale_;
  bool is_causal_;
};

template <typename scalar_t, int DQK, int DV>
void launch(
    const at::Tensor& q,
    const at::Tensor& q_v,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    float* lse_ptr,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqused_k,
    const at::Tensor& block_table,
    float softmax_scale,
    bool is_causal) {
  using T = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  using Kernel = MlaQvVarlenKernel<T, DQK, DV>;

  const int32_t total_q = q.size(0);
  const int32_t num_heads = q.size(1);
  const int32_t num_heads_kv = k.size(2);
  const int32_t heads_per_kv = num_heads / num_heads_kv;
  const int32_t num_head_blocks =
      (heads_per_kv + kHeadsPerWg - 1) / kHeadsPerWg;
  const int32_t batch_size = seqused_k.size(0);
  const int32_t page_size = k.size(1);
  const int64_t num_groups =
      static_cast<int64_t>(total_q) * num_heads_kv * num_head_blocks;

  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<T, 1> slm_kv(sycl::range<1>(kTileN * (DV + DQK)), cgh);
    sycl::local_accessor<int32_t, 1> slm_valid(sycl::range<1>(kTileN), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(num_groups * kWgSize, kWgSize),
        Kernel(
            reinterpret_cast<const T*>(q.data_ptr()),
            reinterpret_cast<const T*>(q_v.data_ptr()),
            reinterpret_cast<const T*>(k.data_ptr()),
            reinterpret_cast<const T*>(v.data_ptr()),
            reinterpret_cast<T*>(out.data_ptr()),
            lse_ptr,
            cu_seqlens_q.data_ptr<int32_t>(),
            seqused_k.data_ptr<int32_t>(),
            block_table.data_ptr<int32_t>(),
            slm_kv,
            slm_valid,
            q.stride(0),
            q.stride(1),
            q_v.stride(0),
            q_v.stride(1),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            block_table.stride(0),
            batch_size,
            total_q,
            page_size,
            num_heads_kv,
            heads_per_kv,
            num_head_blocks,
            softmax_scale,
            is_causal));
  });
}

void check_int32_device(
    const at::Tensor& t, const at::Tensor& ref, const char* name) {
  TORCH_CHECK(t.scalar_type() == at::kInt, name, " must be int32");
  TORCH_CHECK(t.device() == ref.device(), name, " must be on q's device");
}

}  // namespace

std::vector<at::Tensor> mla_qv_varlen_fwd(
    const at::Tensor& q,
    const at::Tensor& q_v,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& out_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqused_k,
    const at::Tensor& block_table,
    double softmax_scale,
    bool is_causal,
    bool return_softmax_lse) {
  const auto dtype = q.scalar_type();
  TORCH_CHECK(
      dtype == at::kBFloat16 || dtype == at::kHalf,
      "mla_qv_varlen_fwd only supports bf16 and fp16");
  TORCH_CHECK(q.is_xpu(), "q must be on XPU");
  for (const at::Tensor* t : {&q_v, &k, &v}) {
    TORCH_CHECK(t->scalar_type() == dtype, "q, q_v, k, v must share dtype");
    TORCH_CHECK(t->device() == q.device(), "q, q_v, k, v must share device");
  }
  TORCH_CHECK(q.dim() == 3, "q must be [total_q, num_heads, d_qk]");
  TORCH_CHECK(q_v.dim() == 3, "q_v must be [total_q, num_heads, d_v]");
  TORCH_CHECK(
      k.dim() == 4, "k must be [num_blocks, page_size, num_heads_kv, d_qk]");
  TORCH_CHECK(
      v.dim() == 4, "v must be [num_blocks, page_size, num_heads_kv, d_v]");

  const int64_t total_q = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t d_qk = q.size(2);
  const int64_t d_v = q_v.size(2);
  TORCH_CHECK(
      q_v.size(0) == total_q && q_v.size(1) == num_heads,
      "q_v must match q in tokens and heads");
  TORCH_CHECK(
      k.size(0) == v.size(0) && k.size(1) == v.size(1) &&
          k.size(2) == v.size(2),
      "k and v must share num_blocks, page_size and num_heads_kv");
  TORCH_CHECK(k.size(3) == d_qk, "k head size must match q");
  TORCH_CHECK(v.size(3) == d_v, "v head size must match q_v");
  const int64_t num_heads_kv = k.size(2);
  TORCH_CHECK(
      num_heads_kv > 0 && num_heads % num_heads_kv == 0,
      "num_heads must be a multiple of num_heads_kv");
  TORCH_CHECK(k.size(1) > 0, "page_size must be positive");
  for (const at::Tensor* t : {&q, &q_v, &k, &v}) {
    TORCH_CHECK(t->stride(-1) == 1, "last dim of q, q_v, k, v must be dense");
  }

  check_int32_device(cu_seqlens_q, q, "cu_seqlens_q");
  check_int32_device(seqused_k, q, "seqused_k");
  check_int32_device(block_table, q, "block_table");
  TORCH_CHECK(
      cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous(),
      "cu_seqlens_q must be a contiguous 1-D tensor");
  TORCH_CHECK(
      seqused_k.dim() == 1 && seqused_k.is_contiguous(),
      "seqused_k must be a contiguous 1-D tensor");
  const int64_t batch_size = seqused_k.size(0);
  TORCH_CHECK(batch_size > 0, "batch must not be empty");
  TORCH_CHECK(
      cu_seqlens_q.size(0) == batch_size + 1,
      "cu_seqlens_q must have batch + 1 entries");
  TORCH_CHECK(
      block_table.dim() == 2 && block_table.size(0) == batch_size &&
          block_table.stride(1) == 1,
      "block_table must be [batch, max_blocks] with dense rows");

  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.scalar_type() == dtype, "out must match q dtype");
    TORCH_CHECK(out.device() == q.device(), "out must be on q's device");
    TORCH_CHECK(
        out.dim() == 3 && out.size(0) == total_q && out.size(1) == num_heads &&
            out.size(2) == d_v,
        "out must be [total_q, num_heads, d_v]");
    TORCH_CHECK(out.stride(-1) == 1, "last dim of out must be dense");
  } else {
    out = torch::empty({total_q, num_heads, d_v}, q.options());
  }
  at::Tensor softmax_lse;
  if (return_softmax_lse) {
    softmax_lse =
        torch::empty({num_heads, total_q}, q.options().dtype(at::kFloat));
  }
  if (total_q == 0) {
    return {out, softmax_lse};
  }

  const at::DeviceGuard device_guard(q.device());
  float* lse_ptr = return_softmax_lse ? softmax_lse.data_ptr<float>() : nullptr;
  const float scale = static_cast<float>(softmax_scale);

  if (d_qk == 64 && d_v == 512) {
    if (dtype == at::kBFloat16) {
      launch<c10::BFloat16, 64, 512>(
          q,
          q_v,
          k,
          v,
          out,
          lse_ptr,
          cu_seqlens_q,
          seqused_k,
          block_table,
          scale,
          is_causal);
    } else {
      launch<c10::Half, 64, 512>(
          q,
          q_v,
          k,
          v,
          out,
          lse_ptr,
          cu_seqlens_q,
          seqused_k,
          block_table,
          scale,
          is_causal);
    }
  } else {
    TORCH_CHECK(
        false,
        "mla_qv_varlen_fwd: unsupported head sizes d_qk=",
        d_qk,
        ", d_v=",
        d_v,
        " (supported: d_qk=64, d_v=512)");
  }
  return {out, softmax_lse};
}

}  // namespace vllm::mla_qv
