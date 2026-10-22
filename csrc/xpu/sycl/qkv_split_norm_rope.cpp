#include <sycl/sycl.hpp>

#include <cstdint>
#include <ATen/DeviceGuard.h>
#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {

// Fused QKV post-processing for gated / non-gated attention:
//
//   split(qkv) -> per-head RMSNorm of q and k -> (M)RoPE on q and k
//   [-> copy of the raw output gate]
//
// Semantics follow the unfused vLLM path:
//   * RMSNorm (vllm.ir.ops.rms_norm with an fp32 weight):
//       y = fp16(x * rsqrt(mean(x^2) + eps) * (w + weight_offset))
//     where the multiply is done in fp32 (weight_offset = 1 for Gemma-style
//     "(1 + w)" norms such as Qwen3-Next).
//   * NeoX RoPE on the first rotary_dim channels of each head, cos/sin read
//     from cos_sin_cache[pos, :] = [cos(rotary_dim / 2) | sin(rotary_dim / 2)].
//   * positions is [T] (plain RoPE) or [3, T] (MRoPE). For MRoPE the position
//     row of rotary pair p is chosen per mrope_section, interleaved
//     (THWTHW..., Qwen3-VL / Qwen3.5) or chunked (TTT..HHH..WWW).
//
// qkv row layout: gated   -> [Hq x (q[D], gate[D]), Hk x k[D], Hk x v[D]]
//                 no gate -> [Hq x q[D], Hk x k[D], Hk x v[D]]
// v is not touched (callers take it as a view of qkv).
//
// One sub-group of 16 lanes handles one (token, head); each lane owns D / 16
// contiguous channels. The RoPE partner of channel i < rotary_dim / 2 is
// channel i + rotary_dim / 2, which lives in lane
// lane + (rotary_dim / 2) / (D / 16) at the same register slot.
constexpr int kQkvSgSize = 16;

template <typename T, int D>
class qkv_split_norm_rope_kernel {
  static constexpr int EPL = D / kQkvSgSize;

 public:
  qkv_split_norm_rope_kernel(
      const T* qkv,
      int64_t qkv_row_stride,
      const int64_t* positions,
      int64_t pos_row_stride,
      bool is_mrope,
      const T* q_weight,
      const T* k_weight,
      const T* cos_sin_cache,
      T* q_out,
      T* k_out,
      T* gate_out,
      int num_q_heads,
      int num_kv_heads,
      int rotary_dim,
      float eps,
      float weight_offset,
      int mrope_s1,
      int mrope_s2,
      bool mrope_interleaved,
      int mrope_s0)
      : qkv(qkv),
        qkv_row_stride(qkv_row_stride),
        positions(positions),
        pos_row_stride(pos_row_stride),
        is_mrope(is_mrope),
        q_weight(q_weight),
        k_weight(k_weight),
        cos_sin_cache(cos_sin_cache),
        q_out(q_out),
        k_out(k_out),
        gate_out(gate_out),
        num_q_heads(num_q_heads),
        num_kv_heads(num_kv_heads),
        rotary_dim(rotary_dim),
        eps(eps),
        weight_offset(weight_offset),
        mrope_s1(mrope_s1),
        mrope_s2(mrope_s2),
        mrope_interleaved(mrope_interleaved),
        mrope_s0(mrope_s0) {}

  // Position row (0 = T, 1 = H, 2 = W) used for rotary pair p.
  inline int mrope_row(int p) const {
    if (!is_mrope) return 0;
    if (mrope_interleaved) {
      if (p % 3 == 1 && p < mrope_s1 * 3) return 1;
      if (p % 3 == 2 && p < mrope_s2 * 3) return 2;
      return 0;
    }
    if (p < mrope_s0) return 0;
    if (p < mrope_s0 + mrope_s1) return 1;
    return 2;
  }

  [[sycl::reqd_sub_group_size(kQkvSgSize)]] void
  operator()(sycl::nd_item<2> item) const {
    const int64_t t = item.get_group(0);
    const int head = item.get_group(1);
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const bool gated = gate_out != nullptr;
    const bool is_q = head < num_q_heads;
    const int q_stride = gated ? 2 * D : D;

    const T* row = qkv + t * qkv_row_stride;
    const T* src;
    const T* w;
    T* dst;
    if (is_q) {
      src = row + head * q_stride;
      w = q_weight;
      dst = q_out + (t * num_q_heads + head) * D;
    } else {
      const int kh = head - num_q_heads;
      src = row + num_q_heads * q_stride + kh * D;
      w = k_weight;
      dst = k_out + (t * num_kv_heads + kh) * D;
    }

    const int c0 = lane * EPL;
    float x[EPL];
    float ssq = 0.f;
#pragma unroll
    for (int e = 0; e < EPL; ++e) {
      x[e] = static_cast<float>(src[c0 + e]);
      ssq += x[e] * x[e];
    }
    ssq = sycl::reduce_over_group(sg, ssq, sycl::plus<float>());
    const float inv = sycl::rsqrt(ssq / D + eps);
#pragma unroll
    for (int e = 0; e < EPL; ++e) {
      const float wf = static_cast<float>(w[c0 + e]) + weight_offset;
      // Round to the activation dtype like the unfused norm output.
      x[e] = static_cast<float>(static_cast<T>(x[e] * inv * wf));
    }

    const int half = rotary_dim / 2;
    const int lane_shift = half / EPL;
    const bool in_rot = c0 < rotary_dim;
    const bool first_half = c0 < half;
    const int partner = first_half ? lane + lane_shift : lane - lane_shift;
    const int src_lane = in_rot ? partner : lane;
#pragma unroll
    for (int e = 0; e < EPL; ++e) {
      const float other = sycl::select_from_group(sg, x[e], src_lane);
      if (in_rot) {
        const int p = first_half ? c0 + e : c0 + e - half;
        const int64_t pos = positions[mrope_row(p) * pos_row_stride + t];
        const T* cs = cos_sin_cache + pos * rotary_dim;
        const float c = static_cast<float>(cs[p]);
        const float s = static_cast<float>(cs[half + p]);
        x[e] = first_half ? x[e] * c - other * s : x[e] * c + other * s;
      }
    }
#pragma unroll
    for (int e = 0; e < EPL; ++e) {
      dst[c0 + e] = static_cast<T>(x[e]);
    }

    if (is_q && gated) {
      const T* g = src + D;
      T* gdst = gate_out + (t * num_q_heads + head) * D;
#pragma unroll
      for (int e = 0; e < EPL; ++e) {
        gdst[c0 + e] = g[c0 + e];
      }
    }
  }

 private:
  const T* qkv;
  int64_t qkv_row_stride;
  const int64_t* positions;
  int64_t pos_row_stride;
  bool is_mrope;
  const T* q_weight;
  const T* k_weight;
  const T* cos_sin_cache;
  T* q_out;
  T* k_out;
  T* gate_out;
  int num_q_heads;
  int num_kv_heads;
  int rotary_dim;
  float eps;
  float weight_offset;
  int mrope_s1;
  int mrope_s2;
  bool mrope_interleaved;
  int mrope_s0;
};

template <typename scalar_t, int D>
void launch_qkv_split_norm_rope(
    const torch::Tensor& qkv,
    const torch::Tensor& positions,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos_sin_cache,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    const std::optional<torch::Tensor>& gate_out,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    double eps,
    double weight_offset,
    const std::vector<int64_t>& mrope_section,
    bool mrope_interleaved) {
  using T = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  const int64_t num_tokens = qkv.size(0);
  if (num_tokens == 0) return;
  const bool is_mrope = positions.dim() == 2;
  const int s0 = is_mrope ? mrope_section[0] : 0;
  const int s1 = is_mrope ? mrope_section[1] : 0;
  const int s2 = is_mrope ? mrope_section[2] : 0;
  T* gate_ptr = gate_out.has_value()
                    ? reinterpret_cast<T*>(gate_out->data_ptr<scalar_t>())
                    : nullptr;

  auto& queue = vllm::xpu::vllmGetQueue();
  sycl::range<2> local(1, kQkvSgSize);
  sycl::range<2> global(num_tokens, (num_q_heads + num_kv_heads) * kQkvSgSize);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(global, local),
        qkv_split_norm_rope_kernel<T, D>(
            reinterpret_cast<const T*>(qkv.data_ptr<scalar_t>()),
            qkv.stride(0),
            positions.data_ptr<int64_t>(),
            is_mrope ? positions.stride(0) : 0,
            is_mrope,
            reinterpret_cast<const T*>(q_weight.data_ptr<scalar_t>()),
            reinterpret_cast<const T*>(k_weight.data_ptr<scalar_t>()),
            reinterpret_cast<const T*>(cos_sin_cache.data_ptr<scalar_t>()),
            reinterpret_cast<T*>(q_out.data_ptr<scalar_t>()),
            reinterpret_cast<T*>(k_out.data_ptr<scalar_t>()),
            gate_ptr,
            static_cast<int>(num_q_heads),
            static_cast<int>(num_kv_heads),
            static_cast<int>(rotary_dim),
            static_cast<float>(eps),
            static_cast<float>(weight_offset),
            s1,
            s2,
            mrope_interleaved,
            s0));
  });
}

}  // namespace vllm

void qkv_split_norm_rope(
    const torch::Tensor& qkv,                      // [T, W]
    const torch::Tensor& positions,                // int64 [T] or [3, T]
    const torch::Tensor& q_weight,                 // [D]
    const torch::Tensor& k_weight,                 // [D]
    const torch::Tensor& cos_sin_cache,            // [max_pos, rotary_dim]
    torch::Tensor& q_out,                          // [T, Hq * D]
    torch::Tensor& k_out,                          // [T, Hk * D]
    const std::optional<torch::Tensor>& gate_out,  // [T, Hq * D] if gated
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    double eps,
    double weight_offset,
    std::vector<int64_t> mrope_section,
    bool mrope_interleaved) {
  const at::DeviceGuard device_guard(qkv.device());
  const auto dtype = qkv.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16,
      "qkv_split_norm_rope: qkv must be fp16 or bf16");
  for (const torch::Tensor* t :
       {&q_weight,
        &k_weight,
        &cos_sin_cache,
        static_cast<const torch::Tensor*>(&q_out),
        static_cast<const torch::Tensor*>(&k_out)})
    TORCH_CHECK(
        t->scalar_type() == dtype,
        "qkv_split_norm_rope: all tensors must share the qkv dtype");
  TORCH_CHECK(
      head_dim == 128 || head_dim == 256,
      "qkv_split_norm_rope: head_dim must be 128 or 256");
  const int64_t epl = head_dim / vllm::kQkvSgSize;
  TORCH_CHECK(
      rotary_dim > 0 && rotary_dim <= head_dim && rotary_dim % 2 == 0 &&
          (rotary_dim / 2) % epl == 0,
      "qkv_split_norm_rope: unsupported rotary_dim");
  TORCH_CHECK(
      qkv.dim() == 2 && qkv.stride(1) == 1,
      "qkv_split_norm_rope: qkv must be 2D with unit inner stride");
  const bool gated = gate_out.has_value();
  const int64_t T = qkv.size(0);
  TORCH_CHECK(
      qkv.size(1) == (gated ? 2 : 1) * num_q_heads * head_dim +
                         2 * num_kv_heads * head_dim,
      "qkv_split_norm_rope: qkv width does not match the head layout");
  TORCH_CHECK(
      positions.scalar_type() == at::kLong,
      "qkv_split_norm_rope: positions must be int64");
  if (positions.dim() == 1) {
    TORCH_CHECK(positions.size(0) == T && positions.stride(0) == 1);
  } else {
    TORCH_CHECK(
        positions.dim() == 2 && positions.size(0) == 3 &&
            positions.size(1) == T && positions.stride(1) == 1,
        "qkv_split_norm_rope: positions must be [T] or [3, T]");
    TORCH_CHECK(
        mrope_section.size() == 3 &&
            mrope_section[0] + mrope_section[1] + mrope_section[2] ==
                rotary_dim / 2,
        "qkv_split_norm_rope: mrope_section must have 3 entries summing to "
        "rotary_dim / 2");
  }
  TORCH_CHECK(
      q_weight.numel() == head_dim && k_weight.numel() == head_dim &&
      q_weight.is_contiguous() && k_weight.is_contiguous());
  TORCH_CHECK(
      cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == rotary_dim &&
          cos_sin_cache.is_contiguous(),
      "qkv_split_norm_rope: cos_sin_cache must be [max_pos, rotary_dim]");
  TORCH_CHECK(
      q_out.is_contiguous() && q_out.numel() == T * num_q_heads * head_dim);
  TORCH_CHECK(
      k_out.is_contiguous() && k_out.numel() == T * num_kv_heads * head_dim);
  if (gated) {
    TORCH_CHECK(
        gate_out->scalar_type() == dtype && gate_out->is_contiguous() &&
        gate_out->numel() == T * num_q_heads * head_dim);
  }

  VLLM_DISPATCH_HALF_TYPES(dtype, "qkv_split_norm_rope", [&] {
    if (head_dim == 256) {
      vllm::launch_qkv_split_norm_rope<scalar_t, 256>(
          qkv,
          positions,
          q_weight,
          k_weight,
          cos_sin_cache,
          q_out,
          k_out,
          gate_out,
          num_q_heads,
          num_kv_heads,
          rotary_dim,
          eps,
          weight_offset,
          mrope_section,
          mrope_interleaved);
    } else {
      vllm::launch_qkv_split_norm_rope<scalar_t, 128>(
          qkv,
          positions,
          q_weight,
          k_weight,
          cos_sin_cache,
          q_out,
          k_out,
          gate_out,
          num_q_heads,
          num_kv_heads,
          rotary_dim,
          eps,
          weight_offset,
          mrope_section,
          mrope_interleaved);
    }
  });
}
