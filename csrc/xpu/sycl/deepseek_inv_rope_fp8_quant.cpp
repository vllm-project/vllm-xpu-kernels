#include <sycl/sycl.hpp>
#include "utils.h"
#include "inv_rope_ns.h"
#include <cstdint>
#include <type_traits>
#include <torch/all.h>

namespace vllm {
inline namespace VLLM_INV_ROPE_IMPL_NS {

namespace syclex = sycl::ext::oneapi::experimental;
// Sub-group block messages: lane l of a PER_LOAD-element load owns index
// i*SG_SIZE + l, i.e. the same strided lane -> element mapping as a manual
// `ptr[lane + i*SG_SIZE]`, but emitted as one wide block message instead of
// PER_LOAD separate per-lane messages.
static constexpr auto kStriped =
    syclex::properties{syclex::data_placement_striped};

using bf16_t = sycl::ext::oneapi::bfloat16;
using fp8_t = uint8_t;

static constexpr int QUANT_GROUP_SIZE = 128;
static constexpr int MXFP8_GROUP_SIZE = 32;
static constexpr float FP8_E4M3_MAX = 448.0f;

// Ceil to power-of-2; also returns biased exponent in *out_scale_exp.
inline float fast_ceil_pow2(float x, uint32_t& out_scale_exp) {
  uint32_t bits;
  __builtin_memcpy(&bits, &x, 4);
  if (bits & 0x7FFFFF) {
    bits = (bits + 0x800000) & 0xFF800000u;
  }
  out_scale_exp = bits >> 23;
  float result;
  __builtin_memcpy(&result, &bits, 4);
  return result;
}

// Same rounding, but for callers that only want the exponent: adding 2^23-1
// carries into the exponent iff the mantissa is non-zero, which is the ceil.
inline uint32_t ceil_pow2_exp(float x) {
  return (sycl::bit_cast<uint32_t>(x) + 0x7FFFFFu) >> 23;
}

// Divide by pow2 scale + convert to fp8_e4m3 in pure integer arithmetic.
inline uint8_t fused_scale_to_fp8(float val, uint32_t scale_exp) {
  uint32_t bits;
  __builtin_memcpy(&bits, &val, 4);
  uint32_t sign = bits >> 31;
  uint32_t v_exp = (bits >> 23) & 0xFFu;
  uint32_t v_mant = bits & 0x7FFFFFu;

  uint32_t mant3 = v_mant >> 20;
  uint32_t remainder = v_mant & 0xFFFFFu;
  // Round-to-nearest-even
  if (remainder > 0x80000u || (remainder == 0x80000u && (mant3 & 1u))) {
    mant3++;
  }
  uint32_t e4m3_exp = v_exp - scale_exp + 7u + (mant3 >> 3);
  mant3 &= 7u;
  uint32_t result = (e4m3_exp << 3) | mant3;
  result = (v_exp + 7u >= scale_exp) ? result : 0u;

  return static_cast<uint8_t>((sign << 7) | (result & 0x7Fu));
}

// `fcvt` with an e4m3 destination only exists on Xe3. _xpu_C is preprocessed
// once but AOT compiled for every device in AOT_DEVICES, so the macro can only
// be set for an Xe3-only build; a combined Xe2+Xe3 build has to take the
// software path. CMake defines VLLM_XPU_SYCL_FCVT_ENABLED accordingly, and
// VLLM_FP8_QUANT_DISABLE_ASM is an escape hatch for IGC versions that
// miscompile it.
#if defined(__SYCL_DEVICE_ONLY__) && defined(VLLM_XPU_SYCL_FCVT_ENABLED) && \
    !defined(VLLM_FP8_QUANT_DISABLE_ASM)
  #define VLLM_INV_ROPE_USE_FCVT 1
#endif

#ifdef VLLM_INV_ROPE_USE_FCVT
// Hardware half->e4m3 convert: one instruction for the whole sub-group, versus
// ~12 integer ops per element for the software path above.
inline uint8_t fcvt_half_e4m3(sycl::half h) {
  uint8_t dst;
  asm("{\n"
      ".decl IN_HF v_type=G type=HF num_elts=16 alias=<%1,0>\n"
      ".decl OUT_B v_type=G type=B  num_elts=16 alias=<%0,0>\n"
      "fcvt (M1_NM, 16) OUT_B(0,0)<1>  IN_HF(0,0)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(h));
  return dst;
}
#endif

// vISA fcvt has no fp32 source, so the hardware path is fp32->fp16->e4m3. For
// bf16-sourced values that is still a single rounding: 8 significant bits
// survive the power-of-two scale and fit fp16 exactly. Rope outputs are full
// fp32 and would round twice, so they take the software path instead.
inline uint8_t scale_to_fp8(float val, float inv_scale, uint32_t scale_exp,
                            bool from_bf16) {
#ifdef VLLM_INV_ROPE_USE_FCVT
  if (from_bf16) return fcvt_half_e4m3(static_cast<sycl::half>(val * inv_scale));
#endif
  (void)inv_scale;
  (void)from_bf16;
  return fused_scale_to_fp8(val, scale_exp);
}

// Fused inv RoPE + FP8 quant kernel. One sub-group (16 lanes) per (token,
// head), each lane holding 2 contiguous bf16 per register so that a 32-element
// quant block is exactly one register across the sub-group.
//
// MXFP8 selects 32-element blocks with raw E8M0 exponent scales; otherwise the
// legacy 128-element blocks with float32 power-of-two scales are produced.
template <bool MXFP8>
class deepseek_inv_rope_fp8_quant_kernel {
 public:
  static constexpr int SG_SIZE = 16;
  // Narrowest useful register: 4B per lane, so a load covers 32 elements. Both
  // scale formats are a whole number of registers, and 8B or 16B per lane
  // measured slower on both paths.
  using load_t = uint32_t;
  using store_t = uint16_t;
  static constexpr int ELEMS_PER_REG = 2;
  static constexpr int REGS_PER_LANE = 16;
  static constexpr int PER_LOAD = 4;
  static constexpr int NLOADS = REGS_PER_LANE / PER_LOAD;
  // Rope is elements 448..511, i.e. the last two registers; lane l owns rope
  // pair k * 16 + l of register FIRST_ROPE_REG + k.
  static constexpr int ROPE_REGS = 2;
  static constexpr int FIRST_ROPE_REG = REGS_PER_LANE - ROPE_REGS;
  static constexpr int CHUNKS_PER_HEAD = 4;
  static constexpr int MX_BLOCKS_PER_HEAD = 16;
  static constexpr int SCALES_PER_HEAD =
      MXFP8 ? MX_BLOCKS_PER_HEAD : CHUNKS_PER_HEAD;
  static constexpr int REGS_PER_BLOCK = REGS_PER_LANE / SCALES_PER_HEAD;
  // The 16 e8m0 bytes of a head, packed four to a word.
  static constexpr int MX_SCALE_WORDS = MX_BLOCKS_PER_HEAD / 4;

  using scale_t = std::conditional_t<MXFP8, uint8_t, float>;

  static_assert(
      !MXFP8 || MX_BLOCKS_PER_HEAD == 16,
      "MXFP8 scale store splits 16 exponent bytes across the two lane halves");

  // One butterfly step of a 16-way simultaneous max: each lane keeps the half
  // of the surviving value set it will own and ships the other half to its
  // xor-MASK partner. Four steps reduce all 16 block maxima in 8+4+2+1
  // shuffles, versus 16 four-level allreduces (64).
  template <int MASK>
  static void reduce_step(sycl::sub_group sg, int lane, float* v) {
#pragma unroll
    for (int i = 0; i < MASK; ++i) {
      const float keep = (lane & MASK) ? v[i + MASK] : v[i];
      const float send = (lane & MASK) ? v[i] : v[i + MASK];
      v[i] = sycl::fmax(keep, sycl::permute_group_by_xor(sg, send, MASK));
    }
  }

  deepseek_inv_rope_fp8_quant_kernel(
      const bf16_t* __restrict__ o,
      const int64_t* __restrict__ positions,
      const float* __restrict__ cos_sin_cache,
      fp8_t* __restrict__ fp8_out,
      scale_t* __restrict__ scale_out,
      int64_t num_tokens,
      int64_t num_heads,
      int64_t heads_per_group,
      int64_t head_dim,
      int64_t nope_dim,
      int64_t half_rope,
      int64_t o_stride_token,
      int64_t o_stride_head,
      int64_t cache_stride_pos,
      int64_t fp8_stride_group,
      int64_t fp8_stride_token,
      int64_t scale_stride_group,
      int64_t scale_stride_token)
      : o_(o),
        positions_(positions),
        cos_sin_cache_(cos_sin_cache),
        fp8_out_(fp8_out),
        scale_out_(scale_out),
        num_tokens_(num_tokens),
        num_heads_(num_heads),
        heads_per_group_(heads_per_group),
        head_dim_(head_dim),
        nope_dim_(nope_dim),
        half_rope_(half_rope),
        o_stride_token_(o_stride_token),
        o_stride_head_(o_stride_head),
        cache_stride_pos_(cache_stride_pos),
        fp8_stride_group_(fp8_stride_group),
        fp8_stride_token_(fp8_stride_token),
        scale_stride_group_(scale_stride_group),
        scale_stride_token_(scale_stride_token) {}

  [[sycl::reqd_sub_group_size(SG_SIZE)]]
  void operator()(sycl::nd_item<3> item) const {
    const int64_t token_idx = item.get_global_id(0);
    const int64_t group = item.get_global_id(1);
    const int64_t flat = item.get_global_id(2);
    const int64_t head_in_group = flat >> 4;
    const int lane = static_cast<int>(flat & (SG_SIZE - 1));
    const int64_t head_idx = group * heads_per_group_ + head_in_group;
    auto sg = item.get_sub_group();

    // positions -> cos_sin_cache is a serialized two-level load chain. Issue it
    // before the bulk loads so it overlaps with them instead of stalling the
    // sub-group right before the stores.
    const int64_t pos = positions_[token_idx];
    const float* cos_ptr = cos_sin_cache_ + pos * cache_stride_pos_;
    // Rope register FIRST_ROPE_REG + k of lane l holds pair k * 16 + l, which
    // is exactly a striped block load.
    float c2[ROPE_REGS], s2[ROPE_REGS];
    syclex::group_load(sg, cos_ptr, sycl::span<float, ROPE_REGS>{c2, ROPE_REGS},
                       kStriped);
    syclex::group_load(sg, cos_ptr + half_rope_,
                       sycl::span<float, ROPE_REGS>{s2, ROPE_REGS}, kStriped);

    const load_t* src = reinterpret_cast<const load_t*>(
        o_ + token_idx * o_stride_token_ + head_idx * o_stride_head_);
    store_t* dst = reinterpret_cast<store_t*>(
        fp8_out_ + group * fp8_stride_group_ + token_idx * fp8_stride_token_ +
        head_in_group * head_dim_);
    scale_t* scale_dst = scale_out_ + group * scale_stride_group_ +
                         token_idx * scale_stride_token_ +
                         head_in_group * SCALES_PER_HEAD;

    // Striped: raw[b*PER_LOAD + i] is load_t index b*PER_LOAD*16 + i*16 + lane,
    // i.e. the same lane -> element mapping as a manual ptr[lane + r*16], but
    // issued as NLOADS wide block messages instead of REGS_PER_LANE narrow
    // ones.
    load_t raw[REGS_PER_LANE];
#pragma unroll
    for (int b = 0; b < NLOADS; ++b)
      syclex::group_load(
          sg, src + b * PER_LOAD * SG_SIZE,
          sycl::span<load_t, PER_LOAD>{raw + b * PER_LOAD, PER_LOAD}, kStriped);

    float vals[REGS_PER_LANE][ELEMS_PER_REG];
#pragma unroll
    for (int r = 0; r < REGS_PER_LANE; ++r)
#pragma unroll
      for (int j = 0; j < ELEMS_PER_REG; ++j)
        vals[r][j] = static_cast<float>(
            sycl::bit_cast<bf16_t>(static_cast<uint16_t>(raw[r] >> (16 * j))));

#pragma unroll
    for (int k = 0; k < ROPE_REGS; ++k) {
      const int r = FIRST_ROPE_REG + k;
      const float x_even = vals[r][0];
      const float x_odd = vals[r][1];
      vals[r][0] = x_even * c2[k] + x_odd * s2[k];
      vals[r][1] = x_odd * c2[k] - x_even * s2[k];
    }

    // Per-block: absmax reduce -> pow2 scale -> fp8 quantize. A block is one
    // register for MXFP8 and four for the 128-element float32 format.
    store_t out[REGS_PER_LANE];
    sycl::vec<float, CHUNKS_PER_HEAD> scales;
    uint32_t scale_pack[MX_SCALE_WORDS] = {};

    if constexpr (MXFP8) {
      // A block is a single register, so its absmax is a whole-sub-group
      // reduce; doing 16 of those independently costs 64 shuffles. The
      // butterfly instead leaves lane l owning block l's absmax after 15,
      // which also means the exponent is computed once per lane rather than 16
      // times.
      float v[REGS_PER_LANE];
#pragma unroll
      for (int r = 0; r < REGS_PER_LANE; ++r)
        v[r] = sycl::fmax(sycl::fabs(vals[r][0]), sycl::fabs(vals[r][1]));
      reduce_step<8>(sg, lane, v);
      reduce_step<4>(sg, lane, v);
      reduce_step<2>(sg, lane, v);
      reduce_step<1>(sg, lane, v);

      // Only the exponent is stored, so the scale itself is never materialised.
      const uint32_t e =
          ceil_pow2_exp(sycl::fmax(v[0], 1e-10f) * (1.0f / FP8_E4M3_MAX));
      // Broadcast the 16 exponents back as packed bytes: two OR steps give
      // lanes 4g..4g+3 word g, two exchanges give every lane all four words.
      uint32_t w = e << (8 * (lane & 3));
      w |= sycl::permute_group_by_xor(sg, w, 1);
      w |= sycl::permute_group_by_xor(sg, w, 2);
      const uint32_t w4 = sycl::permute_group_by_xor(sg, w, 4);
      const uint32_t lo = (lane & 4) ? w4 : w;
      const uint32_t hi = (lane & 4) ? w : w4;
      const uint32_t lo8 = sycl::permute_group_by_xor(sg, lo, SG_SIZE / 2);
      const uint32_t hi8 = sycl::permute_group_by_xor(sg, hi, SG_SIZE / 2);
      scale_pack[0] = (lane & 8) ? lo8 : lo;
      scale_pack[1] = (lane & 8) ? hi8 : hi;
      scale_pack[2] = (lane & 8) ? lo : lo8;
      scale_pack[3] = (lane & 8) ? hi : hi8;
    }

#pragma unroll
    for (int blk = 0; blk < SCALES_PER_HEAD; ++blk) {
      uint32_t scale_exp;
      float inv_scale;
      if constexpr (MXFP8) {
        scale_exp = (scale_pack[blk / 4] >> (8 * (blk % 4))) & 0xFFu;
        // The scale is a power of two, so its reciprocal is just the negated
        // exponent: 2^(e-127) -> 2^(127-e) is bias 254 - e.
        inv_scale = sycl::bit_cast<float>((254u - scale_exp) << 23);
      } else {
        float local_absmax = 0.0f;
#pragma unroll
        for (int i = 0; i < REGS_PER_BLOCK; ++i)
#pragma unroll
          for (int j = 0; j < ELEMS_PER_REG; ++j)
            local_absmax = sycl::fmax(
                local_absmax, sycl::fabs(vals[REGS_PER_BLOCK * blk + i][j]));

        const float block_absmax =
            sycl::reduce_over_group(sg, local_absmax, sycl::maximum<float>());
        scales[blk] = fast_ceil_pow2(
            sycl::fmax(block_absmax, 1e-10f) * (1.0f / FP8_E4M3_MAX), scale_exp);
        inv_scale = 1.0f / scales[blk];
      }

#pragma unroll
      for (int i = 0; i < REGS_PER_BLOCK; ++i) {
        // The rope registers need the software convert; every other index is
        // untouched bf16 and can take the hardware one.
        const int r = REGS_PER_BLOCK * blk + i;
        uint32_t packed = 0;
#pragma unroll
        for (int j = 0; j < ELEMS_PER_REG; ++j)
          packed |= static_cast<uint32_t>(scale_to_fp8(
                        vals[r][j], inv_scale, scale_exp, r < FIRST_ROPE_REG))
                    << (8 * j);
        out[r] = static_cast<store_t>(packed);
      }
    }

    if constexpr (!MXFP8) {
      if (lane == 0)
        *reinterpret_cast<sycl::vec<float, CHUNKS_PER_HEAD>*>(scale_dst) =
            scales;
    }

#pragma unroll
    for (int b = 0; b < NLOADS; ++b)
      syclex::group_store(
          sg, sycl::span<const store_t, PER_LOAD>{out + b * PER_LOAD, PER_LOAD},
          dst + b * PER_LOAD * SG_SIZE, kStriped);

    if constexpr (MXFP8) {
      if ((lane & (SG_SIZE / 2 - 1)) == 0)
        *reinterpret_cast<sycl::vec<uint32_t, 2>*>(
            scale_dst + (lane ? SCALES_PER_HEAD / 2 : 0)) =
            sycl::vec<uint32_t, 2>{scale_pack[lane ? 2 : 0],
                                   scale_pack[lane ? 3 : 1]};
    }
  }

 private:
  const bf16_t* __restrict__ o_;
  const int64_t* __restrict__ positions_;
  const float* __restrict__ cos_sin_cache_;
  fp8_t* __restrict__ fp8_out_;
  scale_t* __restrict__ scale_out_;
  int64_t num_tokens_, num_heads_;
  int64_t heads_per_group_, head_dim_, nope_dim_, half_rope_;
  int64_t o_stride_token_, o_stride_head_;
  int64_t cache_stride_pos_;
  int64_t fp8_stride_group_, fp8_stride_token_;
  int64_t scale_stride_group_, scale_stride_token_;
};

}  // namespace VLLM_INV_ROPE_IMPL_NS
}  // namespace vllm

namespace VLLM_INV_ROPE_NS {

// PyTorch wrapper: inv RoPE + FP8 quant.
// Returns (fp8_out [G,T,hpg*D], scale_out [G,T,hpg*blocks]) where blocks is
// D/128 with float32 scales, or D/32 with float8_e8m0fnu scales for MXFP8.
std::tuple<torch::Tensor, torch::Tensor> deepseek_inv_rope_fp8_quant(
    const torch::Tensor& attn_output,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    int64_t n_groups,
    int64_t heads_per_group,
    int64_t nope_dim,
    int64_t rope_dim,
    bool use_mxfp8) {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  using fp8_t = uint8_t;
  constexpr int SG_SIZE =
      vllm::deepseek_inv_rope_fp8_quant_kernel<false>::SG_SIZE;
  constexpr int QUANT_GROUP_SIZE = vllm::QUANT_GROUP_SIZE;

  const auto& o = attn_output;
  TORCH_CHECK(o.dim() == 3, "o must be 3D [num_tokens, num_heads, head_dim]");
  TORCH_CHECK(o.scalar_type() == torch::kBFloat16, "o must be bfloat16");
  TORCH_CHECK(o.is_contiguous(), "o must be contiguous");
  TORCH_CHECK(
      positions.scalar_type() == torch::kLong, "positions must be int64");
  TORCH_CHECK(
      cos_sin_cache.scalar_type() == torch::kFloat32,
      "cos_sin_cache must be float32");
  TORCH_CHECK(
      cos_sin_cache.stride(-1) == 1,
      "cos_sin_cache must have contiguous last dim");

  const int64_t num_tokens = o.size(0);
  const int64_t num_heads = o.size(1);
  const int64_t head_dim = o.size(2);

  TORCH_CHECK(
      positions.size(0) == num_tokens,
      "positions length must equal num_tokens");

  TORCH_CHECK(
      num_heads == n_groups * heads_per_group,
      "num_heads must equal n_groups * heads_per_group");
  TORCH_CHECK(
      head_dim == nope_dim + rope_dim,
      "head_dim must equal nope_dim + rope_dim");
  TORCH_CHECK(rope_dim % 2 == 0, "rope_dim must be even");
  TORCH_CHECK(
      head_dim % QUANT_GROUP_SIZE == 0,
      "head_dim must be divisible by QUANT_GROUP_SIZE (128)");
  TORCH_CHECK(
      cos_sin_cache.size(-1) == rope_dim,
      "cos_sin_cache last dim must equal rope_dim");
  TORCH_CHECK(
      head_dim == 512 && nope_dim == 448 && rope_dim == 64,
      "Currently only supports head_dim=512, nope_dim=448, rope_dim=64");

  const int64_t half_rope = rope_dim / 2;
  const int64_t d = heads_per_group * head_dim;
  const int64_t blocks_per_head =
      use_mxfp8 ? head_dim / vllm::MXFP8_GROUP_SIZE
                : head_dim / QUANT_GROUP_SIZE;
  const int64_t num_scale_blocks = heads_per_group * blocks_per_head;

  auto fp8_out = torch::empty(
      {n_groups, num_tokens, d}, o.options().dtype(torch::kFloat8_e4m3fn));

  auto scale_out = torch::empty(
      {n_groups, num_tokens, num_scale_blocks},
      o.options().dtype(
          use_mxfp8 ? torch::kFloat8_e8m0fnu : torch::kFloat32));

  if (num_tokens == 0) {
    return {fp8_out, scale_out};
  }

  const int64_t o_stride_token = o.stride(0);
  const int64_t o_stride_head = o.stride(1);
  const int64_t cache_stride_pos = cos_sin_cache.stride(0);
  const int64_t fp8_stride_group = fp8_out.stride(0);
  const int64_t fp8_stride_token = fp8_out.stride(1);
  const int64_t scale_stride_group = scale_out.stride(0);
  const int64_t scale_stride_token = scale_out.stride(1);

  // The group / head_in_group split is encoded in the grid so the kernel never
  // has to do a 64-bit integer div/mod.
  int64_t hpg_per_wg = 1;
  for (int64_t candidate : {16, 8, 4, 2}) {
    if (heads_per_group % candidate == 0) {
      hpg_per_wg = candidate;
      break;
    }
  }

  sycl::range<3> local(1, 1, hpg_per_wg * SG_SIZE);
  sycl::range<3> global(num_tokens, n_groups, heads_per_group * SG_SIZE);

  auto& q = vllm::xpu::vllmGetQueue();
  auto launch = [&](auto mxfp8_tag) {
    using kernel_t =
        vllm::deepseek_inv_rope_fp8_quant_kernel<decltype(mxfp8_tag)::value>;
    using scale_ptr_t = typename kernel_t::scale_t;
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(global, local),
          kernel_t(
              reinterpret_cast<const bf16_t*>(o.data_ptr()),
              reinterpret_cast<const int64_t*>(positions.data_ptr()),
              reinterpret_cast<const float*>(cos_sin_cache.data_ptr()),
              reinterpret_cast<fp8_t*>(fp8_out.data_ptr()),
              reinterpret_cast<scale_ptr_t*>(scale_out.data_ptr()),
              num_tokens,
              num_heads,
              heads_per_group,
              head_dim,
              nope_dim,
              half_rope,
              o_stride_token,
              o_stride_head,
              cache_stride_pos,
              fp8_stride_group,
              fp8_stride_token,
              scale_stride_group,
              scale_stride_token));
    });
  };

  if (use_mxfp8) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }

  return {fp8_out, scale_out};
}

}  // namespace VLLM_INV_ROPE_NS
