#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>

#if __has_include(<sycl/ext/intel/experimental/grf_size_properties.hpp>)
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#define MOE_GATHER_HAS_GRF_PROP 1
#endif

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

constexpr int kSubGroupSize = 32;
constexpr int kWorkGroupSize = 256;
constexpr int kMaxRowBytesPerItem = 64;
constexpr int kMinRowBytesPerItem = 16;
// Fewest work-items a row is ever split across; at this width a work-group
// holds four sibling rows instead of one.
constexpr int kMinLanesPerRow = 64;
// Below this many rows the dispatch cannot fill the device and is dominated by
// launch latency rather than by the gather itself.
constexpr int kSaturationTokens = 256;
// Up to this many rows the gather still cannot hide a narrow per-item access,
// so the row is covered in as few passes as the vector width allows.
constexpr int kMultiTokenGroupMaxTokens = 2048;
// Per-rank expert counts: below the first, expert parallelism has thinned the
// routing tables enough for the sparse kernel to pay; above the second, the
// tables are dense and every row feeds the accumulator.
constexpr int kShardedExpertsMax = 48;
constexpr int kDenseExpertsMin = 80;
// Row width bands. A narrow row cannot keep a work-group busy however it is
// sliced; a wide-block row is one a work-group covers in a single pass of the
// widest access; past the very-wide mark the widest access stops paying off.
constexpr int kNarrowRowBytes = 3072;
constexpr int kMidRowBytes = 2048;
constexpr int kWideRowBytes = 5120;
constexpr int kWideBlockRowBytes = 4096;
constexpr int kVeryWideRowBytes = 10240;
// A long routing table keeps more accumulators live, which is where the large
// register file earns its reduced occupancy.
constexpr int kWideAccumTopk = 8;

#ifndef MOE_GATHER_GRF
  #define MOE_GATHER_GRF 256
#endif

template <typename T>
struct Cvt;

template <>
struct Cvt<sycl::ext::oneapi::bfloat16> {
  using storage_t = uint16_t;
  using element_t = sycl::ext::oneapi::bfloat16;

  static float to(storage_t value) {
    return static_cast<float>(sycl::bit_cast<element_t>(value));
  }
  static storage_t from(float value) {
    return sycl::bit_cast<storage_t>(static_cast<element_t>(value));
  }
};

template <>
struct Cvt<sycl::half> {
  using storage_t = uint16_t;

  static float to(storage_t value) {
    return static_cast<float>(sycl::bit_cast<sycl::half>(value));
  }
  static storage_t from(float value) {
    return sycl::bit_cast<storage_t>(static_cast<sycl::half>(value));
  }
};

template <>
struct Cvt<float> {
  using storage_t = float;

  static float to(storage_t value) { return value; }
  static storage_t from(float value) { return value; }
};

template <typename T, int VEC, int BLOCK, int LANES>
struct DirectRowOps {
  using storage_t = typename Cvt<T>::storage_t;
  using vec_t = sycl::vec<storage_t, VEC>;

  struct Geometry {
    int num_vecs;
    int num_blocks;
    int tail_first;
  };

  static Geometry geometry(int num_vecs) {
    Geometry g;
    g.num_vecs = num_vecs;
    g.num_blocks = num_vecs / BLOCK;
    g.tail_first = g.num_blocks * BLOCK;
    return g;
  }

  static void store(vec_t* ptr, const vec_t& value) { *ptr = value; }
  static vec_t load(const vec_t* ptr) { return *ptr; }

  static vec_t scale(const vec_t& values, float weight) {
    vec_t result;
#pragma unroll
    for (int e = 0; e < VEC; ++e)
      result[e] = Cvt<T>::from(Cvt<T>::to(values[e]) * weight);
    return result;
  }

  static void zero(vec_t* output, int lane, const Geometry& g) {
    const vec_t zero_value(static_cast<storage_t>(0));
    for (int b = lane; b < g.num_blocks; b += LANES) {
      vec_t* dst = output + b * BLOCK;
#pragma unroll
      for (int p = 0; p < BLOCK; ++p)
        store(dst + p, zero_value);
    }
    // With one vector per block the blocks cover the whole row, so the tail is
    // provably empty and must not cost a branch on the hot small-row path.
    if constexpr (BLOCK > 1) {
      for (int v = g.tail_first + lane; v < g.num_vecs; v += LANES)
        store(output + v, zero_value);
    }
  }

  static void copy_scaled(
      vec_t* output,
      const T* input,
      float weight,
      int lane,
      const Geometry& g) {
    const vec_t* source = reinterpret_cast<const vec_t*>(input);

    for (int b = lane; b < g.num_blocks; b += LANES) {
      const vec_t* src = source + b * BLOCK;
      vec_t* dst = output + b * BLOCK;
#pragma unroll
      for (int p = 0; p < BLOCK; ++p)
        store(dst + p, scale(load(src + p), weight));
    }

    if constexpr (BLOCK > 1) {
      for (int v = g.tail_first + lane; v < g.num_vecs; v += LANES)
        store(output + v, scale(load(source + v), weight));
    }
  }

  template <int TOPK>
  static void accumulate(
      vec_t* output,
      const T* moe_output,
      const int (&rows)[TOPK],
      const float (&scores)[TOPK],
      int src_stride,
      int lane,
      const Geometry& g) {
    for (int b = lane; b < g.num_blocks; b += LANES) {
      float accumulator[BLOCK][VEC];
#pragma unroll
      for (int p = 0; p < BLOCK; ++p) {
#pragma unroll
        for (int e = 0; e < VEC; ++e)
          accumulator[p][e] = 0.0f;
      }

#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        if (rows[k] < 0) continue;
        const vec_t* src =
            reinterpret_cast<const vec_t*>(
                moe_output + static_cast<size_t>(rows[k]) * src_stride) +
            b * BLOCK;
#pragma unroll
        for (int p = 0; p < BLOCK; ++p) {
          const vec_t values = load(src + p);
#pragma unroll
          for (int e = 0; e < VEC; ++e)
            accumulator[p][e] =
                sycl::fma(Cvt<T>::to(values[e]), scores[k], accumulator[p][e]);
        }
      }

      vec_t* dst = output + b * BLOCK;
#pragma unroll
      for (int p = 0; p < BLOCK; ++p) {
        vec_t result;
#pragma unroll
        for (int e = 0; e < VEC; ++e)
          result[e] = Cvt<T>::from(accumulator[p][e]);
        store(dst + p, result);
      }
    }

    if constexpr (BLOCK > 1) {
      for (int v = g.tail_first + lane; v < g.num_vecs; v += LANES) {
        float accumulator[VEC];
#pragma unroll
        for (int e = 0; e < VEC; ++e)
          accumulator[e] = 0.0f;

#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          if (rows[k] < 0) continue;
          const vec_t values = load(
              reinterpret_cast<const vec_t*>(
                  moe_output + static_cast<size_t>(rows[k]) * src_stride) +
              v);
#pragma unroll
          for (int e = 0; e < VEC; ++e)
            accumulator[e] =
                sycl::fma(Cvt<T>::to(values[e]), scores[k], accumulator[e]);
        }

        vec_t result;
#pragma unroll
        for (int e = 0; e < VEC; ++e)
          result[e] = Cvt<T>::from(accumulator[e]);
        store(output + v, result);
      }
    }
  }
};

template <typename T, int TOPK, int VEC, int BLOCK, int LANES, bool SPARSE>
class MoeGather {
 public:
  MoeGather(
      T* output,
      const T* moe_output,
      const float* topk_weights,
      const int* unpermuted_row_to_permuted_row,
      const int num_experts,
      const int num_tokens,
      const int hidden_size,
      const int src_stride)
      : output(output),
        moe_output(moe_output),
        topk_weights(topk_weights),
        unpermuted_row_to_permuted_row(unpermuted_row_to_permuted_row),
        num_experts(num_experts),
        num_tokens(num_tokens),
        hidden_size(hidden_size),
        src_stride(src_stride) {}

  void operator()(sycl::nd_item<1> item) const {
    constexpr int kTokensPerGroup = kWorkGroupSize / LANES;
    int token_idx;
    int lane;
    if constexpr (kTokensPerGroup == 1) {
      token_idx = static_cast<int>(item.get_group(0));
      lane = static_cast<int>(item.get_local_id(0));
    } else {
      const int local_id = static_cast<int>(item.get_local_id(0));
      token_idx = static_cast<int>(item.get_group(0)) * kTokensPerGroup +
                  local_id / LANES;
      lane = local_id % LANES;
    }
    if (token_idx >= num_tokens) return;

    using Ops = DirectRowOps<T, VEC, BLOCK, LANES>;
    using vec_t = typename Ops::vec_t;
    const auto row_geometry = Ops::geometry(hidden_size / VEC);
    vec_t* output_row = reinterpret_cast<vec_t*>(
        output + static_cast<size_t>(token_idx) * hidden_size);

    // The routing table is read once and kept in registers: the zero and
    // single-row paths would otherwise walk it again, and that second pass is
    // pure overhead on the short rows where the kernel is launch bound.
    const int base = token_idx * TOPK;
    int rows[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k)
      rows[k] = unpermuted_row_to_permuted_row[base + k];

    // With expert parallelism most of a token's routing table points at rows
    // another rank owns, so the common case is a row that is either untouched
    // or scaled from a single source. Skipping the accumulator for those is
    // what keeps the short, launch-bound dispatches cheap -- but it costs a
    // second pass over the table, which a dense table never recovers.
    if constexpr (SPARSE) {
      int valid_rows = 0;
      int first_valid = 0;
#pragma unroll
      for (int k = TOPK - 1; k >= 0; --k) {
        if (rows[k] >= 0) {
          ++valid_rows;
          first_valid = k;
        }
      }

      if (valid_rows == 0) {
        Ops::zero(output_row, lane, row_geometry);
        return;
      }

      if (valid_rows == 1) {
        Ops::copy_scaled(
            output_row,
            moe_output + static_cast<size_t>(rows[first_valid]) * src_stride,
            topk_weights[base + first_valid],
            lane,
            row_geometry);
        return;
      }
    }

    float scores[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k)
      scores[k] = topk_weights[base + k];

    Ops::template accumulate<TOPK>(
        output_row, moe_output, rows, scores, src_stride, lane, row_geometry);
  }

 private:
  T* output;
  const T* moe_output;
  const float* topk_weights;
  const int* unpermuted_row_to_permuted_row;
  const int num_experts;
  const int num_tokens;
  const int hidden_size;
  const int src_stride;
};

template <typename T, int TOPK, int VEC, int BLOCK, int LANES, bool SPARSE>
void launch_direct_moe_gather(
    T* output,
    const T* moe_output,
    const float* topk_weights,
    const int* unpermuted_row_to_permuted_row,
    int num_experts,
    int num_tokens,
    int hidden_size,
    int src_stride,
    sycl::queue& queue) {
  constexpr int kTokensPerGroup = kWorkGroupSize / LANES;
  const size_t groups =
      (static_cast<size_t>(num_tokens) + kTokensPerGroup - 1) / kTokensPerGroup;
  const sycl::range<1> local(kWorkGroupSize);
  const sycl::range<1> global(groups * kWorkGroupSize);
  const sycl::nd_range<1> range(global, local);

  namespace syclex = sycl::ext::oneapi::experimental;
#ifdef MOE_GATHER_HAS_GRF_PROP
  const syclex::properties kernel_props{
      syclex::sub_group_size<kSubGroupSize>,
      sycl::ext::intel::experimental::grf_size<MOE_GATHER_GRF>};
#else
  const syclex::properties kernel_props{
      syclex::sub_group_size<kSubGroupSize>};
#endif

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        range,
        kernel_props,
        MoeGather<T, TOPK, VEC, BLOCK, LANES, SPARSE>{
            output,
            moe_output,
            topk_weights,
            unpermuted_row_to_permuted_row,
            num_experts,
            num_tokens,
            hidden_size,
            src_stride});
  });
}

template <typename T>
void MoeGatherLauncher(
    T* output,
    const T* moe_output,
    const float* topk_weights,
    const int* unpermuted_row_to_permuted_row,
    const int num_experts,
    const int num_tokens,
    const int topk,
    const int hidden_size,
    const int src_stride,
    sycl::queue& queue) {
  constexpr int kVectorWidth = 32 / static_cast<int>(sizeof(T));

  int vec_width = kVectorWidth;
  while (vec_width > 1 &&
         (hidden_size % vec_width != 0 || src_stride % vec_width != 0 ||
          reinterpret_cast<uintptr_t>(output) % (vec_width * sizeof(T)) != 0 ||
          reinterpret_cast<uintptr_t>(moe_output) % (vec_width * sizeof(T)) !=
              0)) {
    vec_width /= 2;
  }

  const int row_bytes = hidden_size * static_cast<int>(sizeof(T));
  // Tuning policy fitted to a sweep of
  // (routing density x lanes-per-row x bytes-per-item) over the production
  // shape set; GRF is fixed (see MOE_GATHER_GRF above). The three token
  // regimes below are the ones where the dispatch is limited by different
  // things: launch latency, group supply and memory throughput respectively.
  // `num_experts` is the per-rank count, so a small value means expert
  // parallelism left most of each routing table pointing at another rank --
  // that is what makes the sparse kernel pay.
  int tuned_bytes = kMinRowBytesPerItem;
  int tuned_lanes = kWorkGroupSize;
  bool tuned_sparse = false;

  if (num_tokens < kSaturationTokens) {
    // Too few rows to fill the device: each row owns a work-group and the only
    // lever left is how much of it the kernel has to set up per token.
    if (num_experts < kShardedExpertsMax) {
      tuned_sparse = true;
      if (topk < kWideAccumTopk) {
        if (row_bytes < kNarrowRowBytes) {
          tuned_sparse = false;
        }
      } else if (row_bytes >= kWideRowBytes) {
        tuned_bytes = kMaxRowBytesPerItem;
      } else if (num_tokens < 2) {
        tuned_bytes = 2 * kMinRowBytesPerItem;
      }
    }
  } else if (num_tokens < kMultiTokenGroupMaxTokens) {
    // Enough rows to keep the device busy but not enough to hide a narrow
    // per-item access, so the row is covered in as few passes as possible.
    if (row_bytes < kWideBlockRowBytes) {
      if (row_bytes < kMidRowBytes) {
        tuned_bytes = 2 * kMinRowBytesPerItem;
      } else {
        tuned_sparse = true;
      }
    } else {
      tuned_sparse = true;
      if (num_experts >= kShardedExpertsMax) {
        tuned_bytes = kMaxRowBytesPerItem;
      } else {
        tuned_bytes = 2 * kMinRowBytesPerItem;
        if (row_bytes < kVeryWideRowBytes) tuned_lanes = kMinLanesPerRow;
      }
    }
  } else {
    // Saturated: the gather is bandwidth bound, so what matters is how many
    // rows a work-group walks and how wide each of its accesses is.
    if (num_experts < kShardedExpertsMax) {
      if (row_bytes < kWideBlockRowBytes) {
        tuned_bytes = 2 * kMinRowBytesPerItem;
      } else if (row_bytes < kVeryWideRowBytes) {
        tuned_sparse = true;
        tuned_lanes = 2 * kMinLanesPerRow;
      } else {
        tuned_lanes = kMinLanesPerRow;
      }
    } else {
      tuned_sparse = true;
      if (num_experts < kDenseExpertsMin) {
        tuned_bytes = 2 * kMinRowBytesPerItem;
        tuned_lanes = kMinLanesPerRow;
      } else {
        tuned_bytes = kMaxRowBytesPerItem;
        tuned_lanes = 2 * kMinLanesPerRow;
      }
    }
  }

  // Never hand an item more bytes than the row can keep busy.
  while (tuned_bytes > kMinRowBytesPerItem && row_bytes < tuned_bytes * 8) {
    tuned_bytes /= 2;
  }

  const int bytes_per_item = tuned_bytes;
  const int lanes = tuned_lanes;
  const bool sparse_routing = tuned_sparse;

#define LAUNCH(TOPK, VEC, BLOCK, LANES)                         \
  launch_direct_moe_gather<T, TOPK, VEC, BLOCK, LANES, SPARSE>( \
      output,                                                   \
      moe_output,                                               \
      topk_weights,                                             \
      unpermuted_row_to_permuted_row,                           \
      num_experts,                                              \
      num_tokens,                                               \
      hidden_size,                                              \
      src_stride,                                               \
      queue)

#define DISPATCH_TOPK(VEC, BLOCK, LANES)                    \
  switch (topk) {                                           \
    case 1:                                                 \
      LAUNCH(1, VEC, BLOCK, LANES);                         \
      break;                                                \
    case 2:                                                 \
      LAUNCH(2, VEC, BLOCK, LANES);                         \
      break;                                                \
    case 3:                                                 \
      LAUNCH(3, VEC, BLOCK, LANES);                         \
      break;                                                \
    case 4:                                                 \
      LAUNCH(4, VEC, BLOCK, LANES);                         \
      break;                                                \
    case 6:                                                 \
      LAUNCH(6, VEC, BLOCK, LANES);                         \
      break;                                                \
    case 7:                                                 \
      LAUNCH(7, VEC, BLOCK, LANES);                         \
      break;                                                \
    case 8:                                                 \
      LAUNCH(8, VEC, BLOCK, LANES);                         \
      break;                                                \
    case 10:                                                \
      LAUNCH(10, VEC, BLOCK, LANES);                        \
      break;                                                \
    case 12:                                                \
      LAUNCH(12, VEC, BLOCK, LANES);                        \
      break;                                                \
    case 16:                                                \
      LAUNCH(16, VEC, BLOCK, LANES);                        \
      break;                                                \
    default:                                                \
      TORCH_CHECK(false, "error: not support TOPK=", topk); \
  }

// A work-item covers BLOCK vectors, so 64B/32B/16B granularity maps onto the
// full 32-byte vector with BLOCK 2 or 1, and onto the half-width vector.
#define DISPATCH_GRAN(LANES)                     \
  switch (bytes_per_item) {                      \
    case 16:                                     \
      DISPATCH_TOPK(kVectorWidth / 2, 1, LANES); \
      break;                                     \
    case 32:                                     \
      DISPATCH_TOPK(kVectorWidth, 1, LANES);     \
      break;                                     \
    default:                                     \
      DISPATCH_TOPK(kVectorWidth, 2, LANES);     \
      break;                                     \
  }

#define DISPATCH_LANES()                                            \
  switch (lanes) {                                                  \
    case 64:                                                        \
      DISPATCH_GRAN(64);                                            \
      break;                                                        \
    case 128:                                                       \
      DISPATCH_GRAN(128);                                           \
      break;                                                        \
    case 256:                                                       \
      DISPATCH_GRAN(256);                                           \
      break;                                                        \
    default:                                                        \
      TORCH_CHECK(                                                  \
          false,                                                    \
          "moe_gather: lanes per row must be 64, 128 or 256, got ", \
          lanes);                                                   \
  }

  if (vec_width == kVectorWidth) {
    if (sparse_routing) {
      constexpr bool SPARSE = true;
      DISPATCH_LANES();
    } else {
      constexpr bool SPARSE = false;
      DISPATCH_LANES();
    }
  } else {
    constexpr bool SPARSE = true;
    switch (vec_width) {
      case 8:
        DISPATCH_TOPK(8, 1, 256);
        break;
      case 4:
        DISPATCH_TOPK(4, 1, 256);
        break;
      case 2:
        DISPATCH_TOPK(2, 1, 256);
        break;
      default:
        DISPATCH_TOPK(1, 1, 256);
        break;
    }
  }

#undef DISPATCH_LANES
#undef DISPATCH_GRAN
#undef DISPATCH_TOPK
#undef LAUNCH
}

}  // namespace moe
}  // namespace vllm

void moe_gather(
    torch::Tensor& output,              // [num_tokens, hidden_size]
    const torch::Tensor& moe_output,    // [num_tokens * topk, hidden_size]
    const torch::Tensor& topk_weights,  // [num_tokens, topk]
    const torch::Tensor& unpermuted_row_to_permuted_row,  // [num_tokens * topk]
    const int64_t num_experts) {
  const int num_tokens = topk_weights.size(0);
  const int topk = topk_weights.size(1);
  const int hidden_size = output.size(1);

  TORCH_CHECK(
      topk_weights.scalar_type() == torch::kFloat32,
      "topk_weights must be float32");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

  const int64_t src_stride64 = moe_output.stride(-2);
  TORCH_CHECK(
      moe_output.stride(-1) == 1,
      "moe_output must be unit-stride along the hidden dimension");
  TORCH_CHECK(
      src_stride64 >= hidden_size,
      "moe_output row stride ",
      src_stride64,
      " is smaller than hidden_size ",
      hidden_size);
  TORCH_CHECK(
      src_stride64 <= std::numeric_limits<int>::max(),
      "moe_output row stride does not fit in int32");
  const int src_stride = static_cast<int>(src_stride64);

  const at::DeviceGuard device_guard(output.device());
  auto& queue = vllm::xpu::vllmGetQueue();

#define LAUNCH_MOE_GATHER(T)                                             \
  vllm::moe::MoeGatherLauncher<T>(                                       \
      reinterpret_cast<T*>(output.data_ptr()),                           \
      reinterpret_cast<T*>(moe_output.data_ptr()),                       \
      reinterpret_cast<float*>(topk_weights.data_ptr()),                 \
      reinterpret_cast<int*>(unpermuted_row_to_permuted_row.data_ptr()), \
      num_experts,                                                       \
      num_tokens,                                                        \
      topk,                                                              \
      hidden_size,                                                       \
      src_stride,                                                        \
      queue);

  if (output.scalar_type() == torch::kFloat16) {
    using scalar_t = sycl::half;
    LAUNCH_MOE_GATHER(scalar_t);
  } else if (output.scalar_type() == torch::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    LAUNCH_MOE_GATHER(scalar_t);
  } else if (output.scalar_type() == torch::kFloat32) {
    using scalar_t = float;
    LAUNCH_MOE_GATHER(scalar_t);
  } else {
    throw std::runtime_error("Unsupported data type in moe_gather");
  }
#undef LAUNCH_MOE_GATHER
}