// SPDX-License-Identifier: Apache-2.0
#define MINICPM5_MONO_FP16 1
#include "minicpm5/mono.h"

#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

#if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) && \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED)
namespace vllm::minicpm5::mono_fp16 {
template <int TP>
class DraftLayer;

namespace {
constexpr int Rows = DraftRows;

inline UInt8 draft_weight_tile(sycl::sub_group sg, const scalar_t* row) {
  namespace ex = sycl::ext::oneapi::experimental;
  sycl::vec<uint32_t, 8> values;
  auto source = sycl::address_space_cast<
      sycl::access::address_space::global_space,
      sycl::access::decorated::yes>(reinterpret_cast<const uint32_t*>(row));
  ex::group_load(
      sg,
      source.get_decorated(),
      values,
      ex::properties{
          ex::data_placement_striped,
          ex::contiguous_memory,
          ex::full_group,
          ex::alignment<32>});
  return sycl::bit_cast<UInt8>(values);
}

// Each WG owns its output columns, so the two TP exchanges can follow
// projection with only a local handoff. Split K across its remaining SGs.
template <
    int K,
    int N,
    int Slices,
    bool Gate = false,
    bool LocalOut = false,
    bool PackedOut = false>
void draft_project(
    sycl::nd_item<1> item,
    const scalar_t* weight,
    const uint32_t* packed,
    float* partials,
    scalar_t* out) {
  constexpr int Tiles = N / 16;
  constexpr int TilesPerGroup = (Tiles + Groups - 1) / Groups;
  constexpr int BaseChunks = (K / 16) / Slices;
  constexpr int ExtraChunks = (K / 16) % Slices;
  static_assert(Subgroups >= TilesPerGroup * Slices && K % 16 == 0);
  const auto sg = item.get_sub_group();
  const int s = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
  const int lid = item.get_local_linear_id();
  const int wg = item.get_group_linear_id();
  const int tile = wg * TilesPerGroup + s / Slices;
  if (s < TilesPerGroup * Slices && tile < Tiles) {
    const int part = s % Slices;
    const int first = part * BaseChunks + sycl::min(part, ExtraChunks);
    const int chunks = BaseChunks + int(part < ExtraChunks);
    constexpr int TileElements = 16 * 16;
    const auto* row = weight + (size_t(tile) * (K / 16) + first) * TileElements;
    const auto* up = row + (Gate ? size_t(N) * K : 0);
    Float8 values{}, ups{};
    auto w = draft_weight_tile(sg, row);
    UInt8 u{};
    if constexpr (Gate) u = draft_weight_tile(sg, up);
  #pragma unroll 2
    for (int c = 0; c + 1 < chunks; ++c) {
      const auto next = draft_weight_tile(sg, row + (c + 1) * TileElements);
      UInt8 next_up{};
      if constexpr (Gate)
        next_up = draft_weight_tile(sg, up + (c + 1) * TileElements);
      const UInt4 a = projection_a<Rows>(packed, first + c, lane);
      dpas16(values, a, w);
      if constexpr (Gate) dpas16(ups, a, u);
      w = next;
      if constexpr (Gate) u = next_up;
    }
    const auto a = projection_a<Rows>(packed, first + chunks - 1, lane);
    dpas16(values, a, w);
    if constexpr (Gate) dpas16(ups, a, u);
  #pragma unroll
    for (int r = 0; r < Rows; ++r) {
      const int at = (s * Rows + r) * 16 + lane;
      partials[at] = values[r];
      if constexpr (Gate) partials[Subgroups * Rows * 16 + at] = ups[r];
    }
  }
  reference_wg_bar(item.get_group());
  if constexpr (PackedOut) {
    static_assert(Gate && !LocalOut);
    auto* packed_out = reinterpret_cast<uint32_t*>(out);
    for (int job = lid; job < TilesPerGroup * 8 * Rows; job += Local) {
      const int local_tile = job / 8 / Rows;
      const int tile = wg * TilesPerGroup + local_tile;
      if (tile >= Tiles) continue;
      const int r = job / 8 % Rows, pair = job % 8;
      scalar_t value[2];
      for (int half = 0; half < 2; ++half) {
        float gate = 0.f, up = 0.f;
        for (int part = 0; part < Slices; ++part) {
          const int at =
              ((local_tile * Slices + part) * Rows + r) * 16 + pair * 2 + half;
          gate += partials[at];
          up += partials[Subgroups * Rows * 16 + at];
        }
        const float g = float(scalar_t(gate));
        value[half] =
            scalar_t((g / (1.f + sycl::exp(-g))) * float(scalar_t(up)));
      }
      packed_out[(tile * Rows + r) * 8 + pair] = pack16(value[0], value[1]);
    }
    return;
  }
  for (int job = lid; job < TilesPerGroup * 16 * Rows; job += Local) {
    const int local_tile = job / 16 / Rows;
    const int tile = wg * TilesPerGroup + local_tile;
    if (tile >= Tiles) continue;
    const int r = job / 16 % Rows, n = tile * 16 + job % 16;
    float value = 0.f, up = 0.f;
  #pragma unroll
    for (int p = 0; p < Slices; ++p) {
      const int at = ((local_tile * Slices + p) * Rows + r) * 16 + job % 16;
      value += partials[at];
      if constexpr (Gate) up += partials[Subgroups * Rows * 16 + at];
    }
    if constexpr (Gate) {
      const float g = float(scalar_t(value));
      value = (g / (1.f + sycl::exp(-g))) * float(scalar_t(up));
    }
    if constexpr (LocalOut) {
      static_assert(N == H && TilesPerGroup * 16 == H / Groups);
      out[r * (H / Groups) + local_tile * 16 + job % 16] = scalar_t(value);
    } else {
      out[r * N + n] = scalar_t(value);
    }
  }
}

void draft_norm_pack(
    sycl::nd_item<1> item,
    const scalar_t* hidden,
    const scalar_t* residual,
    const scalar_t* weight,
    uint32_t* packed,
    float* partial) {
  const auto sg = item.get_sub_group();
  const int lid = item.get_local_linear_id();
  const int col = lid * 4;
  const int subgroup = sg.get_group_linear_id();
  #pragma unroll 1
  for (int r = 0; r < Rows; ++r) {
    float sum = 0.f;
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int offset = r * H + col + j;
      const scalar_t value(
          float(hidden[offset]) + (residual ? float(residual[offset]) : 0.f));
      sum += float(value) * float(value);
    }
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    if (sg.get_local_linear_id() == 0) partial[r * Subgroups + subgroup] = sum;
  }
  reference_wg_bar(item.get_group());
  if (lid < Rows) {
    float sum = 0.f;
  #pragma unroll 1
    for (int i = 0; i < Subgroups; ++i)
      sum += partial[lid * Subgroups + i];
    partial[Rows * Subgroups + lid] = sycl::rsqrt(sum / H + 1e-6f);
  }
  reference_wg_bar(item.get_group());
  #pragma unroll 1
  for (int r = 0; r < Rows; ++r) {
    const float inv = partial[Rows * Subgroups + r];
    scalar_t normalized[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int offset = r * H + col + j;
      const scalar_t value(
          float(hidden[offset]) + (residual ? float(residual[offset]) : 0.f));
      normalized[j] = scalar_t(float(value) * inv * float(weight[col + j]));
    }
    const int tile = ((col / 16) * Rows + r) * 8 + (col % 16) / 2;
    packed[tile] = pack16(normalized[0], normalized[1]);
    packed[tile + 1] = pack16(normalized[2], normalized[3]);
  }
  reference_wg_bar(item.get_group());
}

template <int TP>
inline void norm_rotary_and_store(
    sycl::nd_item<1> item,
    DraftWeights<scalar_t> weights,
    Workspace work,
    Step step) {
  constexpr int Q = 32 / TP, KV = 2 / TP, QKV = (Q + 2 * KV) * D;
  const auto sg = item.get_sub_group();
  const int lane = sg.get_local_linear_id();
  const int id = sg.get_group_linear_id() * Groups + item.get_group_linear_id();
  if (id >= Rows * (Q + KV)) return;
  const int r = id / (Q + KV), head = id % (Q + KV);
  const int64_t slot = step.slots[r];
  if (slot < 0) return;
  auto* qkv = work.qkv + r * QKV;
  auto* p = qkv + head * D;
  const bool is_key = KV == 1 ? head == Q : head >= Q;
  const auto* weight = is_key ? weights.k_norm : weights.q_norm;
  const auto* rope = step.rope_cache + step.positions[r] * D;
  float sum = 0.f;
  #pragma unroll
  for (int d = lane; d < D; d += 16)
    sum += float(p[d]) * float(p[d]);
  sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
  const float inv = sycl::rsqrt(sum / D + 1e-6f);
  int64_t cache_offset = (slot / step.block_size) * step.cache_block_stride +
                         (slot % step.block_size) * step.cache_token_stride;
  if constexpr (KV > 1)
    cache_offset += (is_key ? head - Q : 0) * step.cache_head_stride;
  #pragma unroll
  for (int d = lane; d < D / 2; d += 16) {
    const float a = float(scalar_t(float(p[d]) * inv * float(weight[d])));
    const float b =
        float(scalar_t(float(p[d + D / 2]) * inv * float(weight[d + D / 2])));
    const float c = float(rope[d]), s = float(rope[d + D / 2]);
    const scalar_t first(a * c - b * s), second(b * c + a * s);
    if (is_key) {
      step.cache[cache_offset + d] = first;
      step.cache[cache_offset + d + D / 2] = second;
      const auto* v = qkv + (KV == 1 ? Q + 1 : head + KV) * D;
      step.cache[cache_offset + d + D] = v[d];
      step.cache[cache_offset + d + 3 * D / 2] = v[d + D / 2];
    } else {
      p[d] = first;
      p[d + D / 2] = second;
    }
  }
}

template <int TP, int Phase>
bool draft_all_reduce(
    sycl::nd_item<1> item, const scalar_t* local, Step step, uint32_t call) {
  const int lid = item.get_local_linear_id();
  constexpr int Pairs = Rows * (H / Groups / 2);
  const int r = lid / (H / Groups / 2);
  const int col =
      item.get_group_linear_id() * (H / Groups) + 2 * (lid % (H / Groups / 2));
  const int idx = r * H + col;
  auto* out = Phase == 0 ? step.res_out : step.output;
  const int partial = r * (H / Groups) + 2 * (lid % (H / Groups / 2));
  float a = 0.f, b = 0.f;
  if (lid < Pairs) {
    a = float(local[partial]);
    b = float(local[partial + 1]);
  }
  if constexpr (TP == 2) {
    if (!reduce_fp16_packet<Rows>(item, a, b, Phase, call, step)) return false;
  }
  if (lid >= Pairs) return true;
  if constexpr (Phase == 0) {
    const auto residual = [&](int i) {
      return scalar_t(
          float(step.hidden[i]) +
          (step.residual ? float(step.residual[i]) : 0.f));
    };
    a = float(scalar_t(a)) + float(residual(idx));
    b = float(scalar_t(b)) + float(residual(idx + 1));
  }
  out[idx] = scalar_t(a);
  out[idx + 1] = scalar_t(b);
  return true;
}
}  // namespace

template <int TP>
void launch_draft(
    sycl::queue& queue,
    DraftWeights<scalar_t> weights,
    Workspace work,
    Step step) {
  constexpr int QDim = 32 / TP * D;
  constexpr int QKV = (32 + 4) / TP * D;
  constexpr int Intermediate = 8192 / TP;
  constexpr int ProjectionSlices = TP == 1 ? 2 : 4;
  constexpr int TileColumns = TP == 1 ? QDim : Intermediate;
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<scalar_t, 1> tp_out(Rows * (H / Groups), cgh);
    // Sized for the down-projection A tile and reused as Q or attention merge.
    sycl::local_accessor<UInt4, 1> tiles(Rows * TileColumns / 8, cgh);
    sycl::local_accessor<float, 1> partials(2 * Subgroups * Rows * 16, cgh);
    sycl::local_accessor<float, 1> norms(Rows * (Subgroups + 1), cgh);
    sycl::ext::oneapi::experimental::properties properties{
        sycl::ext::intel::experimental::grf_size<256>};
    cgh.parallel_for<DraftLayer<TP>>(
        sycl::nd_range<1>(Groups * Local, Local),
        properties,
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const int global = item.get_global_linear_id();
          const int lid = item.get_local_linear_id();
          const int wg = item.get_group_linear_id();
          const uint32_t call = epoch_load(step.control + Epoch) + 1;
          if (global < Rows) {
            const int r = global, length = step.lengths[r];
            const int64_t slot = step.slots[r], pos = step.positions[r];
            if (slot != -1) {
              bool valid = slot >= 0 && pos >= 0 && pos < length &&
                           pos < step.rope_cache_rows && length <= MaxContext &&
                           (length + step.block_size - 1) / step.block_size <=
                               step.block_table_cols;
              if (valid) {
                const int page =
                    step.blocks
                        [r * step.block_table_stride + pos / step.block_size];
                valid = page >= 0 && page < step.cache_blocks &&
                        slot == int64_t(page) * step.block_size +
                                    pos % step.block_size;
              }
              for (int i = 0; i < r; ++i)
                valid &= step.slots[i] != slot;
              if (!valid) fail(step.control, 3, 0, call, global, step.health);
            }
          }
          for (int r = 0; r < Rows; ++r) {
            const int length = step.lengths[r];
            if (length <= 0 || length > MaxContext) continue;
            const int pages = (length + step.block_size - 1) / step.block_size;
            if (step.slots[r] < 0 || pages > step.block_table_cols) continue;
            for (int b = global; b < pages; b += Groups * Local) {
              const int page = step.blocks[r * step.block_table_stride + b];
              if (page < 0 || page >= step.cache_blocks)
                fail(step.control, 3, 0, call, global, step.health);
            }
          }
          // Phase1 completes validation before the first KV write.
          if (wg == 0 && lid < Rows * (2 / TP) * 2)
            Flag(step.control[Epoch + 16 + lid]).store(0);
          auto* local =
              tp_out.get_multi_ptr<sycl::access::decorated::no>().get();
          auto* a = reinterpret_cast<uint32_t*>(
              tiles.get_multi_ptr<sycl::access::decorated::no>().get());
          auto* p = partials.get_multi_ptr<sycl::access::decorated::no>().get();
          auto* norm = norms.get_multi_ptr<sycl::access::decorated::no>().get();
          draft_norm_pack(
              item, step.hidden, step.residual, weights.input_norm, a, norm);
          draft_project<H, QKV, TP == 1 ? 3 : ProjectionSlices>(
              item, weights.qkv, a, p, work.qkv);
          if (!grid_barrier_handoff<true>(
                  item, step.control, 1, call, step.health))
            return;
          norm_rotary_and_store<TP>(item, weights, work, step);
          if (!grid_barrier_handoff<true>(
                  item, step.control, 2, call, step.health))
            return;
          auto* plan = reinterpret_cast<int*>(norm);
          if (step.block_size == 64 &&
              (step.cache_token_stride == 2 * D ||
               (TP == 1 && step.cache_token_stride == 4 * D)) &&
              step.cache_block_stride == 64 * (2 / TP) * 2 * D) {
            reference_attention<TP, Rows, false, true>(
                item, work, step, a, plan);
          } else {
            reference_attention<TP, Rows, false>(item, work, step, a, plan);
          }
          if (!grid_barrier_handoff<true>(
                  item, step.control, 3, call, step.health))
            return;
          reference_attention_merge<TP, Rows, false>(
              item, work, step, reinterpret_cast<float*>(a), plan);
          if (!grid_barrier_handoff<true>(
                  item, step.control, 4, call, step.health))
            return;
          pack_projection_input<Rows, QDim, false>(
              item, work.attention, QDim, a);
          reference_wg_bar(item.get_group());
          draft_project<QDim, H, 8, false, true>(
              item, weights.out, a, p, local);
          reference_wg_bar(item.get_group());
          if (!draft_all_reduce<TP, 0>(item, local, step, call)) return;
          if (!grid_barrier_handoff<true>(
                  item, step.control, 5, call, step.health))
            return;
          draft_norm_pack(
              item, step.res_out, nullptr, weights.post_norm, a, norm);
          // Phase6 publishes gate/up directly in the down input's DPAS layout.
          auto* packed_intermediate = TP == 1 ? work.qkv : work.intermediate;
          draft_project<H, Intermediate, ProjectionSlices, true, false, true>(
              item, weights.gate_up, a, p, packed_intermediate);
          if (!grid_barrier_handoff<true>(
                  item, step.control, 6, call, step.health))
            return;
          draft_project<Intermediate, H, 8, false, true>(
              item,
              weights.down,
              reinterpret_cast<const uint32_t*>(packed_intermediate),
              p,
              local);
          reference_wg_bar(item.get_group());
          if (!draft_all_reduce<TP, 1>(item, local, step, call)) return;
          if (global == 0) Flag(step.control[Epoch]).store(call);
        });
  });
}

void dispatch_draft(
    sycl::queue& queue,
    DraftWeights<scalar_t> weights,
    Workspace work,
    Step step,
    int tp) {
  if (tp == 1)
    launch_draft<1>(queue, weights, work, step);
  else
    launch_draft<2>(queue, weights, work, step);
}
}  // namespace vllm::minicpm5::mono_fp16
#endif
