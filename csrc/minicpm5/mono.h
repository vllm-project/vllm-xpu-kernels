// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/experimental/clock.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>

#include <cstdint>
#include <cmath>
#include <type_traits>

#include "minicpm5/dpas.h"
#include "minicpm5/arguments.h"

namespace vllm::minicpm5::MINICPM5_MONO_NAMESPACE {
constexpr int H = 2048, E = 160, Top = 16, D = 128;
constexpr int Groups = 32, Local = 512, Subgroups = Local / 16;
constexpr int MaxRows = 8, Phases = 16;
#ifdef MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED
constexpr int ControlWords = Phases * 16 + 16 + MaxRows * 4;
#else
constexpr int ControlWords = Phases * 16 + 16;
#endif
constexpr int Epoch = Phases * 16, Failed = Epoch + 1;
constexpr int FailureKind = Failed + 1, FailurePhase = Failed + 2;
constexpr int FailureCall = Failed + 3, FailureObserved = Failed + 4;
constexpr uint64_t WaitCycles = 1ull << 35;
constexpr size_t TaggedMailboxWords = 2 * MaxRows * H;
constexpr int MailboxPairs = MaxRows * (H / Groups / 2);
constexpr int ReadyStride = 8;  // One cache line per WG.
constexpr size_t PacketWords = Groups * (MailboxPairs / 2 + ReadyStride);
constexpr size_t MailboxWords = TaggedMailboxWords + 4 * PacketWords;

using Weights = WeightPointers<scalar_t>;
using Workspace = WorkspacePointers<scalar_t>;
using Step = StepPointers<scalar_t>;

using Flag = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>;
using Mail = sycl::atomic_ref<
    uint64_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>;

// The epoch is written only at the end of the preceding in-order launch.
// A plain L1-uncached read avoids an atomic_or RMW on every work-item. The
// intra-launch hand-off counters retain release/acquire semantics below.
inline uint32_t epoch_load(const uint32_t* pointer) {
  namespace ex = sycl::ext::oneapi::experimental;
  namespace intel = sycl::ext::intel::experimental;
  ex::annotated_ptr<
      const uint32_t,
      decltype(ex::properties{intel::read_hint<intel::cache_control<
          intel::cache_mode::uncached,
          ex::cache_level::L1>>})>
      value(pointer);
  return *value;
}

inline uint64_t clock_now() {
  return sycl::ext::oneapi::experimental::clock<
      sycl::ext::oneapi::experimental::clock_scope::sub_group>();
}

inline void fail(
    uint32_t* control,
    uint32_t kind,
    uint32_t phase,
    uint32_t call,
    uint32_t observed,
    uint32_t* health = nullptr) {
  if (health) Flag(health[0]).store(1);
  uint32_t expected = 0;
  if (Flag(control[Failed]).compare_exchange_strong(expected, 1)) {
    Flag(control[FailureKind]).store(kind);
    Flag(control[FailurePhase]).store(phase);
    Flag(control[FailureCall]).store(call);
    Flag(control[FailureObserved]).store(observed);
  }
}

// Directly reused from the reference wg_bar: SLM-only synchronization does
// not wait for unrelated global preloads or invalidate the data cache.
inline void reference_wg_bar(sycl::group<1> group) {
#ifdef __SYCL_DEVICE_ONLY__
  __spirv_ControlBarrier(
      __spv::Scope::Workgroup,
      __spv::Scope::Workgroup,
      __spv::MemorySemanticsMask::AcquireRelease |
          __spv::MemorySemanticsMask::WorkgroupMemory);
#else
  sycl::group_barrier(group);
#endif
}

#include "minicpm5/reference_norm.h"

template <int LocalWriters = Local>
inline void grid_signal(
    sycl::nd_item<1> item,
    uint32_t* control,
    int phase,
    uint32_t call,
    uint32_t* health = nullptr) {
  // Every actual global writer releases before the leader publishes the WG.
  // Read-only lanes need not drain unrelated preloads at this handoff.
  static_assert(LocalWriters > 0 && LocalWriters <= Local);
  if constexpr (LocalWriters == Local) {
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
  } else if (item.get_local_linear_id() < LocalWriters) {
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
  }
  reference_wg_bar(item.get_group());
  if (item.get_local_linear_id() == 0)
    Flag(control[phase * 16]).fetch_add(1, sycl::memory_order::release);
}

template <int GlobalReaders = Groups * Local, bool LeaderAcquire = false>
inline bool grid_wait(
    sycl::nd_item<1> item,
    uint32_t* control,
    int phase,
    uint32_t call,
    uint32_t* health = nullptr) {
  // Match reference wait_for: one polling lane per WG, relaxed polls, and
  // health/clock checks only on the slow path. An acquire on every poll
  // invalidates L1 repeatedly and stalls the entire persistent grid.
  if (item.get_local_linear_id() == 0) {
    const uint64_t start = clock_now();
    uint32_t polls = 0;
    while (int32_t(
               Flag(control[phase * 16]).load(sycl::memory_order::relaxed) -
               Groups * call) < 0) {
      if (++polls % 256 == 0 &&
          (Flag(control[Failed]).load() || (health && Flag(health[0]).load()) ||
           clock_now() - start > WaitCycles)) {
        fail(control, 1, phase, call, Flag(control[phase * 16]).load(), health);
        break;
      }
    }
  }
  static_assert(GlobalReaders >= 0 && GlobalReaders <= Groups * Local);
  if constexpr (LeaderAcquire) {
    // Counter release-sequence -> leader device acquire -> WG barrier.
    // The barrier covers global AND local memory, propagating the acquired
    // producer writes to all consumers; it is not the SLM-only wg_bar.
    if (item.get_local_linear_id() == 0)
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::device);
    sycl::group_barrier(item.get_group(), sycl::memory_scope::work_group);
  } else {
    reference_wg_bar(item.get_group());
    static_assert(GlobalReaders >= 0 && GlobalReaders <= Groups * Local);
    // Reused global scratch requires a consumer acquire. The default covers
    // every work-item. Two callers prove a smaller payload-reader set: B1
    // rotary owners after QKV, and no payload readers after final completion.
    // Every WG leader still acquires before checking the shared health flags.
    if constexpr (GlobalReaders == Groups * Local) {
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::device);
    } else if (
        item.get_local_linear_id() == 0 ||
        item.get_global_linear_id() < GlobalReaders) {
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::device);
    }
  }
  // Every producer has completed before this point. Read the sticky health
  // once per work-group and broadcast it instead of issuing 512 RMW loads
  // and acquire invalidations against the same cache line.
  bool healthy = false;
  if (item.get_local_linear_id() == 0)
    healthy =
        Flag(control[Failed]).load(sycl::memory_order::relaxed) == 0 &&
        (!health || Flag(health[0]).load(sycl::memory_order::relaxed) == 0);
  return sycl::group_broadcast(item.get_group(), healthy, 0);
}

inline bool grid_barrier(
    sycl::nd_item<1> item,
    uint32_t* control,
    int phase,
    uint32_t call,
    uint32_t* health = nullptr) {
  grid_signal(item, control, phase, call, health);
  return grid_wait(item, control, phase, call, health);
}

template <bool LeaderAcquire>
inline bool grid_barrier_handoff(
    sycl::nd_item<1> item,
    uint32_t* control,
    int phase,
    uint32_t call,
    uint32_t* health = nullptr) {
  if constexpr (LeaderAcquire) {
    grid_signal(item, control, phase, call, health);
    return grid_wait<Groups * Local, true>(item, control, phase, call, health);
  } else {
    return grid_barrier(item, control, phase, call, health);
  }
}

inline bool row_metadata_valid(
    int length,
    int64_t position,
    int64_t slot,
    int block_size,
    int64_t cache_blocks,
    int64_t block_table_cols,
    int64_t rope_cache_rows) {
  if (slot == -1) return true;
  return slot >= 0 && block_size > 0 && cache_blocks > 0 &&
         slot / block_size < cache_blocks && length > 0 &&
         length <= MaxContext && position == int64_t(length) - 1 &&
         (position < 32768 || position < rope_cache_rows) &&
         (int64_t(length) + block_size - 1) / block_size <= block_table_cols;
}

inline void
rotary_pair(const Step& step, int r, int d, scalar_t& cs, scalar_t& sn) {
  const int64_t position = step.positions[r];
  if (position >= 32768 && step.slots[r] >= 0) {
    cs = step.rope_cache[position * D + d];
    sn = step.rope_cache[position * D + D / 2 + d];
  } else {
    // Preserve the qualified short-context rounding; long positions use
    // the model's canonical cache instead of an independent pow/trig path.
    const float angle = float(position) * sycl::pow(5000000.f, -2.f * d / D);
    cs = scalar_t(sycl::cos(angle));
    sn = scalar_t(sycl::sin(angle));
  }
}

inline uint64_t encode_mail(float value, uint32_t tag) {
  const uint32_t bits = sycl::bit_cast<uint32_t>(value);
  const uint64_t t = tag & 0xffffu;
  return (t << 48) | (uint64_t(bits >> 16) << 32) | (t << 16) |
         (bits & 0xffffu);
}

inline bool mail_matches(uint64_t value, uint32_t tag) {
  const uint32_t t = tag & 0xffffu;
  return uint32_t(value >> 48) == t && ((value >> 16) & 0xffffu) == t;
}

inline uint32_t mail_payload(uint64_t value) {
  return (uint32_t(value >> 32) << 16) | (value & 0xffffu);
}

inline uint32_t receive_payload_slow(
    uint64_t* pointer,
    uint32_t tag,
    uint32_t* control,
    uint32_t offset,
    uint32_t* health = nullptr) {
  const uint64_t start = clock_now();
  uint64_t value;
  uint32_t polls = 0;
  do {
    value = Mail(*pointer).load();
    if (++polls % 256 == 0 &&
        (Flag(control[Failed]).load() || (health && Flag(health[0]).load()) ||
         clock_now() - start > WaitCycles)) {
      fail(
          control,
          2,
          offset,
          tag,
          uint32_t(value >> 48) | (uint32_t((value >> 16) & 0xffffu) << 16),
          health);
      return sycl::bit_cast<uint32_t>(float(NAN));
    }
  } while (!mail_matches(value, tag));
  return mail_payload(value);
}

inline uint32_t receive_payload(
    uint64_t* pointer,
    uint32_t tag,
    uint32_t* control,
    uint32_t offset,
    uint32_t* health = nullptr) {
  const uint64_t value = Mail(*pointer).load();
  if (!mail_matches(value, tag))
    return receive_payload_slow(pointer, tag, control, offset, health);
  return mail_payload(value);
}

inline float receive(
    uint64_t* pointer,
    uint32_t tag,
    uint32_t* control,
    uint32_t offset,
    uint32_t* health = nullptr) {
  return sycl::bit_cast<float>(
      receive_payload(pointer, tag, control, offset, health));
}

// Reference ar_two transport: both local partials have already been rounded
// to BF16, so two values fit losslessly in one tagged mailbox word.
template <int TP>
inline void reduce_pair(
    float& first,
    float& second,
    int phase,
    int row,
    int col,
    uint32_t tag,
    const Step& step) {
  if constexpr (TP == 1) return;
  const size_t offset = (size_t(phase) * MaxRows + row) * H + col / 2;
  const uint32_t payload = pack16(scalar_t(first), scalar_t(second));
  Mail(step.peer[offset])
      .store(encode_mail(sycl::bit_cast<float>(payload), tag));
  const uint32_t peer = receive_payload(
      step.mailbox + offset, tag, step.control, offset, step.health);
  first += float(sycl::bit_cast<scalar_t>(uint16_t(peer)));
  second += float(sycl::bit_cast<scalar_t>(uint16_t(peer >> 16)));
}

// One FP16 pair per work-item is one dword in a contiguous WG message.
// Double buffering prevents an early peer from overwriting an unread packet.
template <int Rows>
inline bool reduce_fp16_packet(
    sycl::nd_item<1> item,
    float& first,
    float& second,
    int phase,
    uint32_t call,
    const Step& step) {
  static_assert(Rows > 0 && Rows <= MaxRows);
  constexpr int Pairs = Rows * (H / Groups / 2);
  const int lid = item.get_local_linear_id();
  const int wg = item.get_group_linear_id();
  const size_t base =
      TaggedMailboxWords + (phase * 2 + (call & 1u)) * PacketWords;
  const size_t ready = base + Groups * MailboxPairs / 2 + wg * ReadyStride;
  const size_t index = wg * MailboxPairs + lid;
  if (lid < Pairs) {
    auto* destination = reinterpret_cast<volatile uint32_t*>(step.peer + base);
    destination[index] = pack16(scalar_t(first), scalar_t(second));
  }
  item.barrier(sycl::access::fence_space::global_and_local);
  bool ok = false;
  if (lid == 0) {
#ifdef __SYCL_DEVICE_ONLY__
    asm volatile("lsc_fence.ugm.clean.sysrel" ::: "memory");
#endif
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    const uint64_t tag = (uint64_t(1) << 32) | call;
    Mail(step.peer[ready]).store(tag);
    const uint64_t start = clock_now();
    uint32_t polls = 0;
    uint64_t incoming;
    while ((incoming = Mail(step.mailbox[ready]).load()) != tag) {
      if (++polls % 256 == 0 && (Flag(step.control[Failed]).load() ||
                                 (step.health && Flag(step.health[0]).load()) ||
                                 clock_now() - start > WaitCycles)) {
        fail(step.control, 2, phase, call, uint32_t(incoming), step.health);
        break;
      }
    }
    ok = incoming == tag;
#ifdef __SYCL_DEVICE_ONLY__
    asm volatile("lsc_fence.ugm.invalidate.sysacq" ::: "memory");
#endif
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
  }
  item.barrier(sycl::access::fence_space::global_and_local);
  if (!sycl::group_broadcast(item.get_group(), ok, 0)) return false;
  if (lid < Pairs) {
    const auto* source = reinterpret_cast<const uint32_t*>(step.mailbox + base);
    const uint32_t peer = source[index];
    first += float(sycl::bit_cast<scalar_t>(uint16_t(peer)));
    second += float(sycl::bit_cast<scalar_t>(uint16_t(peer >> 16)));
  }
  return true;
}

template <int TP>
inline float reduce(
    float value, int phase, int row, int col, uint32_t tag, const Step& step) {
  if constexpr (TP == 1) return value;
  const size_t offset = (size_t(phase) * MaxRows + row) * H + col;
  Mail(step.peer[offset]).store(encode_mail(value, tag));
  const float peer =
      receive(step.mailbox + offset, tag, step.control, offset, step.health);
  return value + peer;
}

inline float fp8(uint8_t value) {
  return sycl::bit_cast<float>(uint32_t(fp8_bits(value)) << 16);
}

// Issue the next 16-byte row load before computing the current block. The
// same weight registers feed all rows; BF16 inputs keep their full range.
template <int Rows, int K>
inline void projection(
    sycl::sub_group sg,
    const uint8_t* weight,
    const scalar_t* input,
    int input_stride,
    float (&result)[Rows]) {
  const int lane = sg.get_local_linear_id();
  for (int r = 0; r < Rows; ++r)
    result[r] = 0.f;
  sycl::uint4 current =
      *reinterpret_cast<const sycl::uint4*>(weight + lane * 16);
  for (int offset = lane * 16; offset < K; offset += 256) {
    sycl::uint4 next;
    if (offset + 256 < K)
      next = *reinterpret_cast<const sycl::uint4*>(weight + offset + 256);
    auto bytes = sycl::bit_cast<sycl::vec<uint8_t, 16>>(current);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float w = fp8(bytes[j]);
#pragma unroll
      for (int r = 0; r < Rows; ++r)
        result[r] += w * float(input[r * input_stride + offset + j]);
    }
    if (offset + 256 < K) current = next;
  }
#pragma unroll
  for (int r = 0; r < Rows; ++r)
    result[r] = sycl::reduce_over_group(sg, result[r], sycl::plus<float>());
}

#include "minicpm5/reference_projection.h"
#include "minicpm5/reference_attention.h"
#include "minicpm5/reference_router.h"
#include "minicpm5/reference_moe.h"
#include "minicpm5/reference_gemv.h"

#ifdef MINICPM5_MONO_DPAS_ENABLED
// DPAS QK uses the eight GQA queries as matrix rows. The FP32 online softmax
// and PV accumulation reuse each loaded value vector across those queries.
template <int TP, int Rows>
inline void matrix_attention(sycl::nd_item<1> item, Workspace work, Step step) {
  constexpr int Q = 32 / TP, KV = 2 / TP;
  constexpr int QKV = 2 * Q * D + 2 * KV * D;
  auto sg = item.get_sub_group();
  const int lane = sg.get_local_linear_id();
  const int wg = item.get_group_linear_id();
  for (int job = sg.get_group_linear_id(); job < Rows * KV * 2;
       job += Subgroups) {
    const int r = job / (KV * 2), kvh = job / 2 % KV;
    const int head = kvh * 16 + (job % 2) * 8;
    const int length = step.slots[r] < 0 ? 0 : step.lengths[r];
    const int chunk = (length + Groups - 1) / Groups;
    const int begin = wg * chunk, end = sycl::min(begin + chunk, length);
    Float8 maximum, denominator{};
    Float8 values[8] = {};
    int offsets[8];
  #pragma unroll
    for (int h = 0; h < 8; ++h) {
      maximum[h] = -INFINITY;
      offsets[h] = r * QKV + (head + h) * 2 * D;
    }
    for (int tile = begin; tile < end; tile += 16) {
      const int t = tile + lane;
      int64_t key_offset = 0;
      if (t < end) {
        const int block =
            step.blocks[r * step.block_table_stride + t / step.block_size];
        key_offset = int64_t(block) * step.cache_block_stride +
                     kvh * step.cache_head_stride +
                     (t % step.block_size) * step.cache_token_stride;
      }
      Float8 scores{};
      for (int k = 0; k < D; k += 16) {
        UInt8 keys;
  #pragma unroll
        for (int p = 0; p < 8; ++p)
          keys[p] = t < end ? pack16(
                                  step.cache[key_offset + k + 2 * p],
                                  step.cache[key_offset + k + 2 * p + 1])
                            : 0u;
        dpas16(scores, input_tile<8>(lane, work.qkv, k, offsets), keys);
      }
      Float8 probabilities;
  #pragma unroll
      for (int h = 0; h < 8; ++h) {
        const float score =
            t < end ? scores[h] * 0.08838834764831844f : -INFINITY;
        const float block_max =
            sycl::reduce_over_group(sg, score, sycl::maximum<float>());
        const float next = sycl::max(maximum[h], block_max);
        const float old = sycl::exp(maximum[h] - next);
        probabilities[h] = sycl::exp(score - next);
        denominator[h] =
            denominator[h] * old +
            sycl::reduce_over_group(sg, probabilities[h], sycl::plus<float>());
        maximum[h] = next;
  #pragma unroll
        for (int d = 0; d < 8; ++d)
          values[h][d] *= old;
      }
      for (int token = 0; token < 16 && tile + token < end; ++token) {
        const int pos = tile + token;
        const int block =
            step.blocks[r * step.block_table_stride + pos / step.block_size];
        const int64_t offset =
            int64_t(block) * step.cache_block_stride +
            kvh * step.cache_head_stride +
            (pos % step.block_size) * step.cache_token_stride + D;
        float loaded[8];
  #pragma unroll
        for (int d = 0; d < 8; ++d)
          loaded[d] = float(step.cache[offset + lane * 8 + d]);
  #pragma unroll
        for (int h = 0; h < 8; ++h) {
          const float p = sycl::select_from_group(sg, probabilities[h], token);
  #pragma unroll
          for (int d = 0; d < 8; ++d)
            values[h][d] += p * loaded[d];
        }
      }
    }
  #pragma unroll
    for (int h = 0; h < 8; ++h) {
      const size_t offset =
          ((size_t(r) * Groups + wg) * Q + head + h) * (D + 2);
  #pragma unroll
      for (int d = 0; d < 8; ++d)
        work.attention_partial[offset + lane * 8 + d] = values[h][d];
      if (lane == 0) {
        work.attention_partial[offset + D] = maximum[h];
        work.attention_partial[offset + D + 1] = denominator[h];
      }
    }
  }
}

#endif

template <int TP, int Rows, bool Dense>
class Layer;
template <int TP, int Rows, bool Dense>
class PaddedLayer;
template <int TP, int Rows, bool Dense>
class TiledQkvLayer;
template <int TP, int Rows, bool FixedPages>
class BatchedVerifyLayer;
template <int TP, int Rows, bool FixedPages>
class BatchedBestLayer;
template <int TP, int Rows, bool FixedPages>
class BatchedPreOLayer;
template <int TP, int Rows, bool FixedPages>
class BatchedPostOLayer;
class BatchedTailLayer;
class BatchedRows4TailLayer;

enum class LayerInput {
  Full,
  GatedAttention,
  OutputProjection,
  PostAttentionNorm,
  PostAttentionNormRows4
};

inline int expert_choice(const int* experts, int row, int expert) {
  for (int k = 0; k < Top; ++k)
    if (experts[row * Top + k] == expert) return k;
  return -1;
}

inline bool first_expert(const int* experts, int position) {
  for (int i = 0; i < position; ++i)
    if (experts[i] == experts[position]) return false;
  return true;
}

template <
    int TP,
    int Rows,
    bool Dense,
    bool IgnorePadding = false,
    bool TiledQkv = false,
    bool FastVerify = false,
    bool FixedVerifyPages = false,
    bool CombinedOrdinaryM8 = false,
    LayerInput Input = LayerInput::Full>
void launch(sycl::queue& queue, Weights weights, Workspace work, Step step) {
  static_assert(!IgnorePadding || ((Rows == 4 || Rows == 8) && !Dense));
  constexpr int Q = 32 / TP, KV = 2 / TP, QDim = Q * D;
  constexpr int QKV = 2 * QDim + 2 * KV * D;
  constexpr int I = 512 / TP, DenseI = 8192 / TP;
  // Router partials reuse the dead attention buffer, including in the
  // smaller non-reference-attention allocation accepted by the host ABI.
  static_assert(Groups * Rows * E <= MaxRows * Groups * Q * (D + 2));
  constexpr int IS = DenseI > (Top + 1) * I ? DenseI : (Top + 1) * I;
  constexpr int KParts = 8;
#ifdef MINICPM5_MONO_DPAS_ENABLED
  #ifdef MINICPM5_MONO_DPAS_ROW1_ENABLED
  constexpr bool Matrix = true;
  #else
  constexpr bool Matrix = Rows > 1;
  #endif
#else
  constexpr bool Matrix = false;
#endif
#ifdef MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED
  constexpr bool LocalProjection = Matrix;
  constexpr bool LocalSharedActivation = Matrix && !Dense;
#else
  constexpr bool LocalProjection = false;
  constexpr bool LocalSharedActivation = false;
#endif
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
  constexpr bool LocalDown = LocalProjection;
  constexpr bool LocalSelection = Matrix;
#else
  constexpr bool LocalDown = LocalProjection && Dense;
  constexpr bool LocalSelection = false;
#endif
#ifdef MINICPM5_MONO_FP16
  static_assert(!FastVerify || (Rows == 8 && IgnorePadding));
  constexpr bool VerifyBatch = FastVerify && LocalProjection && LocalSelection;
#else
  constexpr bool VerifyBatch = false;
#endif
  constexpr bool FromPreO = Input == LayerInput::GatedAttention;
  constexpr bool FromPostO = Input == LayerInput::OutputProjection;
  constexpr bool AlternateGuPartial = FromPreO || FromPostO;
  constexpr int AdditionalGuPartial = AlternateGuPartial ? Rows * H : 0;
  constexpr bool TailRows4 = Input == LayerInput::PostAttentionNormRows4;
  constexpr bool TailOnly = TailRows4 || Input == LayerInput::PostAttentionNorm;
  static_assert(!FixedVerifyPages || VerifyBatch);
  static_assert(!TailOnly || TailRows4 || CombinedOrdinaryM8);
  static_assert(
      !TailRows4 ||
      (TP == 1 && Rows == 4 && !Dense && IgnorePadding && Matrix && !TiledQkv &&
       !FastVerify && !FixedVerifyPages && !CombinedOrdinaryM8));
  static_assert(
      !(FromPreO || FromPostO) ||
      (TP == 2 && Rows == 8 && !Dense && FixedVerifyPages && VerifyBatch &&
       !CombinedOrdinaryM8));
  static_assert(
      !CombinedOrdinaryM8 ||
      (TP == 1 && Rows == 8 && !Dense && FixedVerifyPages));
  constexpr bool InlineKV = Rows == 1 || VerifyBatch;
  constexpr bool B1Tail = TP == 2 && Rows == 1 && LocalProjection;
  constexpr bool B1Prt = B1Tail && !Dense;
  static_assert(!TiledQkv || (Rows == 1 && Matrix));

#ifdef MINICPM5_MONO_FP16
  // A single device acquire is propagated to every payload reader by the
  // global/local WG barrier in grid_wait, including multirow consumers.
  constexpr bool GroupAcquire = LocalProjection;
#else
  constexpr bool GroupAcquire = B1Tail;
#endif
  // No in-kernel consumer follows AR2. Kernel completion orders all output
  // and counter stores before the next command on the validated in-order
  // queue. Phase9 is the last global handoff in this specialization.
#ifdef MINICPM5_MONO_FP16
  constexpr bool KernelExit = !Dense && GroupAcquire && LocalDown &&
#else
  constexpr bool KernelExit = B1Prt && GroupAcquire && LocalDown &&
#endif
                              LocalSharedActivation && LocalSelection;
#ifdef MINICPM5_MONO_FP16
  constexpr bool BatchedPrPreload = Rows == 8 && LocalProjection;
  constexpr bool BatchedQkvPreload = Rows == 8 && LocalProjection;
#else
  constexpr bool BatchedPrPreload = TP == 2 && Rows == 4 && LocalProjection;
  constexpr bool BatchedQkvPreload = BatchedPrPreload;
#endif
#ifdef MINICPM5_MONO_FP16
  constexpr int BatchedProjectionChunks = 8;
#else
  constexpr int BatchedProjectionChunks = 16;
#endif
#ifdef MINICPM5_MONO_FP16
  constexpr int OutputProjectionChunks = QDim / (Subgroups / 2 * 16);
#else
  constexpr int OutputProjectionChunks = BatchedProjectionChunks;
#endif
  static_assert(!B1Prt || Groups * (E + 512) <= MaxRows * Groups * Q * (D + 2));
#ifdef MINICPM5_MONO_FP16
  // Tiled QKV reuses normalization storage for sixteen K partials per tile.
  // TP1 assigns seventeen tiles per WG, TP2 assigns up to nine.
  constexpr int QkvTilesPerGroup = (QKV / 16 + Groups - 1) / Groups;
  constexpr int CanonicalQkvExtra =
      TiledQkv ? QkvTilesPerGroup * 16 * 16 * 2 - H : 0;
#else
  constexpr int CanonicalQkvExtra = TP == 2 && Rows == 1 ? 256 : 0;
#endif
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<scalar_t, 1> normed(
        Rows * H + CanonicalQkvExtra + AdditionalGuPartial, cgh);
    // Reuse norm scalars for the B1 rotary table, then the chunk plan.
    constexpr int NormWords = Rows == 1 ? D + 4 : Rows * (Subgroups + 1) + 3;
    sycl::local_accessor<float, 1> norms(NormWords, cgh);
#if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) ||  \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED) || \
    defined(MINICPM5_MONO_REFERENCE_MOE_ENABLED)
    constexpr int QWords = Rows * KV * 2 * (D / 16) * 16 * 4;
    constexpr int MergeWords =
        Rows * AttentionChunks<TP, Rows>::Slots + Rows +
        (Rows == 1 ? AttentionChunks<TP, Rows>::Slots + Subgroups + 1 : 0);
    constexpr int AttentionWords = QWords > MergeWords ? QWords : MergeWords;
    constexpr int PackedWords = QDim / 16 * Rows * 8;
    // B1 routed down reduces two partitions for every expert in this SLM.
    constexpr int DownWords = Rows == 1 && !Dense ? Subgroups * 64 : 0;
    // Wide gate/up keeps four output tiles per subgroup for every row.
    constexpr int GateUpWords =
        !Dense && Rows <= 2 ? Subgroups * 4 * Rows * 16 : 0;
    constexpr int MoeWords = DownWords > GateUpWords ? DownWords : GateUpWords;
    constexpr int ProjectionWords =
        PackedWords > MoeWords ? PackedWords : MoeWords;
    constexpr int StandardWords =
        AttentionWords > ProjectionWords ? AttentionWords : ProjectionWords;
  #if defined(MINICPM5_MONO_FP16) && \
      defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED)
    constexpr int KvStagingWords =
        FixedVerifyPages && TP == 1 && Rows == 8 ? 4 * SharedKvTileWords : 0;
  #else
    constexpr int KvStagingWords = 0;
  #endif
    constexpr int TemporaryWords =
        StandardWords > KvStagingWords ? StandardWords : KvStagingWords;
    // Only TP1 Rows4 TailOnly uses the second 32 KiB GU partial plane.
    constexpr int AlternateGuWords = TailRows4 ? Subgroups * Rows * 64 : 0;
    sycl::local_accessor<UInt4, 1> attention_slm(
        (TemporaryWords + AlternateGuWords + 3) / 4, cgh);
#endif
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
    sycl::local_accessor<int, 1> selected(Rows * Top, cgh);
    sycl::local_accessor<int, 1> union_experts(Rows * Top, cgh);
    sycl::local_accessor<int, 1> union_choices(Rows * Top * Rows, cgh);
    sycl::local_accessor<int, 1> union_count(1, cgh);
#endif
    sycl::ext::oneapi::experimental::properties properties{
        sycl::ext::intel::experimental::grf_size<256>};
    using BaseKernel = std::conditional_t<
        TiledQkv,
        TiledQkvLayer<TP, Rows, Dense>,
        std::conditional_t<
            IgnorePadding,
            PaddedLayer<TP, Rows, Dense>,
            Layer<TP, Rows, Dense>>>;
    using StandardKernel = std::conditional_t<
        VerifyBatch,
        BatchedVerifyLayer<TP, Rows, FixedVerifyPages>,
        BaseKernel>;
    using FullKernel = std::conditional_t<
        CombinedOrdinaryM8,
        BatchedBestLayer<TP, Rows, FixedVerifyPages>,
        StandardKernel>;
    using PreOKernel = std::conditional_t<
        FromPreO,
        BatchedPreOLayer<TP, Rows, FixedVerifyPages>,
        FullKernel>;
    using AttentionTailKernel = std::conditional_t<
        FromPostO,
        BatchedPostOLayer<TP, Rows, FixedVerifyPages>,
        PreOKernel>;
    using SplitKernel =
        std::conditional_t<TailOnly, BatchedTailLayer, AttentionTailKernel>;
    using Kernel =
        std::conditional_t<TailRows4, BatchedRows4TailLayer, SplitKernel>;
    cgh.parallel_for<Kernel>(
        sycl::nd_range<1>(Groups * Local, Local),
        properties,
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const int lid = item.get_local_linear_id();
          const int wg = item.get_group_linear_id();
          const int global = wg * Local + lid;
          auto sg = item.get_sub_group();
          const int lane = sg.get_local_linear_id();
          // Spread even a short DPAS tile range over all resident groups.
          // Group-major IDs put the 128 output-projection tiles on only four
          // of the 32 groups, leaving the other Xe cores waiting at the grid
          // hand-off. Subgroup-major IDs retain unique ownership and the same
          // per-output accumulation order.
          const int sid = sg.get_group_linear_id() * Groups + wg;
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
          float routed[2];
#endif
          const uint32_t call = epoch_load(step.control + Epoch) + 1u;
          const auto* x =
              TailOnly
                  ? step.hidden
                  : normed.template get_multi_ptr<sycl::access::decorated::no>()
                        .get();
#if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) ||  \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED) || \
    defined(MINICPM5_MONO_REFERENCE_MOE_ENABLED)
          auto* reference_storage = reinterpret_cast<uint32_t*>(
              attention_slm
                  .template get_multi_ptr<sycl::access::decorated::no>()
                  .get());
          auto* reduction_storage = reinterpret_cast<float*>(
              normed.template get_multi_ptr<sycl::access::decorated::no>()
                  .get());
#endif
          sycl::uint4 b1_output_raw[2][B1Tail ? 8 : 1];
          sycl::uint4
              batched_output_raw[BatchedPrPreload ? OutputProjectionChunks : 1];
          if constexpr (FromPreO || FromPostO || TailOnly) {
            // The preceding Graph nodes have published the activation.
            // Slots only provide a padding mask for these split inputs.
            if (global < Rows && step.slots[global] < -1)
              fail(
                  step.control, 3, TailOnly ? 6 : 5, call, global, step.health);
          } else {
            for (int r = global; r < Rows; r += Groups * Local) {
              const int64_t slot = step.slots[r];
              const int length = step.lengths[r];
              if (!row_metadata_valid(
                      length,
                      step.positions[r],
                      slot,
                      step.block_size,
                      step.cache_blocks,
                      step.block_table_cols,
                      step.rope_cache_rows)) {
                fail(step.control, 3, 12, call, global, step.health);
                continue;
              }
              if (slot < 0) continue;
#ifndef MINICPM5_MONO_REFERENCE_RUNTIME_ENABLED
              for (int b = 0;
                   b < (length + step.block_size - 1) / step.block_size;
                   ++b) {
                const int block = step.blocks[r * step.block_table_stride + b];
                if (block < 0 || block >= step.cache_blocks)
                  fail(step.control, 3, 12, call, global, step.health);
              }
#endif
              const int current = step.blocks
                                      [r * step.block_table_stride +
                                       (length - 1) / step.block_size];
              if (slot != int64_t(current) * step.block_size +
                              (length - 1) % step.block_size)
                fail(step.control, 3, 12, call, global, step.health);
              for (int other = 0; other < r; ++other)
                if (step.slots[other] == slot)
                  fail(step.control, 3, 12, call, global, step.health);
            }
#ifdef MINICPM5_MONO_REFERENCE_RUNTIME_ENABLED
            // Page entries are independent. Keep row/slot checks above, and
            // distribute the table scan instead of serializing it on each row.
            const bool shared_pages =
                VerifyBatch && step.block_table_stride == 0;
            for (int r = 0; r < (shared_pages ? 1 : Rows); ++r) {
              int length = step.lengths[r];
              if (shared_pages) {
                // Row validation has its own checks above. Bound this read
                // even while another work-group is reporting a bad row.
                length = 0;
  #pragma unroll
                for (int i = 0; i < Rows; ++i)
                  if (step.slots[i] >= 0)
                    length = sycl::max(length, step.lengths[i]);
              } else if (
                  step.slots[r] < 0 || !row_metadata_valid(
                                           length,
                                           step.positions[r],
                                           step.slots[r],
                                           step.block_size,
                                           step.cache_blocks,
                                           step.block_table_cols,
                                           step.rope_cache_rows)) {
                continue;
              }
              const int pages =
                  shared_pages
                      ? int(sycl::min(
                            (int64_t(length) + step.block_size - 1) /
                                step.block_size,
                            step.block_table_cols))
                      : (length + step.block_size - 1) / step.block_size;
              for (int b = global; b < pages; b += Groups * Local) {
                const int block = step.blocks[r * step.block_table_stride + b];
                if (block < 0 || block >= step.cache_blocks)
                  fail(step.control, 3, 12, call, global, step.health);
              }
            }
#endif
            if (!grid_barrier_handoff<GroupAcquire>(
                    item, step.control, 12, call, step.health))
              return;
#ifdef MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED
            if (wg == 0 && lid < Rows * KV * 2)
              Flag(step.control[Epoch + 16 + lid]).store(0);
#endif
            sycl::uint4 batched_qkv_raw
                [BatchedQkvPreload ? BatchedProjectionChunks : 1];
            if constexpr (BatchedQkvPreload)
              reference_batched_projection_preload(
                  item, weights.qkv, batched_qkv_raw);
            reference_norm<Rows, true, TP == 2>(
                item,
                step.hidden,
                step.residual,
                weights.input_norm,
                normed.template get_multi_ptr<sycl::access::decorated::no>()
                    .get(),
                norms.template get_multi_ptr<sycl::access::decorated::no>()
                    .get());
            reference_wg_bar(item.get_group());
            {
              if constexpr (Matrix) {
#ifdef MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED
  #ifdef MINICPM5_MONO_FP16
                if constexpr (TiledQkv)
                  reference_fp16_tiled_qkv<QKV>(
                      item,
                      weights.qkv,
                      x,
                      work.qkv,
                      weights.qkv_scale[0],
                      reference_storage,
                      reduction_storage);
                else
  #endif
                  reference_projection_nk<
                      Rows,
                      H,
                      QKV,
                      H,
                      -1,
                      false,
                      BatchedQkvPreload,
                      BatchedQkvPreload,
  #ifdef MINICPM5_MONO_FP16
                      false>(
  #else
                    TP == 2 && Rows == 1,
                    TiledQkv>(
  #endif
                      item,
                      weights.qkv,
                      x,
                      H,
                      work.qkv,
                      QKV,
                      weights.qkv_scale[0],
                      reference_storage,
                      reduction_storage,
                      nullptr,
                      0,
                      nullptr,
                      batched_qkv_raw);
#else
                int offsets[Rows];
                for (int r = 0; r < Rows; ++r)
                  offsets[r] = r * H;
                for (int tile = sid; tile < QKV / 16;
                     tile += Groups * Subgroups) {
                  const auto out = projection_nk<Rows, H>(
                      sg, weights.qkv + size_t(tile * 16) * H, x, offsets);
                  for (int r = 0; r < Rows; ++r)
                    work.qkv[r * QKV + tile * 16 + lane] =
                        scalar_t(out[r] * weights.qkv_scale[0]);
                }
#endif
              } else {
#ifdef MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED
                reference_gemv_nk<H, QKV>(
                    item, weights.qkv, x, work.qkv, weights.qkv_scale[0]);
#else
                for (int n = sid; n < QKV; n += Groups * Subgroups) {
                  float out[Rows];
                  projection<Rows, H>(
                      sg, weights.qkv + size_t(n) * H, x, H, out);
                  if (lane == 0)
                    for (int r = 0; r < Rows; ++r)
                      work.qkv[r * QKV + n] =
                          scalar_t(out[r] * weights.qkv_scale[0]);
                }
#endif
              }
              if constexpr (Rows == 1) {
                grid_signal(item, step.control, 0, call, step.health);
                // Only rotary owners need this WG-local table. Its construction
                // does not consume QKV and can overlap the global hand-off.
                if (wg * Local < (Q + KV) * (D / 2) && lid < D / 2) {
                  scalar_t cs, sn;
                  rotary_pair(step, 0, lid, cs, sn);
                  norms[lid] = float(cs);
                  norms[D / 2 + lid] = float(sn);
                }
                if (!grid_wait<(Q + KV) * (D / 2), GroupAcquire>(
                        item, step.control, 0, call, step.health))
                  return;
              } else {
                if (!grid_barrier_handoff<GroupAcquire>(
                        item, step.control, 0, call, step.health))
                  return;
              }

              // One thread owns both halves of a NeoX rotary pair.
              for (int pair = global; pair < Rows * (Q + KV) * (D / 2);
                   pair += Groups * Local) {
                const int r = pair / ((Q + KV) * (D / 2));
                const int h = pair / (D / 2) % (Q + KV), d = pair % (D / 2);
                const int base =
                    r * QKV + (h < Q ? h * 2 * D : 2 * QDim + (h - Q) * D);
                scalar_t cs, sn;
                if constexpr (Rows == 1) {
                  cs = scalar_t(norms[d]);
                  sn = scalar_t(norms[D / 2 + d]);
                } else {
                  rotary_pair(step, r, d, cs, sn);
                }
                const float a = float(work.qkv[base + d]);
                const float b = float(work.qkv[base + d + D / 2]);
                const scalar_t first = scalar_t(a * float(cs) - b * float(sn));
                const scalar_t second = scalar_t(b * float(cs) + a * float(sn));
                work.qkv[base + d] = first;
                work.qkv[base + d + D / 2] = second;
                if constexpr (InlineKV) {
                  // The K-pair owner also owns these cache elements. V is ready
                  // from phase0; no cross-WG rotary-to-cache wait is needed.
                  const int64_t slot = step.slots[r];
                  if (h >= Q && slot >= 0) {
                    const int kvh = h - Q;
                    const int64_t offset =
                        (slot / step.block_size) * step.cache_block_stride +
                        kvh * step.cache_head_stride +
                        (slot % step.block_size) * step.cache_token_stride;
                    step.cache[offset + d] = first;
                    step.cache[offset + d + D / 2] = second;
                    const int value = r * QKV + 2 * QDim + KV * D + kvh * D + d;
                    step.cache[offset + D + d] = work.qkv[value];
                    step.cache[offset + D + d + D / 2] =
                        work.qkv[value + D / 2];
                  }
                }
              }
              if constexpr (InlineKV) {
              } else {
                if (!grid_barrier_handoff<GroupAcquire>(
                        item, step.control, 1, call, step.health))
                  return;
                for (int v = global; v < Rows * KV * D; v += Groups * Local) {
                  const int r = v / (KV * D), h = v / D % KV, d = v % D;
                  const int64_t slot = step.slots[r];
                  if (slot < 0) continue;
                  const int64_t offset =
                      (slot / step.block_size) * step.cache_block_stride +
                      h * step.cache_head_stride +
                      (slot % step.block_size) * step.cache_token_stride;
                  step.cache[offset + d] =
                      work.qkv[r * QKV + 2 * QDim + h * D + d];
                  step.cache[offset + D + d] =
                      work.qkv[r * QKV + 2 * QDim + KV * D + h * D + d];
                }
              }
              if (!grid_barrier_handoff<GroupAcquire>(
                      item, step.control, 2, call, step.health))
                return;

#ifdef MINICPM5_MONO_DPAS_ENABLED
  #ifdef MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED
              auto* chunk_plan = reinterpret_cast<int*>(
                  norms.template get_multi_ptr<sycl::access::decorated::no>()
                      .get());
              reference_attention<TP, Rows, true, FixedVerifyPages>(
                  item, work, step, reference_storage, chunk_plan);
  #else
              matrix_attention<TP, Rows>(item, work, step);
  #endif
#else
              // Context partitions are independent across all resident groups.
              for (int r = 0; r < Rows; ++r) {
                for (int h = sg.get_group_linear_id(); h < Q; h += Subgroups) {
                  const int length = step.slots[r] < 0 ? 0 : step.lengths[r];
                  const int chunk = (length + Groups - 1) / Groups;
                  const int begin = wg * chunk,
                            end = sycl::min(begin + chunk, length);
                  float maximum = -INFINITY, denominator = 0.f;
                  float values[8] = {};
                  const int kvh = h / (Q / KV);
                  for (int t = begin; t < end; ++t) {
                    const int block =
                        step.blocks
                            [r * step.block_table_stride + t / step.block_size];
                    const int64_t offset =
                        int64_t(block) * step.cache_block_stride +
                        kvh * step.cache_head_stride +
                        (t % step.block_size) * step.cache_token_stride;
                    float dot = 0.f;
  #pragma unroll
                    for (int d = 0; d < 8; ++d)
                      dot +=
                          float(work.qkv[r * QKV + h * 2 * D + lane * 8 + d]) *
                          float(step.cache[offset + lane * 8 + d]);
                    dot =
                        sycl::reduce_over_group(sg, dot, sycl::plus<float>()) *
                        0.08838834764831844f;
                    const float next = sycl::max(maximum, dot);
                    const float old = sycl::exp(maximum - next),
                                p = sycl::exp(dot - next);
  #pragma unroll
                    for (int d = 0; d < 8; ++d)
                      values[d] =
                          values[d] * old +
                          p * float(step.cache[offset + D + lane * 8 + d]);
                    denominator = denominator * old + p;
                    maximum = next;
                  }
                  const size_t offset =
                      ((size_t(r) * Groups + wg) * Q + h) * (D + 2);
  #pragma unroll
                  for (int d = 0; d < 8; ++d)
                    work.attention_partial[offset + lane * 8 + d] = values[d];
                  if (lane == 0) {
                    work.attention_partial[offset + D] = maximum;
                    work.attention_partial[offset + D + 1] = denominator;
                  }
                }
              }
#endif
              if constexpr (B1Tail) {
                // mono_layers.hpp C_B signal -> full O preload -> wait ->
                // merge.
                grid_signal(item, step.control, 3, call, step.health);
                reference_b1_output_preload(item, weights.out, b1_output_raw);
                if (!grid_wait<Groups * Local, GroupAcquire>(
                        item, step.control, 3, call, step.health))
                  return;
              } else if constexpr (BatchedPrPreload && !CombinedOrdinaryM8) {
                // PR signal -> full O preload -> wait -> attention merge.
                grid_signal(item, step.control, 3, call, step.health);
                reference_batched_projection_preload<
                    OutputProjectionChunks,
                    QDim>(item, weights.out, batched_output_raw);
                if (!grid_wait<Groups * Local, GroupAcquire>(
                        item, step.control, 3, call, step.health))
                  return;
              } else if (!grid_barrier_handoff<GroupAcquire>(
                             item, step.control, 3, call, step.health)) {
                return;
              }
#ifdef MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED
              reference_attention_merge<TP, Rows>(
                  item,
                  work,
                  step,
                  reinterpret_cast<float*>(reference_storage),
                  chunk_plan);
#else
              for (int v = global; v < Rows * QDim; v += Groups * Local) {
                const int r = v / QDim, h = v / D % Q, d = v % D;
                float maximum = -INFINITY, denominator = 0.f, numerator = 0.f;
                for (int g = 0; g < Groups; ++g)
                  maximum = sycl::max(
                      maximum,
                      work.attention_partial
                          [((size_t(r) * Groups + g) * Q + h) * (D + 2) + D]);
                for (int g = 0; g < Groups; ++g) {
                  const size_t off =
                      ((size_t(r) * Groups + g) * Q + h) * (D + 2);
                  const float p =
                      maximum == -INFINITY
                          ? 0.f
                          : sycl::exp(
                                work.attention_partial[off + D] - maximum);
                  numerator += p * work.attention_partial[off + d];
                  denominator += p * work.attention_partial[off + D + 1];
                }
                const float gate = float(work.qkv[r * QKV + h * 2 * D + D + d]);
                const scalar_t value =
                    scalar_t(denominator > 0 ? numerator / denominator : 0.f);
                work.attention[v] = scalar_t(
                    float(value) *
                    float(scalar_t(1.f / (1.f + sycl::exp(-gate)))));
              }
#endif
            }
            // Output weights are independent of the attention merge. Load them
            // after publishing this WG, before waiting for the other producers.
            if constexpr (LocalProjection)
              grid_signal(item, step.control, 4, call, step.health);
            else if (!grid_barrier_handoff<GroupAcquire>(
                         item, step.control, 4, call, step.health))
              return;
          }
          sycl::uint4 b1_router_raw[B1Prt ? 4 : 1],
              b1_shared_raw[B1Prt ? 4 : 1];
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
          UInt8 router_first[2];
#endif
          if constexpr (TailOnly) {
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
            reference_router_preload<Rows>(item, weights, router_first);
#endif
          } else {
            if constexpr (!FromPostO) {
              const auto* attention = FromPreO ? step.hidden : work.attention;
              if constexpr (B1Prt) {
                reference_b1_router_preload(item, weights, b1_router_raw);
                reference_b1_shared_preload(item, weights, b1_shared_raw);
              }
#ifdef MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED
              if constexpr (CombinedOrdinaryM8 || FromPreO) {
                // Weight prefetch precedes input packing and the optional wait.
                reference_batched_projection_preload<
                    OutputProjectionChunks,
                    QDim>(item, weights.out, batched_output_raw);
              }
              if constexpr (B1Tail) {
                if (!reference_b1_output_gemv(
                        item,
                        b1_output_raw,
                        attention,
                        weights.out_scale[0],
                        reference_storage,
                        norms
                            .template get_multi_ptr<
                                sycl::access::decorated::no>()
                            .get(),
                        step,
                        call))
                  return;
              } else if constexpr (Matrix) {
                if (!reference_projection_nk<
                        Rows,
                        QDim,
                        H,
                        QDim,
                        FromPreO ? -1 : 4,
                        false,
                        BatchedPrPreload,
                        BatchedPrPreload>(
                        item,
                        weights.out,
                        attention,
                        QDim,
                        step.res_out,
                        H,
                        weights.out_scale[0],
                        reference_storage,
                        reduction_storage,
                        step.control,
                        call,
                        step.health,
                        batched_output_raw))
                  return;
              } else {
                reference_gemv_nk<QDim, H>(
                    item,
                    weights.out,
                    attention,
                    step.res_out,
                    weights.out_scale[0]);
              }
#else
              for (int job = sid; job < H / (Matrix ? 16 : 1);
                   job += Groups * Subgroups) {
                const int n = Matrix ? job * 16 + lane : job;
                float out[Rows];
                if constexpr (Matrix) {
                  int offsets[Rows];
                  for (int r = 0; r < Rows; ++r)
                    offsets[r] = r * QDim;
                  const auto result = projection_nk<Rows, QDim>(
                      sg,
                      weights.out + size_t(job * 16) * QDim,
                      attention,
                      offsets);
                  for (int r = 0; r < Rows; ++r)
                    out[r] = result[r];
                } else {
                  projection<Rows, QDim>(
                      sg, weights.out + size_t(n) * QDim, attention, QDim, out);
                }
                if (Matrix || lane == 0)
                  for (int r = 0; r < Rows; ++r) {
                    step.res_out[r * H + n] =
                        scalar_t(out[r] * weights.out_scale[0]);
                  }
              }
#endif
              // Pair consumers read columns produced by other work-items. The
              // reference assigns all 64 columns to this WG; fallback
              // projections scatter ownership across WGs and still require a
              // grid hand-off.
              if constexpr (LocalProjection) {
                if constexpr (B1Tail)
                  reference_wg_bar(item.get_group());
                else
                  sycl::group_barrier(item.get_group());
              } else {
                if (!grid_barrier_handoff<GroupAcquire>(
                        item, step.control, 13, call, step.health))
                  return;
              }
            }
            // The reference projection assigns this WG its contiguous 64
            // columns. Each producer exchanges them directly with its peer WG.
            if constexpr (VerifyBatch && TP == 2) {
              const int r = lid / (H / Groups / 2);
              const int col = wg * (H / Groups) + 2 * (lid % (H / Groups / 2));
              const int n = r * H + col;
              float first = 0.f, second = 0.f;
              if (lid < MailboxPairs) {
                const auto* projection = FromPostO ? step.hidden : step.res_out;
                first = float(projection[n]);
                second = float(projection[n + 1]);
              }
              if (!reduce_fp16_packet<Rows>(item, first, second, 0, call, step))
                return;
              if (lid < MailboxPairs) {
                const float values[2] = {first, second};
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                  const float residual =
                      FromPreO || FromPostO
                          ? float(step.residual[n + j])
                          : float(scalar_t(
                                float(step.hidden[n + j]) +
                                (step.residual ? float(step.residual[n + j])
                                               : 0.f)));
                  step.res_out[n + j] =
                      scalar_t(float(scalar_t(values[j])) + residual);
                }
              }
            } else if (lid < Rows * (H / Groups / 2)) {
              const int r = lid / (H / Groups / 2);
              const int col = wg * (H / Groups) + 2 * (lid % (H / Groups / 2));
              const int n = r * H + col;
              float first, second;
              if constexpr (B1Tail) {
                first = norms[2 * lid];
                second = norms[2 * lid + 1];
              } else {
                first = float(step.res_out[n]);
                second = float(step.res_out[n + 1]);
              }
              reduce_pair<TP>(first, second, 0, r, col, call, step);
              const float values[2] = {first, second};
#pragma unroll
              for (int j = 0; j < 2; ++j) {
                const float residual = float(scalar_t(
                    float(step.hidden[n + j]) +
                    (step.residual ? float(step.residual[n + j]) : 0.f)));
                step.res_out[n + j] =
                    scalar_t(float(scalar_t(values[j])) + residual);
              }
            }
            if constexpr (B1Tail)
              grid_signal<H / Groups / 2>(
                  item, step.control, 5, call, step.health);
            else
              grid_signal(item, step.control, 5, call, step.health);
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
            if constexpr (!Dense && Matrix && !B1Prt)
              reference_router_preload<Rows>(item, weights, router_first);
#endif
            if (!grid_wait<Groups * Local, GroupAcquire>(
                    item, step.control, 5, call, step.health))
              return;
            reference_norm<Rows, false, TP == 2>(
                item,
                step.res_out,
                nullptr,
                weights.post_norm,
                normed.template get_multi_ptr<sycl::access::decorated::no>()
                    .get(),
                norms.template get_multi_ptr<sycl::access::decorated::no>()
                    .get());
            reference_wg_bar(item.get_group());
            for (int r = 0; r < Rows; ++r)
              for (int n = wg * (H / Groups) + lid; n < (wg + 1) * (H / Groups);
                   n += Local)
                work.normed[r * H + n] = normed[r * H + n];
            reference_wg_bar(item.get_group());
          }
          if constexpr (!Dense) {
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
            if constexpr (Matrix) {
              if constexpr (B1Prt)
                reference_b1_router_partial(item, work, x, b1_router_raw);
              else
                reference_router_partial<Rows>(
                    item, weights, work, x, reference_storage, router_first);
            } else
#endif
            {
              for (int e = sid; e < E; e += Groups * Subgroups) {
                float logits[Rows] = {};
                for (int n = lane; n < H; n += 16) {
                  const float w = float(weights.router[e * H + n]);
#pragma unroll
                  for (int r = 0; r < Rows; ++r)
                    logits[r] += w * float(x[r * H + n]);
                }
                for (int r = 0; r < Rows; ++r) {
                  const float v = sycl::reduce_over_group(
                      sg, logits[r], sycl::plus<float>());
                  if (lane == 0) {
                    if (!sycl::isfinite(v) || !sycl::isfinite(weights.bias[e]))
                      fail(step.control, 4, 7, call, global, step.health);
                    work.router[r * E + e] = 1.f / (1.f + sycl::exp(-v));
                  }
                }
              }
            }
          }
          constexpr int SharedI = Dense ? DenseI : I;
          const uint8_t* shared = Dense ? weights.w13 : weights.shared13;
          const float scale =
              Dense ? weights.w13_scale[0] : weights.shared13_scale[0];
#ifdef MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED
          const auto shared_gateup = [&]() {
            reference_shared_gateup<Rows, SharedI>(
                item,
                shared,
                x,
                H,
                work.intermediate + Top * I,
                IS,
                scale,
                reference_storage,
                reduction_storage);
          };
          if constexpr (B1Prt) {
            reference_b1_shared_partial(item, work, x, b1_shared_raw);
          } else if constexpr (LocalSharedActivation) {
            if constexpr (!FixedVerifyPages) shared_gateup();
          } else if constexpr (Matrix) {
            reference_projection_nk<Rows, H, 2 * SharedI>(
                item,
                shared,
                x,
                H,
                work.expert_out,
                (Top + 1) * H,
                scale,
                reference_storage,
                reduction_storage);
          } else {
            reference_gemv_nk<H, 2 * SharedI>(
                item, shared, x, work.expert_out, scale);
          }
#else
          for (int job = sid; job < SharedI / (Matrix ? 16 : 1);
               job += Groups * Subgroups) {
            const int n = Matrix ? job * 16 + lane : job;
            float gate[Rows], up[Rows];
            if constexpr (Matrix) {
              int offsets[Rows];
              for (int r = 0; r < Rows; ++r)
                offsets[r] = r * H;
              const auto g = projection_nk<Rows, H>(
                  sg, shared + size_t(job * 16) * H, x, offsets);
              const auto u = projection_nk<Rows, H>(
                  sg, shared + size_t(SharedI + job * 16) * H, x, offsets);
              for (int r = 0; r < Rows; ++r) {
                gate[r] = g[r];
                up[r] = u[r];
              }
            } else {
              projection<Rows, H>(sg, shared + size_t(n) * H, x, H, gate);
              projection<Rows, H>(
                  sg, shared + size_t(SharedI + n) * H, x, H, up);
            }
            if (Matrix || lane == 0)
              for (int r = 0; r < Rows; ++r) {
                const float g = float(scalar_t(gate[r] * scale)),
                            u = float(scalar_t(up[r] * scale));
                work.intermediate[r * IS + (Dense ? 0 : Top * I) + n] =
                    scalar_t(g / (1.f + sycl::exp(-g)) * u);
              }
          }
#endif
          if (!grid_barrier_handoff<GroupAcquire>(
                  item, step.control, 6, call, step.health))
            return;
          if constexpr (B1Prt) reference_b1_shared_finish(item, weights, work);
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
          if constexpr (!Dense && Matrix) {
            if (!reference_router_finish < Rows,
                TP == 2 && Rows == 1,
                VerifyBatch,
                TailRows4 > (item,
                             weights,
                             work,
                             step,
                             call,
                             reinterpret_cast<float*>(reference_storage)))
              return;
          }
#endif
#ifdef MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED
          if constexpr (!LocalSharedActivation) {
            for (int v = global; v < Rows * SharedI; v += Groups * Local) {
              const int r = v / SharedI, n = v % SharedI;
              const float g = float(work.expert_out[r * (Top + 1) * H + n]);
              const float u =
                  float(work.expert_out[r * (Top + 1) * H + SharedI + n]);
              work.intermediate[r * IS + (Dense ? 0 : Top * I) + n] =
                  scalar_t(g / (1.f + sycl::exp(-g)) * u);
            }
            if (!grid_barrier_handoff<GroupAcquire>(
                    item, step.control, 14, call, step.health))
              return;
          }
#endif
          if constexpr (!Dense) {
            bool selection_ok = true;
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
            auto* local_selected =
                selected.template get_multi_ptr<sycl::access::decorated::no>()
                    .get();
            auto* local_routing =
                norms.template get_multi_ptr<sycl::access::decorated::no>()
                    .get();
            static_assert(NormWords >= Rows * Top);
#endif
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
            if constexpr (VerifyBatch) {
              static_assert(TemporaryWords >= Rows * E + 4 * RouterSortSize);
              if (wg < Rows)
                reference_router_select_row(
                    item,
                    weights,
                    work,
                    step,
                    call,
                    reinterpret_cast<float*>(reference_storage) + wg * E,
                    reinterpret_cast<uint64_t*>(reference_storage + Rows * E));
            } else
#endif
                if ((LocalSelection || wg == 0) &&
                    sg.get_group_linear_id() < Rows) {
              const int r = sg.get_group_linear_id();
              {
                const float* scores = work.router;
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
                if constexpr (Matrix)
                  scores = reinterpret_cast<const float*>(reference_storage);
#endif
                float selection[10];
#pragma unroll
                for (int j = 0; j < 10; ++j)
                  selection[j] = scores[r * E + lane * 10 + j] +
                                 weights.bias[lane * 10 + j];
                float total = 0.f;
                for (int k = 0; k < Top; ++k) {
                  float best = -INFINITY;
                  int index = E;
#pragma unroll
                  for (int j = 0; j < 10; ++j)
                    if (selection[j] > best) {
                      best = selection[j];
                      index = lane * 10 + j;
                    }
                  const float value =
                      sycl::reduce_over_group(sg, best, sycl::maximum<float>());
                  const int chosen = sycl::reduce_over_group(
                      sg, best == value ? index : E, sycl::minimum<int>());
                  if (chosen >= E) {
                    fail(step.control, 4, 7, call, global, step.health);
                    selection_ok = false;
                    break;
                  }
#pragma unroll
                  for (int j = 0; j < 10; ++j)
                    if (lane * 10 + j == chosen) selection[j] = -INFINITY;
                  const float unbiased = scores[r * E + chosen];
                  total += unbiased;
                  if (lane == 0) {
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
                    if constexpr (LocalSelection && !VerifyBatch) {
                      local_selected[r * Top + k] = chosen;
                      local_routing[r * Top + k] = unbiased;
                    } else
#endif
                    {
                      work.experts[r * Top + k] = chosen;
                      work.routing[r * Top + k] = unbiased;
                    }
                  }
                }
                if (!(total > 0.f) || !sycl::isfinite(total)) {
                  selection_ok = false;
                  fail(step.control, 4, 7, call, global, step.health);
                }
                if (lane == 0 && total > 0.f)
                  for (int k = 0; k < Top; ++k) {
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
                    if constexpr (LocalSelection && !VerifyBatch) {
                      local_routing[r * Top + k] *= 3.66f / total;
                      if (wg == 0) {
                        work.experts[r * Top + k] = local_selected[r * Top + k];
                        work.routing[r * Top + k] = local_routing[r * Top + k];
                      }
                    } else
#endif
                      work.routing[r * Top + k] *= 3.66f / total;
                  }
              }
            }
            if constexpr (LocalSelection && !VerifyBatch) {
              if (!sycl::all_of_group(item.get_group(), selection_ok)) return;
#ifdef MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED
            } else if constexpr (FixedVerifyPages) {
              static_assert(LocalSharedActivation && GroupAcquire);
              // Routing and shared activation have independent outputs.
              // Publish Top16 first so other WGs can start expert gate/up.
              // Phase 9 publishes both activation paths before down.
              grid_signal(item, step.control, 7, call, step.health);
              shared_gateup();
              if (!grid_wait<Groups * Local, true>(
                      item, step.control, 7, call, step.health))
                return;
#endif
            } else {
              if (!grid_barrier_handoff<GroupAcquire>(
                      item, step.control, 7, call, step.health))
                return;
            }
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
            if constexpr (VerifyBatch) {
              if (lid < Rows * Top) {
                local_selected[lid] = work.experts[lid];
                local_routing[lid] = work.routing[lid];
              }
              reference_wg_bar(item.get_group());
            }
            if constexpr (Matrix) {
              const auto local_selected =
                  selected.template get_multi_ptr<sycl::access::decorated::no>()
                      .get();
              const auto local_experts =
                  union_experts
                      .template get_multi_ptr<sycl::access::decorated::no>()
                      .get();
              const auto local_choices =
                  union_choices
                      .template get_multi_ptr<sycl::access::decorated::no>()
                      .get();
              const auto local_count =
                  union_count
                      .template get_multi_ptr<sycl::access::decorated::no>()
                      .get();
  #ifdef MINICPM5_MONO_FP16
              if constexpr ((VerifyBatch && Rows == 8) || Rows == 4) {
                static_assert(TemporaryWords >= Rows * E + E / 16);
                // Ordinary selection reads router scores from this SLM.
                if constexpr (!VerifyBatch) reference_wg_bar(item.get_group());
                reference_expert_union_indexed<Rows, IgnorePadding>(
                    item,
                    local_selected,
                    local_experts,
                    local_choices,
                    local_count,
                    step.slots,
                    reinterpret_cast<int*>(reference_storage));
              } else
  #endif
              {
                reference_expert_union<Rows, true, IgnorePadding>(
                    item,
                    work,
                    local_selected,
                    local_experts,
                    local_choices,
                    local_count,
                    step.slots);
              }
              if constexpr (WideMoeGateUp<TP, Rows>) {
                constexpr int GateUpColumns = CombinedOrdinaryM8 ? 32 : 64;
                static_assert(
                    TemporaryWords >= Subgroups * Rows * GateUpColumns);
                reference_moe_gateup_wide<TP, Rows, GateUpColumns, TailRows4>(
                    item,
                    weights,
                    work,
                    TailOnly ? x : work.normed,
                    reinterpret_cast<uint32_t*>(reduction_storage),
                    reinterpret_cast<float*>(reference_storage),
                    local_experts,
                    local_choices,
                    local_count[0]);
              } else {
                reference_moe_gateup<TP, Rows, AlternateGuPartial>(
                    item,
                    weights,
                    work,
                    Rows <= 2 ? reinterpret_cast<uint32_t*>(reduction_storage)
                              : reference_storage,
                    Rows <= 2 ? reinterpret_cast<float*>(reference_storage)
                              : reduction_storage,
                    local_experts,
                    local_choices,
                    local_count[0]);
              }
              grid_signal(item, step.control, 9, call, step.health);
              if (!reference_moe_down<TP, Rows, IgnorePadding>(
                      item,
                      weights,
                      work,
                      local_routing,
                      local_experts,
                      local_choices,
                      local_count[0],
                      reinterpret_cast<scalar_t*>(reduction_storage),
                      routed,
                      reinterpret_cast<float*>(reference_storage),
                      step,
                      call))
                return;
            } else {
              reference_moe_gateup_m1<TP>(
                  item,
                  weights,
                  work,
                  x,
                  reinterpret_cast<float*>(reference_storage));
              if (!grid_barrier_handoff<GroupAcquire>(
                      item, step.control, 9, call, step.health))
                return;
              reference_moe_down_m1<TP>(
                  item,
                  weights,
                  work,
                  reinterpret_cast<float*>(reference_storage));
            }
#else
            if constexpr (Matrix) {
              const int expert_jobs = Rows * Top;
              // Deal the union of all rows' experts once. Results retain
              // per-expert BF16 rounding before routing weights are applied.
              for (int job = sid; job < expert_jobs * (I / 16) * KParts;
                   job += Groups * Subgroups) {
                const int part = job % KParts;
                const int tile = job / KParts % (I / 16);
                const int position = job / (KParts * (I / 16));
                if (!first_expert(work.experts, position)) continue;
                const int expert = work.experts[position];
                int offsets[Rows];
                for (int r = 0; r < Rows; ++r)
                  offsets[r] = r * H;
                const auto* weight = weights.w13 + size_t(expert) * H * 2 * I;
                const auto gate = projection_kn<Rows, H, 2 * I>(
                    sg,
                    weight,
                    tile * 16,
                    work.normed,
                    offsets,
                    part * (H / KParts),
                    (part + 1) * (H / KParts));
                const auto up = projection_kn<Rows, H, 2 * I>(
                    sg,
                    weight,
                    I + tile * 16,
                    work.normed,
                    offsets,
                    part * (H / KParts),
                    (part + 1) * (H / KParts));
                for (int r = 0; r < Rows; ++r) {
                  const int chosen = expert_choice(work.experts, r, expert);
                  if (chosen < 0) continue;
                  const size_t base = ((size_t(r) * Top + chosen) * (2 * I) +
                                       tile * 16 + lane) *
                                          KParts +
                                      part;
                  work.gate_up_partial[base] = gate[r];
                  work.gate_up_partial[base + size_t(I) * KParts] = up[r];
                }
              }
            } else {
              // K partitions load adjacent expert output columns on each lane.
              for (int job = sid; job < Top * (I / 64) * KParts;
                   job += Groups * Subgroups) {
                const int part = job % KParts, tile = job / KParts % (I / 64);
                const int chosen = job / (KParts * (I / 64));
                const int col = tile * 64 + lane * 4;
                for (int r = 0; r < Rows; ++r) {
                  const int expert = work.experts[r * Top + chosen];
                  float gate[4] = {}, up[4] = {};
                  for (int n = part * (H / KParts);
                       n < (part + 1) * (H / KParts);
                       ++n) {
                    const size_t base =
                        (size_t(expert) * H + n) * (2 * I) + col;
                    const auto gw = sycl::bit_cast<sycl::vec<uint8_t, 4>>(
                        *reinterpret_cast<const uint32_t*>(weights.w13 + base));
                    const auto uw = sycl::bit_cast<sycl::vec<uint8_t, 4>>(
                        *reinterpret_cast<const uint32_t*>(
                            weights.w13 + base + I));
                    const float value = float(work.normed[r * H + n]);
  #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                      gate[j] += fp8(gw[j]) * value;
                      up[j] += fp8(uw[j]) * value;
                    }
                  }
  #pragma unroll
                  for (int j = 0; j < 4; ++j) {
                    const size_t base =
                        ((size_t(r) * Top + chosen) * (2 * I) + col + j) *
                            KParts +
                        part;
                    work.gate_up_partial[base] = gate[j];
                    work.gate_up_partial[base + size_t(I) * KParts] = up[j];
                  }
                }
              }
            }
            if (!grid_barrier_handoff<GroupAcquire>(
                    item, step.control, 8, call, step.health))
              return;
            for (int v = global; v < Rows * Top * I; v += Groups * Local) {
              const int r = v / (Top * I), k = v / I % Top, n = v % I;
              const size_t base =
                  ((size_t(r) * Top + k) * (2 * I) + n) * KParts;
              float g = 0.f, u = 0.f;
              for (int part = 0; part < KParts; ++part) {
                g += work.gate_up_partial[base + part];
                u += work.gate_up_partial[base + size_t(I) * KParts + part];
              }
              const float sc = weights.w13_scale[work.experts[r * Top + k]];
              g = float(scalar_t(g * sc));
              u = float(scalar_t(u * sc));
              work.intermediate[r * IS + k * I + n] =
                  scalar_t(g / (1.f + sycl::exp(-g)) * u);
            }
            if (!grid_barrier_handoff<GroupAcquire>(
                    item, step.control, 9, call, step.health))
              return;
            if constexpr (Matrix) {
              const int expert_jobs = Rows * Top;
              for (int job = sid; job < expert_jobs * (H / 16);
                   job += Groups * Subgroups) {
                const int position = job / (H / 16);
                const int column = job % (H / 16) * 16;
                if (!first_expert(work.experts, position)) continue;
                const int expert = work.experts[position];
                int offsets[Rows], choices[Rows];
                for (int r = 0; r < Rows; ++r) {
                  choices[r] = expert_choice(work.experts, r, expert);
                  offsets[r] = choices[r] < 0 ? -1 : r * IS + choices[r] * I;
                }
                const auto result = projection_kn<Rows, I, H>(
                    sg,
                    weights.w2 + size_t(expert) * I * H,
                    column,
                    work.intermediate,
                    offsets,
                    0,
                    I);
                for (int r = 0; r < Rows; ++r)
                  if (choices[r] >= 0)
                    work.expert_out
                        [(r * (Top + 1) + choices[r]) * H + column + lane] =
                        scalar_t(result[r] * weights.w2_scale[expert]);
              }
            } else {
              for (int job = sid; job < Top * (H / 64);
                   job += Groups * Subgroups) {
                const int k = job / (H / 64),
                          col = job % (H / 64) * 64 + lane * 4;
                for (int r = 0; r < Rows; ++r) {
                  const int expert = work.experts[r * Top + k];
                  float result[4] = {};
                  for (int n = 0; n < I; ++n) {
                    const size_t off = (size_t(expert) * I + n) * H + col;
                    const auto w = sycl::bit_cast<sycl::vec<uint8_t, 4>>(
                        *reinterpret_cast<const uint32_t*>(weights.w2 + off));
                    const float value =
                        float(work.intermediate[r * IS + k * I + n]);
  #pragma unroll
                    for (int j = 0; j < 4; ++j)
                      result[j] += fp8(w[j]) * value;
                  }
  #pragma unroll
                  for (int j = 0; j < 4; ++j)
                    work.expert_out[(r * (Top + 1) + k) * H + col + j] =
                        scalar_t(result[j] * weights.w2_scale[expert]);
                }
              }
            }
#endif
          }
          const uint8_t* down = Dense ? weights.w2 : weights.shared2;
          const float down_scale =
              Dense ? weights.w2_scale[0] : weights.shared2_scale[0];
#ifdef MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED
          if constexpr (!Matrix) {
            reference_gemv_nk<SharedI, H>(
                item,
                down,
                work.intermediate + (Dense ? 0 : Top * I),
                work.expert_out + Top * H,
                down_scale,
                reduction_storage);
          } else {
            reference_projection_nk<Rows, SharedI, H, QDim>(
                item,
                down,
                work.intermediate + (Dense ? 0 : Top * I),
                IS,
                work.expert_out + Top * H,
                (Top + 1) * H,
                down_scale,
                reference_storage,
                reduction_storage);
          }
#else
          for (int job = sid; job < H / (Matrix ? 16 : 1);
               job += Groups * Subgroups) {
            const int n = Matrix ? job * 16 + lane : job;
            float out[Rows];
            if constexpr (Matrix) {
              int offsets[Rows];
              for (int r = 0; r < Rows; ++r)
                offsets[r] = r * IS + (Dense ? 0 : Top * I);
              const auto result = projection_nk<Rows, SharedI>(
                  sg,
                  down + size_t(job * 16) * SharedI,
                  work.intermediate,
                  offsets);
              for (int r = 0; r < Rows; ++r)
                out[r] = result[r];
            } else {
              projection<Rows, SharedI>(
                  sg,
                  down + size_t(n) * SharedI,
                  work.intermediate + (Dense ? 0 : Top * I),
                  IS,
                  out);
            }
            if (Matrix || lane == 0)
              for (int r = 0; r < Rows; ++r)
                work.expert_out[(r * (Top + 1) + Top) * H + n] =
                    scalar_t(out[r] * down_scale);
          }
#endif
          if constexpr (LocalDown) {
            // All expert and shared down results for these 64 columns were
            // produced by this WG. Only its own work-items need visibility.
            sycl::group_barrier(item.get_group());
          } else {
            if (!grid_barrier_handoff<GroupAcquire>(
                    item, step.control, 10, call, step.health))
              return;
          }
          const int r = lid / (H / Groups / 2);
          const int col = wg * (H / Groups) + 2 * (lid % (H / Groups / 2));
          float values[2]{};
          if (lid < Rows * (H / Groups / 2)) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              float local =
                  float(work.expert_out[(r * (Top + 1) + Top) * H + col + j]);
              if constexpr (!Dense) {
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
                if constexpr (Matrix) {
                  local = float(scalar_t(routed[j] + local));
                } else
#endif
                {
                  float sum = 0.f;
                  for (int k = 0; k < Top; ++k)
                    sum +=
                        work.routing[r * Top + k] *
                        float(
                            work.expert_out[(r * (Top + 1) + k) * H + col + j]);
                  local = float(scalar_t(float(scalar_t(sum)) + local));
                }
              }
              values[j] = local;
            }
          }
          if constexpr (VerifyBatch && TP == 2) {
            if (!reduce_fp16_packet<Rows>(
                    item, values[0], values[1], 1, call, step))
              return;
          } else if (lid < Rows * (H / Groups / 2)) {
            reduce_pair<TP>(values[0], values[1], 1, r, col, call, step);
          }
          if (lid < Rows * (H / Groups / 2)) {
            step.output[r * H + col] = scalar_t(values[0]);
            step.output[r * H + col + 1] = scalar_t(values[1]);
          }
          if constexpr (KernelExit) {
            // All 32 WGs published phase9 before any WG entered down/AR2.
            // Earlier global waiters are finished. A peer WG may STILL be
            // polling phase9: leave that atomic word untouched. Advance all
            // other counters, including unused phase11, for mixed graph
            // buckets. Queue/graph dependencies publish these stores at exit.
            if (global < Phases && global != 9)
              step.control[global * 16] = Groups * call;
            if (global == 0) Flag(step.control[Epoch]).store(call);
          } else {
            // Output is already published by every work-item's release. Only
            // WG leaders read global health after this final handoff; the
            // remaining epilogue writes counters from register-held values.
            if constexpr (Rows == 1) {
              grid_signal(item, step.control, 11, call, step.health);
              if (!grid_wait<0>(item, step.control, 11, call, step.health))
                return;
            } else {
              if (!grid_barrier_handoff<GroupAcquire>(
                      item, step.control, 11, call, step.health))
                return;
            }
            // All groups reached completion. A later graph bucket may use a
            // phase omitted by this specialization, but its target still uses
            // the shared call epoch. Advance those counters before this launch
            // completes; in-order graph/queue dependencies protect the handoff.
            // Earlier phase users have all reached phase11, so their atomic_ref
            // lifetimes are over. Plain stores suffice before kernel
            // completion. Leave phase11 alone: other groups may still be
            // polling that word.
            if (global < Phases && global != 11)
              step.control[global * 16] = Groups * call;
            if (global == 0) Flag(step.control[Epoch]).store(call);
          }
        });
  });
}

// Keep the legacy device argument layout. Padding is a host-selected
// instantiation of this same layer body, never a field of StepPointers.
template <int TP, int Rows>
void dispatch_layer(
    sycl::queue& queue,
    Weights weights,
    Workspace work,
    Step step,
    bool dense,
    bool ignore_padding) {
  if (dense) {
    launch<TP, Rows, true>(queue, weights, work, step);
  } else if constexpr (Rows == 4 || Rows == 8) {
    if (ignore_padding)
      launch<TP, Rows, false, true>(queue, weights, work, step);
    else
      launch<TP, Rows, false>(queue, weights, work, step);
  } else {
    launch<TP, Rows, false>(queue, weights, work, step);
  }
}
}  // namespace vllm::minicpm5::MINICPM5_MONO_NAMESPACE
