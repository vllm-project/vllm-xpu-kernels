// SPDX-License-Identifier: Apache-2.0
// Adapted from the supplied vocabulary-shard candidate exchange reference.
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/experimental/clock.hpp>

#include <cmath>
#include <cstdint>

namespace vllm::minicpm5::sampling {
constexpr int WGS = 512;
constexpr int KMAX = 64;
constexpr int BMAX = 8;
constexpr int CAP = 4096;  // collected logits per row (SLM)

template <bool MaskNaN = false>
inline uint64_t make_key(float v, int t) {
  // Argmax treats signed zeros as equal and then chooses the smallest ID.
  uint32_t b = sycl::bit_cast<uint32_t>(v);
  const uint32_t magnitude = b & 0x7fffffffu;
  if (magnitude == 0) b = 0;
  if constexpr (MaskNaN) {
    // Match the stock DSpark greedy sampler's NaN-to-negative-infinity rule.
    if (magnitude > 0x7f800000u) b = 0xff800000u;
  }
  const uint32_t u = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return ((uint64_t)u << 32) | (uint32_t)(0xFFFFFFFFu - (uint32_t)t);
}
inline float key_val(uint64_t k) {
  const uint32_t u = (uint32_t)(k >> 32);
  return sycl::bit_cast<float>((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
}
inline int key_tok(uint64_t k) { return (int)(0xFFFFFFFFu - (uint32_t)k); }

// mailbox word: the 16-bit tag in each 32-bit half (a peer write may land as
// two halves), as in the decoder layers
inline uint64_t mb_word(uint32_t v, uint32_t tag) {
  const uint64_t t = tag & 0xffffu;
  return (t << 48) | ((uint64_t)(v >> 16) << 32) | (t << 16) | (v & 0xffffu);
}
inline bool mb_match(uint64_t w, uint32_t tag) {
  const uint32_t t = tag & 0xffffu;
  return (uint32_t)(w >> 48) == t && ((uint32_t)(w >> 16) & 0xffffu) == t;
}
inline uint32_t mb_val(uint64_t w) {
  return ((uint32_t)(w >> 32) << 16) | ((uint32_t)w & 0xffffu);
}
using mb_ref = sycl::atomic_ref<
    uint64_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>;
using u32_ref = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>;
using k64_ref = sycl::atomic_ref<
    uint64_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>;
using h_ref = sycl::atomic_ref<
    int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>;
// the device's health words and the wait bound, as in the layers
// (mono_layers.hpp)
enum { H_FAILED = 0, H_TIMEOUTS };
constexpr uint64_t WAIT_CYCLES = 1ull << 35;
using lcnt_ref = sycl::atomic_ref<
    int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::work_group,
    sycl::access::address_space::local_space>;

constexpr int NW = 32;  // per-row scratch slots
constexpr int NWR =
    4;  // work-groups per row (more: thinner slices, weaker thresholds, slower)

struct Args {
  const uint16_t* lg;    // [B][ld] logits of this rank's shard (fp16 or bf16)
  const uint16_t* bias;  // optional contiguous FP16 bias for DSpark greedy
  int vs, ld, k, rank,
      lo;          // lo: output row stride (>= 2k; [2k, lo) = -inf / 0)
  uint64_t* mine;  // own mailbox: [2 slots][BMAX][KMAX][2] words written by the
                   // peer (slot = tag & 1: a rank can start the next call while
                   // the peer still reads this one)
  uint64_t* peer;  // the peer's mailbox
  uint32_t* tag;   // [0] call counter (advanced by the last work-group to read
                   // it), [1 + row] row arrivals, [1 + BMAX] protocol failures,
                   // [2 + BMAX] call arrivals
  int* health;     // the device's health words (mono_health)
  uint64_t*
      part;   // [BMAX][NW][KMAX] keys: the k largest of each work-group's slice
  float* ov;  // [B][lo]
  int32_t* oi;  // [B][lo]
  int64_t* sampled;
  int32_t *num_sampled, *num_rejected;  // optional direct K1 outputs
};

// the k largest keys of lst[0 .. n) (distinct keys) -> sel[0 .. k) in no
// particular order; every work-item ranks its share of the list against the
// whole list
inline void top_of_list(
    sycl::nd_item<1> it, const uint64_t* lst, int n, int k, uint64_t* sel) {
  const int lid = it.get_local_linear_id();
  for (int i = lid; i < n; i += WGS) {
    const uint64_t ki = lst[i];
    int r = 0;
    for (int j = 0; j < n; ++j)
      r += lst[j] > ki;
    if (r < k) sel[r] = ki;
  }
}

template <bool K1, typename DT, bool DirectGreedy = false, bool AddBias = false>
void shard_topk(sycl::queue& q, const Args& a, int rows) {
  static_assert(!DirectGreedy || K1);
  static_assert(!AddBias || DirectGreedy);
  const int nw = NWR;  // work-groups per row
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint64_t, 1> km(sycl::range<1>(WGS), h);
    sycl::local_accessor<uint64_t, 1> lst(sycl::range<1>(CAP), h);
    sycl::local_accessor<uint64_t, 1> sel(sycl::range<1>(KMAX), h);
    sycl::local_accessor<int, 1> misc(sycl::range<1>(4), h);
    h.parallel_for(
        sycl::nd_range<1>(rows * nw * WGS, WGS),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          const int row = it.get_group(0) / nw, w = it.get_group(0) % nw;
          const int lid = it.get_local_linear_id();
          auto grp = it.get_group();
          const uint32_t tag = u32_ref(a.tag[0]).load() + 1u;
          // the call counter advances once every work-group has read it (the
          // next call starts after this kernel on the in-order queue)
          sycl::group_barrier(grp);
          if (lid == 0) {
            sycl::atomic_ref<
                uint32_t,
                sycl::memory_order::acq_rel,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                arr(a.tag[2 + BMAX]);
            if (arr.fetch_add(1u) == (uint32_t)(rows * nw - 1)) {
              arr.store(0u);
              u32_ref(a.tag[0]).store(tag);
            }
          }
          const uint16_t* lg = a.lg + (size_t)row * a.ld;
          // slice of 8-logit vectors (vs and the slice bounds are multiples of
          // 8)
          const int nv = a.vs / 8, per = (nv + nw - 1) / nw;
          const int v0 = w * per, v1 = sycl::min(nv, v0 + per);
          auto keys8 = [&](int v, uint64_t (&kk)[8]) {
            const sycl::vec<uint16_t, 8> x =
                *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(
                    lg + (size_t)v * 8);
            if constexpr (AddBias) {
              const auto b = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(
                  a.bias + (size_t)row * a.vs + v * 8);
#pragma unroll
              for (int e = 0; e < 8; ++e) {
                const DT sum =
                    DT(float(sycl::bit_cast<DT>(x[e])) +
                       float(sycl::bit_cast<DT>(b[e])));
                kk[e] = make_key<true>(float(sum), v * 8 + e);
              }
            } else {
#pragma unroll
              for (int e = 0; e < 8; ++e)
                kk[e] = make_key((float)sycl::bit_cast<DT>(x[e]), v * 8 + e);
            }
          };
          // 1. the k largest keys of the slice: tau = the k-th largest
          // work-item maximum
          uint64_t m = 0;
          for (int v = v0 + lid; v < v1; v += WGS) {
            uint64_t kk[8];
            keys8(v, kk);
#pragma unroll
            for (int e = 0; e < 8; ++e)
              m = sycl::max(m, kk[e]);
          }
          if constexpr (K1) {
            m = sycl::reduce_over_group(grp, m, sycl::maximum<uint64_t>());
            if (lid == 0) sel[0] = m;
            sycl::group_barrier(grp);
          } else {
            km[lid] = m;
            if (lid == 0) misc[0] = 0;
            sycl::group_barrier(grp);
            {
              int r = 0;
              for (int j = 0; j < WGS; ++j)
                r += km[j] > m;
              if (r == a.k - 1) sel[0] = m;
            }
            sycl::group_barrier(grp);
            const uint64_t tau = sel[0];
            sycl::group_barrier(grp);
            for (int v = v0 + lid; v < v1; v += WGS) {
              uint64_t kk[8];
              keys8(v, kk);
#pragma unroll
              for (int e = 0; e < 8; ++e)
                if (kk[e] >= tau) {
                  const int p = lcnt_ref(misc[0]).fetch_add(1);
                  if (p < CAP) lst[p] = kk[e];
                }
            }
            sycl::group_barrier(grp);
            top_of_list(
                it,
                &lst[0],
                sycl::min(misc[0], CAP),
                a.k,
                &sel[0]);  // sel: descending
            sycl::group_barrier(grp);
          }
          uint64_t* pw = a.part + ((size_t)row * NW + w) * KMAX;
          // The merge handshake uses ordered atomics only (a standalone
          // device-scope fence did not order these stores on this target, so
          // the last work-group could read a slot left by the previous call):
          // the keys are stored with release, the row counter's RMW is
          // acq_rel, and the reads below stay relaxed (ordered by the acquire
          // on the last work-group's RMW; L1 starts clean at every launch).
          if (lid < a.k)
            k64_ref(pw[lid]).store(sel[lid], sycl::memory_order::release);
          if (nw >
              1) {  // 2. the last work-group of the row merges the nw x k keys
            sycl::group_barrier(grp);
            if (lid == 0)
              misc[2] = u32_ref(a.tag[1 + row])
                            .fetch_add(1u, sycl::memory_order::acq_rel) ==
                        (uint32_t)(nw - 1);
            sycl::group_barrier(grp);
            if (!misc[2]) return;
            const uint64_t* pr = a.part + (size_t)row * NW * KMAX;
            // every global top-k key is >= tau2 = the largest k-th key of a
            // work-group (each work-group has k keys >= its own k-th): rank
            // only those
            uint64_t t2 = 0;
            for (int j = 0; j < nw; ++j)
              t2 = sycl::max(
                  t2,
                  k64_ref(const_cast<uint64_t*>(pr)[j * KMAX + a.k - 1])
                      .load());
            if (lid == 0) {
              misc[0] = 0;
              u32_ref(a.tag[1 + row]).store(0u);
            }
            sycl::group_barrier(grp);
            for (int i = lid; i < nw * a.k; i += WGS) {
              const uint64_t ki =
                  k64_ref(const_cast<uint64_t*>(pr)[(i / a.k) * KMAX + i % a.k])
                      .load();
              if (ki >= t2) lst[lcnt_ref(misc[0]).fetch_add(1)] = ki;
            }
            sycl::group_barrier(grp);
            top_of_list(it, &lst[0], misc[0], a.k, &sel[0]);
            sycl::group_barrier(grp);
          }
          // 3. this rank's k by token id -> output part + the peer's mailbox;
          // the peer's part from the own mailbox
          if (lid < a.k) {
            const uint64_t ki = sel[lid];
            const int ti = key_tok(ki);
            int pos = 0;
            for (int j = 0; j < a.k; ++j)
              pos += key_tok(sel[j]) < ti;
            const float v = key_val(ki);
            const int gid = a.rank * a.vs + ti;
            const size_t o = (size_t)row * a.lo + a.rank * a.k + pos;
            a.ov[o] = v;
            a.oi[o] = gid;
            const size_t mw =
                (((size_t)(tag & 1u) * BMAX + row) * KMAX + pos) * 2;
            mb_ref(a.peer[mw]).store(mb_word(sycl::bit_cast<uint32_t>(v), tag));
            mb_ref(a.peer[mw + 1]).store(mb_word((uint32_t)gid, tag));
          }
          for (int j = 2 * a.k + lid; j < a.lo; j += WGS) {  // padding columns
            a.ov[(size_t)row * a.lo + j] = -INFINITY;
            a.oi[(size_t)row * a.lo + j] = 0;
          }
          if (lid < a.k) {  // bounded like the layers' waits: a stalled peer
                            // must not hang the server (peer part -inf,
                            // counted in tag[1 + BMAX], health FAILED)
            namespace syclex = sycl::ext::oneapi::experimental;
            const size_t mw =
                (((size_t)(tag & 1u) * BMAX + row) * KMAX + lid) * 2;
            uint64_t x = 0, y = 0;
            bool okx = false, oky = false;
            uint64_t t0 = 0;
            for (uint32_t n = 0; !(okx && oky); ++n) {
              if (!okx) okx = mb_match(x = mb_ref(a.mine[mw]).load(), tag);
              if (!oky) oky = mb_match(y = mb_ref(a.mine[mw + 1]).load(), tag);
              if ((n & 255u) != 255u) continue;
              const uint64_t now =
                  syclex::clock<syclex::clock_scope::sub_group>();
              if (n == 255u) t0 = now;
              if (h_ref(a.health[H_FAILED]).load() != 0 ||
                  now - t0 > WAIT_CYCLES)
                break;
            }
            const size_t o = (size_t)row * a.lo + (1 - a.rank) * a.k + lid;
            if (okx && oky) {
              a.ov[o] = sycl::bit_cast<float>(mb_val(x));
              a.oi[o] = (int32_t)mb_val(y);
              if constexpr (DirectGreedy) {
                const uint64_t own = make_key<AddBias>(
                    key_val(sel[0]), a.rank * a.vs + key_tok(sel[0]));
                const uint64_t peer = make_key<AddBias>(
                    sycl::bit_cast<float>(mb_val(x)), (int32_t)mb_val(y));
                a.sampled[row] = key_tok(sycl::max(own, peer));
                a.num_sampled[row] = 1;
                a.num_rejected[row] = 0;
              }
            } else {
              a.ov[o] = -INFINITY;
              a.oi[o] = 0;
              if constexpr (DirectGreedy) {
                a.sampled[row] = 0;
                a.num_sampled[row] = 0;
                a.num_rejected[row] = 0;
              }
              if (lid == 0) {
                u32_ref(a.tag[1 + BMAX]).fetch_add(1u);
                h_ref(a.health[H_FAILED]).store(1);
                h_ref(a.health[H_TIMEOUTS]).fetch_add(1);
              }
            }
          }
        });
  });
}

}  // namespace vllm::minicpm5::sampling
