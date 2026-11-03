// SPDX-License-Identifier: Apache-2.0
//
// Top-k candidates of the vocabulary-parallel logits for small decode batches
// at TP2, instead of all-gathering the logits: each rank takes the k largest
// logits of its vocabulary shard per row, the two ranks swap their candidates
// through the P2P mailbox, and both write the same [rank 0's k | rank 1's k]
// list (each part in ascending token id order, so a position order tie-break
// equals the token id one). The k largest logits of the whole vocabulary are
// among these 2k: top-k (k_row <= k) / top-p / greedy sampling of the list
// is exact.
//
// NWR work-groups per row, each on a slice: tau = the k-th largest of the
// work-items' maxima (at least k logits are >= tau), the logits >= tau are
// collected in SLM (typically a few k) and ranked; the last work-group of the
// row (arrival counter) ranks the keys >= the largest per-slice k-th key. Keys
// order by value, then by the smaller token id, so they are distinct.
//
// k = 1 (greedy) has its own kernel: the slice maximum is a work-group
// reduction.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/clock.hpp>

#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

#include "mono_decode_xe2.h"

namespace mono_decode {
namespace {

constexpr int WGS = 512;
constexpr int KMAX = 64;
constexpr int BMAX = 8;
constexpr int CAP = 4096;  // collected logits per row (SLM)

inline uint64_t make_key(float v, int t) {
  const uint32_t b = sycl::bit_cast<uint32_t>(v);
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
  const sycl::half* lg;  // [B][ld] logits of this rank's shard
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

template <bool K1>
void shard_topk(sycl::queue& q, const Args& a, int rows) {
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
          const sycl::half* lg = a.lg + (size_t)row * a.ld;
          // slice of 8-logit vectors (vs and the slice bounds are multiples of
          // 8)
          const int nv = a.vs / 8, per = (nv + nw - 1) / nw;
          const int v0 = w * per, v1 = sycl::min(nv, v0 + per);
          auto keys8 = [&](int v, uint64_t (&kk)[8]) {
            const sycl::vec<sycl::half, 8> x =
                *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                    lg + (size_t)v * 8);
#pragma unroll
            for (int e = 0; e < 8; ++e)
              kk[e] = make_key((float)x[e], v * 8 + e);
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
            } else {
              a.ov[o] = -INFINITY;
              a.oi[o] = 0;
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

std::map<int, uint32_t*> g_tag;
std::map<int, uint64_t*> g_part;

}  // namespace
}  // namespace mono_decode

namespace vllm::mono_decode::xe2 {

int64_t mono_lmhead_mbox_bytes() {
  return (int64_t)(2 * ::mono_decode::BMAX * ::mono_decode::KMAX * 2 *
                   sizeof(uint64_t));
}

int64_t mono_shard_topk_max_k() { return ::mono_decode::KMAX; }

namespace {

// checks of the logits shard and the device's call state (tags, merge
// scratch)
::mono_decode::Args shard_args(
    const torch::Tensor& logits,
    int64_t rank,
    int64_t mbox,
    int64_t peer,
    const char* op) {
  using namespace ::mono_decode;
  const int B = (int)logits.size(0);
  TORCH_CHECK(
      logits.scalar_type() == at::kHalf && logits.dim() == 2 &&
          logits.stride(1) == 1,
      op,
      ": logits must be [B, V] fp16 with unit column stride");
  TORCH_CHECK(
      logits.size(1) % 8 == 0 && logits.stride(0) % 8 == 0 &&
          reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
      op,
      ": V and the row stride must be multiples of 8 (16-B rows)");
  TORCH_CHECK(B >= 1 && B <= BMAX, op, ": 1..8 rows");
  TORCH_CHECK(rank == 0 || rank == 1, op, ": TP2 ranks only");
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const int dev = logits.get_device();
  uint32_t*& tag = g_tag[dev];
  uint64_t*& part = g_part[dev];
  if (!tag) {
    tag = sycl::malloc_device<uint32_t>(3 + BMAX, q);
    q.memset(tag, 0, (3 + BMAX) * 4).wait();
    part = sycl::malloc_device<uint64_t>((size_t)BMAX * NW * KMAX, q);
  }
  Args a{};
  a.lg = reinterpret_cast<const sycl::half*>(logits.data_ptr());
  a.vs = (int)logits.size(1);
  a.ld = (int)logits.stride(0);
  a.rank = (int)rank;
  a.mine = reinterpret_cast<uint64_t*>(mbox);
  a.peer = reinterpret_cast<uint64_t*>(peer);
  a.tag = tag;
  a.part = part;
  a.health = vllm::mono_decode::xe2::health_words(dev);
  return a;
}

}  // namespace

void mono_shard_topk(
    torch::Tensor& vals,
    torch::Tensor& idx,
    const torch::Tensor& logits,
    int64_t k,
    int64_t rank,
    int64_t mbox,
    int64_t peer) {
  using namespace ::mono_decode;
  Args a = shard_args(logits, rank, mbox, peer, "mono_shard_topk");
  const int B = (int)logits.size(0);
  TORCH_CHECK(k >= 1 && k <= KMAX, "mono_shard_topk: k in 1..", KMAX);
  TORCH_CHECK(
      vals.scalar_type() == at::kFloat && vals.is_contiguous() &&
          vals.dim() == 2 && vals.size(0) == B && vals.size(1) >= 2 * k &&
          idx.scalar_type() == at::kInt && idx.is_contiguous() &&
          idx.sizes() == vals.sizes(),
      "mono_shard_topk: outputs must be contiguous [B, >= 2k] f32 / i32");
  a.k = (int)k;
  a.lo = (int)vals.size(1);
  a.ov = vals.data_ptr<float>();
  a.oi = idx.data_ptr<int32_t>();
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  if (k == 1)
    shard_topk<true>(q, a, B);
  else
    shard_topk<false>(q, a, B);
}

}  // namespace vllm::mono_decode::xe2
