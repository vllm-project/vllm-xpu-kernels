// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Sub-group based row-wise top-k (values + int64 indices) over the last
// dimension of a tensor whose size is a multiple of the sub-group width
// (32) and >= 32.
// Ported from xpu-perf's projects/micro_perf/vendor_ops/INTEL/ops/sycl_ext/
// topk_kernel.cpp.
#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>
#include <tuple>

#include <ATen/DeviceGuard.h>

#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {

constexpr int kTopkSgSize = 32;
constexpr int kTopkSgMergeLevels = 5;
// 16 sub-groups is a 512-work-item group; 1024 does not fit a Xe core in the
// 512-GRF mode this is built with and faults the device.
constexpr int kTopkMaxSgs = 16;
// Past this footprint a single sub-group per slice stops covering the DRAM
// latency and splitting the slice over several sub-groups starts to pay.
constexpr int64_t kTopkSplitBytes = 96ll << 20;
// Bytes each lane pulls per loop iteration.
constexpr int kTopkVecBytes = 64;
// Smallest K that uses the sub-group-wide reject threshold.
constexpr int kTopkSharedThrMin = 16;
// Chunks between refreshes of that threshold. It only ever gets tighter, so
// a stale one still rejects safely; this just trades tightness for the
// merge cost.
constexpr int kTopkRefreshChunks = 8;

// 16-bit dtypes compute in float to avoid register spill.
template <typename T>
struct TopkComputeTypeHelper {
  using type = T;
};
template <>
struct TopkComputeTypeHelper<sycl::half> {
  using type = float;
};
template <>
struct TopkComputeTypeHelper<sycl::ext::oneapi::bfloat16> {
  using type = float;
};
template <typename T>
using topk_compute_t = typename TopkComputeTypeHelper<T>::type;

// ==========================================================================
// Per-lane top-K primitives (branch-free insert, bitonic merge, sub-group
// butterfly reduce).
// ==========================================================================
template <typename CompT, typename IndexT, int K, bool Largest>
struct topk_ops {
  static inline bool beats(CompT a, CompT b) {
    if constexpr (Largest)
      return a > b;
    else
      return a < b;
  }

  // Branch-free shift-insert. Slot i reads only the *old* v[i-1]/v[i], so
  // all K slots update independently; the textbook version's loop-carried
  // `inserted` flag serialises K dependent ops per element and costs ~3x
  // here.
  template <bool Eq>
  static inline void insert_body(CompT* v, IndexT* ix, CompT val, IndexT idx) {
#pragma unroll
    for (int i = K - 1; i > 0; --i) {
      bool shift, hit;
      if constexpr (Eq) {
        shift = !beats(v[i - 1], val);
        hit = !beats(v[i], val);
      } else {
        shift = beats(val, v[i - 1]);
        hit = beats(val, v[i]);
      }
      CompT nv = hit ? val : v[i];
      IndexT ni = hit ? idx : ix[i];
      v[i] = shift ? v[i - 1] : nv;
      ix[i] = shift ? ix[i - 1] : ni;
    }
    bool h0;
    if constexpr (Eq)
      h0 = !beats(v[0], val);
    else
      h0 = beats(val, v[0]);
    ix[0] = h0 ? idx : ix[0];
    v[0] = h0 ? val : v[0];
  }

  // `thr` must be a value that no member of the global top-K can fall
  // below, i.e. any lane's K-th best (they all hold K elements >= their
  // own K-th).
  static inline void
  insert(CompT* v, IndexT* ix, CompT val, IndexT idx, CompT thr) {
    if (!beats(val, thr)) return;
    insert_body<false>(v, ix, val, idx);
  }

  // Used only until every slot holds a real element. Compares with
  // equality so that a genuine -inf input still claims a slot and records
  // its index.
  static inline void insert_warm(CompT* v, IndexT* ix, CompT val, IndexT idx) {
    insert_body<true>(v, ix, val, idx);
  }

  // Values-only merge, so a threshold can be computed without disturbing
  // the indices the caller is tracking.
  static inline void merge_vals(CompT* A, const CompT* B) {
#pragma unroll
    for (int i = 0; i < K; ++i) {
      CompT bv = B[K - 1 - i];
      bool take;
      if constexpr (Largest)
        take = bv > A[i];
      else
        take = bv < A[i];
      if (take) A[i] = bv;
    }
#pragma unroll
    for (int stride = K / 2; stride >= 1; stride >>= 1) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
        int j = i ^ stride;
        bool swap;
        if constexpr (Largest)
          swap = (j > i) && (A[i] < A[j]);
        else
          swap = (j > i) && (A[i] > A[j]);
        if (swap) {
          CompT t = A[i];
          A[i] = A[j];
          A[j] = t;
        }
      }
    }
  }

  // The sub-group's true K-th best so far. A lane's own K-th, or the max
  // of those over the sub-group, only proves K elements beat it, so it
  // sits far down the tail: the guard above then rejects for all 32 lanes
  // at once maybe 0.3% of the time, the branch never fires, and every
  // element pays the full O(K) insert. This bound is the tightest valid
  // one, and lifts that to the majority of elements. Non-destructive: the
  // per-lane arrays keep indices.
  static inline CompT
  sg_kth(sycl::sub_group sg, int sg_lid, const CompT* vals) {
    CompT a[K];
#pragma unroll
    for (int i = 0; i < K; ++i)
      a[i] = vals[i];
#pragma unroll
    for (int d = 0; d < kTopkSgMergeLevels; ++d) {
      int partner = sg_lid ^ (1 << d);
      CompT b[K];
#pragma unroll
      for (int i = 0; i < K; ++i)
        b[i] = sycl::select_from_group(sg, a[i], partner);
      merge_vals(a, b);
    }
    return a[K - 1];
  }

  static inline void
  bitonic_merge(CompT* A, IndexT* A_idx, const CompT* B, const IndexT* B_idx) {
#pragma unroll
    for (int i = 0; i < K; ++i) {
      CompT bv = B[K - 1 - i];
      IndexT bi = B_idx[K - 1 - i];
      bool take;
      if constexpr (Largest)
        take = bv > A[i];
      else
        take = bv < A[i];
      if (take) {
        A[i] = bv;
        A_idx[i] = bi;
      }
    }
#pragma unroll
    for (int stride = K / 2; stride >= 1; stride >>= 1) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
        int j = i ^ stride;
        bool swap;
        if constexpr (Largest)
          swap = (j > i) && (A[i] < A[j]);
        else
          swap = (j > i) && (A[i] > A[j]);
        if (swap) {
          CompT tv = A[i];
          A[i] = A[j];
          A[j] = tv;
          IndexT ti = A_idx[i];
          A_idx[i] = A_idx[j];
          A_idx[j] = ti;
        }
      }
    }
  }

  // Butterfly-exchange the per-lane top-K across `levels` sub-group levels.
  static inline void sg_reduce(
      sycl::sub_group sg, int sg_lid, int levels, CompT* vals, IndexT* idx) {
    for (int d = 0; d < levels; ++d) {
      int partner = sg_lid ^ (1 << d);
      CompT pv[K];
      IndexT pi[K];
#pragma unroll
      for (int i = 0; i < K; ++i) {
        pv[i] = sycl::select_from_group(sg, vals[i], partner);
        pi[i] = sycl::select_from_group(sg, idx[i], partner);
      }
      bitonic_merge(vals, idx, pv, pi);
    }
  }

  static constexpr CompT init() {
    return Largest ? -std::numeric_limits<CompT>::infinity()
                   : std::numeric_limits<CompT>::infinity();
  }
};

// One sub-group per slice.
template <
    typename scalar_t,
    int K,
    int VEC_SIZE,
    bool Largest,
    typename IndexT = int>
class subgroup_topk_kernel {
 public:
  using comp_t = topk_compute_t<scalar_t>;
  using ops = topk_ops<comp_t, IndexT, K, Largest>;

  subgroup_topk_kernel(
      const scalar_t* input,
      scalar_t* out_vals,
      int64_t* out_idx,
      IndexT num_slices,
      int64_t slice_size,
      int k)
      : input_(input),
        out_vals_(out_vals),
        out_idx_(out_idx),
        num_slices_(num_slices),
        slice_size_(slice_size),
        k_(k) {}

  void operator()
      [[sycl::reqd_sub_group_size(kTopkSgSize)]] (sycl::nd_item<1> item) const {
    sycl::sub_group sg = item.get_sub_group();
    int sg_lid = sg.get_local_linear_id();
    int sgs_per_wg = item.get_local_range(0) / kTopkSgSize;
    IndexT slice =
        static_cast<IndexT>(item.get_group_linear_id()) * sgs_per_wg +
        sg.get_group_linear_id();
    if (slice >= num_slices_) return;

    const scalar_t* input_slice = input_ + (int64_t)slice * slice_size_;
    scalar_t* out_vals_slice = out_vals_ + (int64_t)slice * k_;
    int64_t* out_idx_slice = out_idx_ + (int64_t)slice * k_;

    comp_t top_vals[K];
    IndexT top_idx_local[K];
#pragma unroll
    for (int i = 0; i < K; ++i) {
      top_vals[i] = ops::init();
      top_idx_local[i] = -1;
    }

    // Aligned vector load: dispatch guarantees input_slice is
    // VEC_SIZE-aligned and slice_size_ % VEC_SIZE == 0, so every chunk is
    // a single wide load.
    struct alignas(VEC_SIZE * sizeof(scalar_t)) VecLoad {
      scalar_t data[VEC_SIZE];
    };
    constexpr int kWarmChunks = (K + VEC_SIZE - 1) / VEC_SIZE;
    constexpr bool kSharedThr = (K >= kTopkSharedThrMin);
    int stride = kTopkSgSize * VEC_SIZE;
    int64_t base = sg_lid * VEC_SIZE;
    comp_t thr = ops::init();
    int refresh = 0;

    for (int c = 0; c < kWarmChunks && base + VEC_SIZE <= slice_size_;
         ++c, base += stride) {
      VecLoad chunk = *reinterpret_cast<const VecLoad*>(input_slice + base);
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v)
        ops::insert_warm(
            top_vals,
            top_idx_local,
            static_cast<comp_t>(chunk.data[v]),
            (IndexT)(base + v));
    }
    for (; base + VEC_SIZE <= slice_size_; base += stride) {
      VecLoad chunk = *reinterpret_cast<const VecLoad*>(input_slice + base);
      if constexpr (kSharedThr) {
        if (refresh == 0) {
          thr = ops::sg_kth(sg, sg_lid, top_vals);
          refresh = kTopkRefreshChunks;
        }
        --refresh;
      }
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        comp_t val = static_cast<comp_t>(chunk.data[v]);
        if constexpr (kSharedThr)
          ops::insert(top_vals, top_idx_local, val, (IndexT)(base + v), thr);
        else
          ops::insert(
              top_vals,
              top_idx_local,
              val,
              (IndexT)(base + v),
              top_vals[K - 1]);
      }
    }
    for (IndexT idx = (IndexT)base; idx < slice_size_ && idx < base + VEC_SIZE;
         ++idx)
      ops::insert_warm(
          top_vals, top_idx_local, static_cast<comp_t>(input_slice[idx]), idx);

    ops::sg_reduce(sg, sg_lid, kTopkSgMergeLevels, top_vals, top_idx_local);

    if (sg_lid == 0) {
      for (int i = 0; i < k_; ++i) {
        out_vals_slice[i] = static_cast<scalar_t>(top_vals[i]);
        out_idx_slice[i] = static_cast<int64_t>(top_idx_local[i]);
      }
    }
  }

 private:
  const scalar_t* input_;
  scalar_t* out_vals_;
  int64_t* out_idx_;
  IndexT num_slices_;
  int64_t slice_size_;
  int k_;
};

// One work-group per slice. The single-sub-group form only exposes
// num_slices * 32 work-items, which is far too shallow to keep this memory
// system busy; spreading a slice over up to 32 sub-groups scales that up by
// the same factor and costs one extra SLM merge at the end.
template <
    typename scalar_t,
    int K,
    int VEC_SIZE,
    bool Largest,
    typename IndexT = int>
class wg_topk_kernel {
 public:
  using comp_t = topk_compute_t<scalar_t>;
  using ops = topk_ops<comp_t, IndexT, K, Largest>;

  wg_topk_kernel(
      const scalar_t* input,
      scalar_t* out_vals,
      int64_t* out_idx,
      IndexT num_slices,
      int64_t slice_size,
      int k,
      sycl::local_accessor<comp_t, 1> slm_vals,
      sycl::local_accessor<IndexT, 1> slm_idx)
      : input_(input),
        out_vals_(out_vals),
        out_idx_(out_idx),
        num_slices_(num_slices),
        slice_size_(slice_size),
        k_(k),
        slm_vals_(slm_vals),
        slm_idx_(slm_idx) {}

  void operator()
      [[sycl::reqd_sub_group_size(kTopkSgSize)]] (sycl::nd_item<1> item) const {
    sycl::sub_group sg = item.get_sub_group();
    const int sg_lid = sg.get_local_linear_id();
    const int sg_id = sg.get_group_linear_id();
    const int nsg = item.get_local_range(0) / kTopkSgSize;
    const IndexT slice = static_cast<IndexT>(item.get_group_linear_id());

    const scalar_t* input_slice = input_ + (int64_t)slice * slice_size_;

    comp_t top_vals[K];
    IndexT top_idx_local[K];
#pragma unroll
    for (int i = 0; i < K; ++i) {
      top_vals[i] = ops::init();
      top_idx_local[i] = -1;
    }

    struct alignas(VEC_SIZE * sizeof(scalar_t)) VecLoad {
      scalar_t data[VEC_SIZE];
    };
    // slice_size_ % VEC_SIZE == 0 and stride is a multiple of VEC_SIZE, so
    // the work-group covers every element exactly once with no tail.
    constexpr int kWarmChunks = (K + VEC_SIZE - 1) / VEC_SIZE;
    constexpr bool kSharedThr = (K >= kTopkSharedThrMin);
    const int lane = sg_id * kTopkSgSize + sg_lid;
    const int64_t stride = (int64_t)nsg * kTopkSgSize * VEC_SIZE;
    int64_t base = (int64_t)lane * VEC_SIZE;
    comp_t thr = ops::init();
    int refresh = 0;

    for (int c = 0; c < kWarmChunks && base + VEC_SIZE <= slice_size_;
         ++c, base += stride) {
      VecLoad chunk = *reinterpret_cast<const VecLoad*>(input_slice + base);
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v)
        ops::insert_warm(
            top_vals,
            top_idx_local,
            static_cast<comp_t>(chunk.data[v]),
            (IndexT)(base + v));
    }
    for (; base + VEC_SIZE <= slice_size_; base += stride) {
      VecLoad chunk = *reinterpret_cast<const VecLoad*>(input_slice + base);
      if constexpr (kSharedThr) {
        if (refresh == 0) {
          thr = ops::sg_kth(sg, sg_lid, top_vals);
          refresh = kTopkRefreshChunks;
        }
        --refresh;
      }
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        comp_t val = static_cast<comp_t>(chunk.data[v]);
        if constexpr (kSharedThr)
          ops::insert(top_vals, top_idx_local, val, (IndexT)(base + v), thr);
        else
          ops::insert(
              top_vals,
              top_idx_local,
              val,
              (IndexT)(base + v),
              top_vals[K - 1]);
      }
    }

    ops::sg_reduce(sg, sg_lid, kTopkSgMergeLevels, top_vals, top_idx_local);

    if (sg_lid == 0) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
        slm_vals_[sg_id * K + i] = top_vals[i];
        slm_idx_[sg_id * K + i] = top_idx_local[i];
      }
    }
    sycl::group_barrier(item.get_group());

    if (sg_id != 0) return;
    if (sg_lid < nsg) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
        top_vals[i] = slm_vals_[sg_lid * K + i];
        top_idx_local[i] = slm_idx_[sg_lid * K + i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < K; ++i) {
        top_vals[i] = ops::init();
        top_idx_local[i] = -1;
      }
    }
    int levels = 0;
    while ((1 << levels) < nsg)
      ++levels;
    ops::sg_reduce(sg, sg_lid, levels, top_vals, top_idx_local);

    if (sg_lid == 0) {
      scalar_t* out_vals_slice = out_vals_ + (int64_t)slice * k_;
      int64_t* out_idx_slice = out_idx_ + (int64_t)slice * k_;
      for (int i = 0; i < k_; ++i) {
        out_vals_slice[i] = static_cast<scalar_t>(top_vals[i]);
        out_idx_slice[i] = static_cast<int64_t>(top_idx_local[i]);
      }
    }
  }

 private:
  const scalar_t* input_;
  scalar_t* out_vals_;
  int64_t* out_idx_;
  IndexT num_slices_;
  int64_t slice_size_;
  int k_;
  sycl::local_accessor<comp_t, 1> slm_vals_;
  sycl::local_accessor<IndexT, 1> slm_idx_;
};

// ==========================================================================
// Launch / dispatch
// ==========================================================================
template <typename scalar_t, int K, int VEC_SIZE, bool Largest>
void launch_subgroup_topk_kernel(
    sycl::queue& q,
    const scalar_t* input,
    scalar_t* out_vals,
    int64_t* out_idx,
    int num_slices,
    int64_t slice_size,
    int k) {
  constexpr int kWgSize = 256;
  constexpr int kSgsPerWg = kWgSize / kTopkSgSize;
  int64_t num_wgs = ((int64_t)num_slices + kSgsPerWg - 1) / kSgsPerWg;

  q.submit([&](sycl::handler& cgh) {
    subgroup_topk_kernel<scalar_t, K, VEC_SIZE, Largest> kfn(
        input, out_vals, out_idx, num_slices, slice_size, k);
    cgh.parallel_for(sycl::nd_range<1>(num_wgs * kWgSize, kWgSize), kfn);
  });
}

template <typename scalar_t, int K, int VEC_SIZE, bool Largest>
void launch_wg_topk_kernel(
    sycl::queue& q,
    const scalar_t* input,
    scalar_t* out_vals,
    int64_t* out_idx,
    int num_slices,
    int64_t slice_size,
    int k,
    int nsg) {
  using comp_t = topk_compute_t<scalar_t>;
  const int wg_size = nsg * kTopkSgSize;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<comp_t, 1> slm_vals(nsg * K, cgh);
    sycl::local_accessor<int, 1> slm_idx(nsg * K, cgh);
    wg_topk_kernel<scalar_t, K, VEC_SIZE, Largest> kfn(
        input, out_vals, out_idx, num_slices, slice_size, k, slm_vals, slm_idx);
    cgh.parallel_for(
        sycl::nd_range<1>((int64_t)num_slices * wg_size, wg_size), kfn);
  });
}

// Choose single-sub-group vs. multi-sub-group (work-group) launch based on
// slice footprint and how many slices there are to keep the device busy.
template <typename scalar_t, int K, int VEC_SIZE, bool Largest>
void launch_topk_size_dispatch(
    sycl::queue& q,
    const scalar_t* input,
    scalar_t* out_vals,
    int64_t* out_idx,
    int num_slices,
    int64_t slice_size,
    int k) {
  // Widening the work-group buys memory parallelism but shrinks the
  // elements each lane sees, and a lane that sees fewer elements holds a
  // looser K-th, which is exactly what the reject threshold is built from.
  // For K=16 the insert is expensive enough that threshold tightness wins
  // over parallelism, so it keeps a much higher per-lane floor than the
  // small-K path.
  constexpr int64_t kLaneElems = K >= 16 ? 1024 : 256;
  const int64_t total_bytes =
      (int64_t)num_slices * slice_size * sizeof(scalar_t);
  int nsg = 1;
  if (total_bytes > kTopkSplitBytes)
    while (nsg < kTopkMaxSgs &&
           (int64_t)nsg * 2 * kTopkSgSize * kLaneElems <= slice_size)
      nsg <<= 1;
  // Few slices: one sub-group each would leave the machine idle, so widen
  // anyway.
  while (nsg < kTopkMaxSgs && (int64_t)num_slices * nsg < 1024 &&
         (int64_t)nsg * 2 * kTopkSgSize * VEC_SIZE <= slice_size)
    nsg <<= 1;

  if (nsg == 1)
    launch_subgroup_topk_kernel<scalar_t, K, VEC_SIZE, Largest>(
        q, input, out_vals, out_idx, num_slices, slice_size, k);
  else
    launch_wg_topk_kernel<scalar_t, K, VEC_SIZE, Largest>(
        q, input, out_vals, out_idx, num_slices, slice_size, k, nsg);
}

// Choose the widest vector load that keeps the slice aligned and covered
// with no tail.
template <typename scalar_t, int K, bool Largest>
void launch_topk_vec_dispatch(
    sycl::queue& q,
    const scalar_t* input,
    scalar_t* out_vals,
    int64_t* out_idx,
    int num_slices,
    int64_t slice_size,
    int k) {
  constexpr int kMaxVec = kTopkVecBytes / sizeof(scalar_t);
  auto align = reinterpret_cast<uintptr_t>(input);
  auto can = [&](int v) {
    return slice_size % v == 0 && kTopkSgSize * v <= slice_size &&
           align % (sizeof(scalar_t) * v) == 0;
  };
  if (can(kMaxVec))
    launch_topk_size_dispatch<scalar_t, K, kMaxVec, Largest>(
        q, input, out_vals, out_idx, num_slices, slice_size, k);
  else if (can(kMaxVec / 4))
    launch_topk_size_dispatch<scalar_t, K, kMaxVec / 4, Largest>(
        q, input, out_vals, out_idx, num_slices, slice_size, k);
  else if (can(kMaxVec / 8))
    launch_topk_size_dispatch<scalar_t, K, kMaxVec / 8, Largest>(
        q, input, out_vals, out_idx, num_slices, slice_size, k);
  else
    launch_topk_size_dispatch<scalar_t, K, 1, Largest>(
        q, input, out_vals, out_idx, num_slices, slice_size, k);
}

// Round k up to the nearest supported per-lane capacity {1,2,4,8,16} and
// dispatch largest/smallest.
template <typename scalar_t>
void launch_topk_dispatch(
    sycl::queue& q,
    const scalar_t* input,
    scalar_t* out_vals,
    int64_t* out_idx,
    int num_slices,
    int64_t slice_size,
    int k,
    bool largest) {
  int k_sel = 1;
  while (k_sel < k && k_sel < 16)
    k_sel <<= 1;
  if (k_sel < k) k_sel = 16;

  switch (k_sel) {
    case 1:
      if (largest)
        launch_topk_vec_dispatch<scalar_t, 1, true>(
            q, input, out_vals, out_idx, num_slices, slice_size, k);
      else
        launch_topk_vec_dispatch<scalar_t, 1, false>(
            q, input, out_vals, out_idx, num_slices, slice_size, k);
      break;
    case 2:
      if (largest)
        launch_topk_vec_dispatch<scalar_t, 2, true>(
            q, input, out_vals, out_idx, num_slices, slice_size, k);
      else
        launch_topk_vec_dispatch<scalar_t, 2, false>(
            q, input, out_vals, out_idx, num_slices, slice_size, k);
      break;
    case 4:
      if (largest)
        launch_topk_vec_dispatch<scalar_t, 4, true>(
            q, input, out_vals, out_idx, num_slices, slice_size, k);
      else
        launch_topk_vec_dispatch<scalar_t, 4, false>(
            q, input, out_vals, out_idx, num_slices, slice_size, k);
      break;
    case 8:
      if (largest)
        launch_topk_vec_dispatch<scalar_t, 8, true>(
            q, input, out_vals, out_idx, num_slices, slice_size, k);
      else
        launch_topk_vec_dispatch<scalar_t, 8, false>(
            q, input, out_vals, out_idx, num_slices, slice_size, k);
      break;
    default:
      if (largest)
        launch_topk_vec_dispatch<scalar_t, 16, true>(
            q, input, out_vals, out_idx, num_slices, slice_size, k);
      else
        launch_topk_vec_dispatch<scalar_t, 16, false>(
            q, input, out_vals, out_idx, num_slices, slice_size, k);
      break;
  }
}

}  // namespace vllm

// Row-wise top-k: returns (values, indices) over the last dimension,
// matching torch.topk(input, k, dim=-1, largest, sorted=True). indices are
// int64. Last dim must be a multiple of 32 (the sub-group width) and >= 32,
// and k must be in [1, 16].
std::tuple<torch::Tensor, torch::Tensor>
topk_row_wise(const torch::Tensor& input, int64_t k, bool largest) {
  const at::DeviceGuard device_guard(input.device());
  CHECK_DEVICE(input);
  TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
  TORCH_CHECK(k > 0 && k <= 16, "k must be in [1, 16]");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16 ||
          input.scalar_type() == at::ScalarType::Float,
      "input must be half, bfloat16, or float32");

  const int64_t dim_size = input.size(-1);
  TORCH_CHECK(
      dim_size % vllm::kTopkSgSize == 0 && dim_size >= vllm::kTopkSgSize,
      "last dim must be a multiple of ",
      vllm::kTopkSgSize,
      " and >= ",
      vllm::kTopkSgSize);
  const int64_t batch = input.numel() / dim_size;

  auto values = torch::empty({batch, k}, input.options());
  auto indices = torch::empty({batch, k}, input.options().dtype(at::kLong));
  if (batch == 0) {
    auto shape = input.sizes().vec();
    shape.back() = k;
    return {values.view(shape), indices.view(shape)};
  }

  auto& q = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "topk", [&] {
    using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
    vllm::launch_topk_dispatch<sycl_t>(
        q,
        reinterpret_cast<const sycl_t*>(input.const_data_ptr<scalar_t>()),
        reinterpret_cast<sycl_t*>(values.data_ptr<scalar_t>()),
        indices.data_ptr<int64_t>(),
        static_cast<int>(batch),
        dim_size,
        static_cast<int>(k),
        largest);
  });

  // Reshape output to match input shape (except last dim -> k).
  auto shape = input.sizes().vec();
  shape.back() = k;
  return {values.view(shape), indices.view(shape)};
}
