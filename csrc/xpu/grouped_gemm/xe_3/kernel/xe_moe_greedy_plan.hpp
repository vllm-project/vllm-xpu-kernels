// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "cutlass/fast_math.h"

#include "../runner/greedy_dispatch.hpp"

namespace gpu::cutlass_kernel::grouped_gemm::greedy {

enum class Kind { Empty, Tiny, Mid, Normal };
enum class Bucket {
  Tiny8,
  Tiny32,
  Tiny64,
  Tiny128,
  Normal32,
  Normal192,
  Normal256,
  Mid192,
  Mid256,
  Mid320
};

struct MidWindow {
  int min = 0;
  int max = 0;
  int hole_min = 0;
  int hole_max = 0;
};

struct MidOverride {
  int n, k;
  MidWindow window;
};

template <std::size_t Size>
constexpr MidWindow mid_window_for(
    MidOverride const (&overrides)[Size], int n, int k, MidWindow fallback) {
  for (auto entry : overrides)
    if (entry.n == n && entry.k == k) return entry.window;
  return fallback;
}

// The helper store is only reachable when every expert holds the same count
// above 128, so the packed row total must exceed groups * 128. Testing that
// host-side bound keeps small and decode-shaped jobs on the smaller
// native-only kernel instead of the larger combined one.
constexpr bool may_have_large_uniform_experts(int64_t total_rows, int groups) {
  return groups > 0 && total_rows > int64_t(groups) * 128;
}

constexpr Kind classify(int rows, MidWindow window) {
  if (rows <= 0) return Kind::Empty;
  if (rows <= 128) return Kind::Tiny;
  return rows >= window.min && rows < window.max &&
                 !(rows >= window.hole_min && rows < window.hole_max)
             ? Kind::Mid
             : Kind::Normal;
}

struct RowTile {
  Bucket bucket;
  int offset;
  int height;
};

struct Tail {
  int first;
  int second;
};

template <int Large, int Medium, int Small, int Floor>
constexpr int leftover(int rows) {
  if (rows <= Floor) return Floor;
  if (rows <= Small) return Small;
  if (rows <= Medium) return Medium;
  return Large;
}

// SYCL-TLA LOOK-2: keep the earlier large tiles and minimize padding in
// the final pair. On a coverage tie prefer the more balanced pair.
template <int Large, int Medium, int Small, int Floor>
constexpr Tail tail_plan(int remainder) {
  const int cover = Large + remainder;
  const Tail pairs[] = {
      {Small, Small},
      {Medium, Small},
      {Medium, Medium},
      {Large, Small},
      {Large, Medium},
      {Large, Large}};
  Tail best{Large, Large};
  for (auto pair : pairs) {
    const int sum = pair.first + pair.second;
    const int best_sum = best.first + best.second;
    if (sum >= cover && sum < best_sum) {
      best = pair;
    }
  }
  const int rest = leftover<Large, Medium, Small, Floor>(remainder);
  return Large + rest < best.first + best.second ? Tail{Large, rest} : best;
}

constexpr int row_tiles(Kind kind, int rows) {
  if (kind == Kind::Empty) return 0;
  if (kind == Kind::Tiny) return 1;
  const int peel = kind == Kind::Mid ? 320 : 256;
  return rows / peel + (rows % peel != 0);
}

constexpr RowTile row_tile(Kind kind, int rows, int index) {
  if (kind == Kind::Tiny) {
    if (rows <= 8) return {Bucket::Tiny8, 0, 8};
    if (rows <= 32) return {Bucket::Tiny32, 0, 32};
    if (rows <= 64) return {Bucket::Tiny64, 0, 64};
    return {Bucket::Tiny128, 0, 128};
  }
  const bool mid = kind == Kind::Mid;
  const int large = mid ? 320 : 256;
  const int full = rows / large;
  const int rem = rows % large;
  int height = large;
  int offset = index * large;
  if (full == 0) {
    height = mid ? leftover<320, 256, 192, 192>(rem)
                 : leftover<256, 192, 192, 32>(rem);
  } else if (rem != 0 && index >= full - 1) {
    const Tail tail = mid ? tail_plan<320, 256, 192, 192>(rem)
                          : tail_plan<256, 192, 192, 32>(rem);
    offset = (full - 1) * large;
    height = tail.first;
    if (index == full) {
      offset += tail.first;
      height = tail.second;
    }
  }
  if (mid) {
    return {
        height == 320   ? Bucket::Mid320
        : height == 256 ? Bucket::Mid256
                        : Bucket::Mid192,
        offset,
        height};
  }
  return {
      height == 256   ? Bucket::Normal256
      : height == 192 ? Bucket::Normal192
                      : Bucket::Normal32,
      offset,
      height};
}

constexpr int64_t padded_rows(int rows) {
  return (int64_t(rows) + 3) & ~int64_t(3);
}

struct Work {
  int expert;
  int rows;
  int n_tile;
  int64_t row_offset;
  int64_t scale_offset;
  RowTile tile;
  constexpr bool valid() const { return expert >= 0; }
};

// Per-workgroup scheduling plan, staged in SLM. The prefix arrays hold
// groups + 1 entries so that entry e is the total contributed by experts
// below e.
//
// The entries are 32-bit because this is SLM on a persistent grid, where extra
// shared memory per workgroup costs occupancy for the whole kernel; halving the
// width measurably recovers throughput on shapes that issue many tiles. An
// expert's row count is the step in the row prefix, so the counts are not
// staged separately.
struct PlanStorage {
  int32_t* tile_prefix;
  int32_t* row_prefix;
  int32_t* scale_prefix;
};

constexpr std::size_t plan_bytes(int groups) {
  return 3 * (std::size_t(groups) + 1) * sizeof(int32_t);
}

inline PlanStorage plan_storage(char* base, int groups) {
  auto* p = reinterpret_cast<int32_t*>(base);
  const std::size_t stride = std::size_t(groups) + 1;
  return {p, p + stride, p + 2 * stride};
}

template <int MidN, int NormalN>
constexpr int64_t n_tiles_for(Kind kind, int n) {
  if (kind == Kind::Tiny) return (int64_t(n) + 511) / 512;
  return kind == Kind::Mid ? (int64_t(n) + MidN - 1) / MidN
                           : (int64_t(n) + NormalN - 1) / NormalN;
}

template <int MidN, int NormalN>
constexpr int64_t tiles_for(int rows, MidWindow window, int n) {
  const Kind kind = classify(rows, window);
  return int64_t(row_tiles(kind, rows)) * n_tiles_for<MidN, NormalN>(kind, n);
}

// Build the plan cooperatively across the whole workgroup.
//
// The scheduler used to find the expert owning a tile by walking the counts
// forward, a serial chain of dependent global loads run in every workgroup.
// At 128 experts that measured ~15us, independent of N, K and the row count.
//
// The scan is Hillis-Steele over SLM. Two cheaper-looking shapes measured
// worse: confining it to one subgroup serialises it and lost 8 to 12%, and
// scanning per subgroup then adding the preceding totals makes every work item
// walk up to 31 totals out of SLM and lost a further 5%. Barriers are not the
// cost here; keeping every work item busy is.
//
// Workgroups are 512 wide and the expert count stays well below that, so each
// work item owns at most one entry. The serial fallback guards the rest.
template <int MidN, int NormalN, typename Item>
void build_plan(
    PlanStorage plan,
    const int* counts,
    int groups,
    int n,
    MidWindow window,
    Item item) {
  const int lid = int(item.get_local_linear_id());
  const int lrange = int(item.get_local_range().size());
  const int entries = groups + 1;
  if (entries > lrange) {
    if (lid == 0) {
      int32_t tile = 0, row = 0, scale = 0;
      plan.tile_prefix[0] = 0;
      plan.row_prefix[0] = 0;
      plan.scale_prefix[0] = 0;
      for (int expert = 0; expert < groups; ++expert) {
        const int rows = counts[expert];
        tile += int32_t(tiles_for<MidN, NormalN>(rows, window, n));
        row += int32_t(rows);
        scale += int32_t(padded_rows(rows));
        plan.tile_prefix[expert + 1] = tile;
        plan.row_prefix[expert + 1] = row;
        plan.scale_prefix[expert + 1] = scale;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
    return;
  }
  // Entry 0 is the empty prefix, so work item i carries expert i - 1.
  const bool owns = lid < entries;
  const bool live = owns && lid > 0;
  const int rows = live ? counts[lid - 1] : 0;
  if (owns) {
    plan.tile_prefix[lid] =
        live ? int32_t(tiles_for<MidN, NormalN>(rows, window, n)) : 0;
    plan.row_prefix[lid] = live ? int32_t(rows) : 0;
    plan.scale_prefix[lid] = live ? int32_t(padded_rows(rows)) : 0;
  }
  item.barrier(sycl::access::fence_space::local_space);
  for (int offset = 1; offset < entries; offset <<= 1) {
    const bool add = owns && lid >= offset;
    int32_t tile = 0, row = 0, scale = 0;
    if (add) {
      tile = plan.tile_prefix[lid - offset];
      row = plan.row_prefix[lid - offset];
      scale = plan.scale_prefix[lid - offset];
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (add) {
      plan.tile_prefix[lid] += tile;
      plan.row_prefix[lid] += row;
      plan.scale_prefix[lid] += scale;
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
}

// The compacted plan is only valid when no expert can exceed a single row
// tile, which is exactly the 128-row ceiling of Kind::Tiny. Every non-empty
// expert then owns one row tile and the same number of column tiles, so a
// tile index maps to an active-expert index by a single division.
constexpr int kFastDecodeMaxRows = 128;

// Depends only on the packed row total, so the schedule is chosen before
// launch. That is a function of the captured shape alone, so under a graph it
// resolves once at capture.
constexpr bool is_decode_shape(int64_t total_rows) {
  return total_rows > 0 && total_rows <= kFastDecodeMaxRows;
}

// One record per active expert, in expert order, plus the active count in the
// slot past the last expert. Four int32 keeps a record a single 16-byte load.
struct PlanRecord {
  int expert;
  int row_offset;
  int scale_offset;
  int rows;
};

// The records follow the scan scratch, aligned so that a record is a single
// aligned 16-byte load.
constexpr std::size_t decode_records_offset(int groups) {
  return (plan_bytes(groups) + 15) & ~std::size_t(15);
}

constexpr std::size_t decode_shared_bytes(int groups) {
  return decode_records_offset(groups) +
         4 * (std::size_t(groups) + 1) * sizeof(int32_t);
}

// Build the compacted decode plan cooperatively, in SLM, inside the GEMM.
//
// A pre-pass kernel would compute this once instead of once per workgroup, but
// decode is captured into a graph, so every launch becomes a node replayed for
// each of the model's grouped GEMMs and costs more than the scan it saves.
//
// `scan` carries the prefixes; tile_prefix holds the active indicator here,
// since under the decode ceiling tile counts are uniform and never consulted.
template <typename Item>
CUTLASS_DEVICE void build_decode_plan(
    PlanStorage scan, int* records, const int* counts, int groups, Item item) {
  const int lid = int(item.get_local_linear_id());
  const int lrange = int(item.get_local_range().size());
  const int entries = groups + 1;
  if (entries > lrange) {
    if (lid == 0) {
      int32_t row = 0, scale = 0, active = 0;
      for (int expert = 0; expert < groups; ++expert) {
        const int rows = counts[expert];
        if (rows > 0) {
          int* record = records + 4 * active;
          record[0] = expert;
          record[1] = row;
          record[2] = scale;
          record[3] = rows;
          ++active;
        }
        row += int32_t(rows);
        scale += int32_t(padded_rows(rows));
      }
      records[4 * groups] = active;
    }
    item.barrier(sycl::access::fence_space::local_space);
    return;
  }
  // Entry 0 is the empty prefix, so work item i carries expert i - 1 and the
  // scan leaves each expert's exclusive prefix in its own slot.
  const bool owns = lid < entries;
  const bool live = owns && lid > 0;
  const int rows = live ? counts[lid - 1] : 0;
  if (owns) {
    scan.tile_prefix[lid] = rows > 0 ? 1 : 0;
    scan.row_prefix[lid] = int32_t(rows);
    scan.scale_prefix[lid] = int32_t(padded_rows(rows));
  }
  item.barrier(sycl::access::fence_space::local_space);
  for (int offset = 1; offset < entries; offset <<= 1) {
    const bool add = owns && lid >= offset;
    int32_t rank = 0, row = 0, scale = 0;
    if (add) {
      rank = scan.tile_prefix[lid - offset];
      row = scan.row_prefix[lid - offset];
      scale = scan.scale_prefix[lid - offset];
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (add) {
      scan.tile_prefix[lid] += rank;
      scan.row_prefix[lid] += row;
      scan.scale_prefix[lid] += scale;
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
  // Slot lid - 1 now holds the totals over experts below lid - 1, which is
  // exactly this expert's rank among the active ones and its two offsets.
  if (live && rows > 0) {
    int* record = records + 4 * scan.tile_prefix[lid - 1];
    record[0] = lid - 1;
    record[1] = scan.row_prefix[lid - 1];
    record[2] = scan.scale_prefix[lid - 1];
    record[3] = rows;
  }
  if (lid == 0) records[4 * groups] = scan.tile_prefix[entries - 1];
  item.barrier(sycl::access::fence_space::local_space);
}

// Decode-only scheduler. Under the 128-row ceiling every non-empty expert is
// Kind::Tiny, owns exactly one row tile and contributes the same number of
// column tiles, so scheduling collapses to one division per tile and none of
// the general scheduler's search or uniform state is needed.
class DecodeScheduler {
 public:
  DecodeScheduler(const int* plan, int groups, int n, int worker, int workers)
      : plan_(reinterpret_cast<const PlanRecord*>(plan)),
        n_tiles_(int((int64_t(n) + 511) / 512)),
        workers_(workers),
        linear_(worker) {
    total_tiles_ = int64_t(plan[4 * groups]) * n_tiles_;
  }

  Work next() {
    const int64_t global = linear_;
    if (global >= total_tiles_) return {-1, 0, 0, 0, 0, {}};
    linear_ = global + workers_;
    const int active = int(global) / n_tiles_;
    const int n_tile = int(global) - active * n_tiles_;
    const PlanRecord record = plan_[active];
    return {
        record.expert,
        record.rows,
        n_tile,
        int64_t(record.row_offset),
        int64_t(record.scale_offset),
        row_tile(Kind::Tiny, record.rows, 0)};
  }

 private:
  const PlanRecord* plan_;
  int n_tiles_;
  int workers_;
  int64_t linear_;
  int64_t total_tiles_;
};

// Counts remain on the device, staged into SLM by build_plan. Tiles are
// assigned from a workgroup-wide prefix sum, so finding the expert that owns
// a tile is a binary search over SLM rather than a walk over global memory.
template <int MidN, int NormalN = 512, bool OptimizeUniform = false>
class Scheduler {
 public:
  constexpr Scheduler(
      PlanStorage plan,
      const int* counts,
      int groups,
      int n,
      MidWindow window,
      int worker,
      int workers,
      bool uniform)
      : counts_(counts),
        tile_prefix_(plan.tile_prefix),
        row_prefix_(plan.row_prefix),
        scale_prefix_(plan.scale_prefix),
        groups_(groups),
        window_(window),
        worker_(worker),
        workers_(workers),
        uniform_(uniform),
        linear_(worker) {
    normal_n_tiles_ = (int64_t(n) + NormalN - 1) / NormalN;
    tiny_n_tiles_ = (int64_t(n) + 511) / 512;
    mid_n_tiles_ = (int64_t(n) + MidN - 1) / MidN;
    rows_ = counts_[0];
    if (!uniform_) {
      total_tiles_ = tile_prefix_[groups_];
      // Resolve the first tile here so next() starts with a valid cached
      // expert, seeding the lower bound for its searches.
      if (linear_ < total_tiles_) {
        int lo = 0, hi = groups_;
        while (lo + 1 < hi) {
          const int mid = (lo + hi) >> 1;
          if (tile_prefix_[mid] <= linear_)
            lo = mid;
          else
            hi = mid;
        }
        expert_ = lo;
        row_offset_ = row_prefix_[lo];
        scale_offset_ = scale_prefix_[lo];
        rows_ = int(row_prefix_[lo + 1]) - int(row_offset_);
      }
      return;
    }
    {
      expert_tiles_ = tiles(rows_);
      total_tiles_ = expert_tiles_ * groups_;
      full_waves_ = total_tiles_ / workers_;
      if (expert_tiles_ > 0 && (expert_tiles_ & (expert_tiles_ - 1)) == 0) {
        divisor_shift_ = 0;
        for (int64_t q = expert_tiles_; q > 1; q >>= 1)
          ++divisor_shift_;
      }
      if constexpr (OptimizeUniform) {
        // Amortize setup over multiple waves, and retain the wide decoder
        // when tile indices cannot safely use FastDivmod's signed input.
        fast_uniform_ = rows_ > 128 && total_tiles_ > workers_ &&
                        total_tiles_ <= std::numeric_limits<int>::max();
        if (fast_uniform_) {
          uniform_kind_ = classify(rows_, window_);
          expert_divmod_ = cutlass::FastDivmod(int(expert_tiles_));
          n_divmod_ = cutlass::FastDivmod(int(columns(uniform_kind_)));
          last_row_tile_ = row_tiles(uniform_kind_, rows_) - 1;
          last_tile_ = row_tile(uniform_kind_, rows_, last_row_tile_);
          penultimate_tile_ = row_tile(
              uniform_kind_,
              rows_,
              last_row_tile_ > 0 ? last_row_tile_ - 1 : 0);
        }
      }
    }
  }

  constexpr Work next() {
    int64_t local = linear_;
    if (uniform_) {
      const bool mirror = wave_ >= full_waves_ / 2 && wave_ < full_waves_;
      const int source = mirror ? workers_ - 1 - worker_ : worker_;
      local = wave_++ * workers_ + source;
      if (local >= total_tiles_) return {-1, 0, 0, 0, 0, {}};
      if constexpr (OptimizeUniform) {
        if (fast_uniform_) {
          int within, m_tile, n_tile;
          expert_divmod_(expert_, within, int(local));
          n_divmod_(m_tile, n_tile, within);
          const bool mid = uniform_kind_ == Kind::Mid;
          const int height = mid ? 320 : 256;
          const RowTile tile =
              m_tile == last_row_tile_ ? last_tile_
              : m_tile == last_row_tile_ - 1
                  ? penultimate_tile_
                  : RowTile{
                        mid ? Bucket::Mid320 : Bucket::Normal256,
                        m_tile * height,
                        height};
          return {
              expert_,
              rows_,
              n_tile,
              int64_t(expert_) * rows_,
              int64_t(expert_) * padded_rows(rows_),
              tile};
        }
      }
      expert_ =
          int(divisor_shift_ >= 0 ? local >> divisor_shift_
                                  : local / expert_tiles_);
      local -= int64_t(expert_) * expert_tiles_;
      row_offset_ = int64_t(expert_) * rows_;
      scale_offset_ = int64_t(expert_) * padded_rows(rows_);
    } else {
      const int64_t global = linear_;
      if (global >= total_tiles_) return {-1, 0, 0, 0, 0, {}};
      // Tiles are pulled in increasing order, so the owning expert never moves
      // backwards and the previous one is a valid lower bound. A linear step
      // is not viable: the grid strides by workers_ tiles, so consecutive
      // tiles for one workgroup can sit tens of experts apart.
      int lo = expert_;
      // Sticky check first: an expert spanning many tiles often owns the next
      // one too, and two reads confirm that against the seven a search costs.
      if (!(tile_prefix_[lo] <= global && global < tile_prefix_[lo + 1])) {
        int hi = groups_;
        while (lo + 1 < hi) {
          const int mid = (lo + hi) >> 1;
          if (tile_prefix_[mid] <= global)
            lo = mid;
          else
            hi = mid;
        }
        expert_ = lo;
        row_offset_ = row_prefix_[lo];
        scale_offset_ = scale_prefix_[lo];
        rows_ = int(row_prefix_[lo + 1]) - int(row_offset_);
      }
      local = global - tile_prefix_[lo];
      linear_ = global + workers_;
    }
    const Kind kind = classify(rows_, window_);
    const int64_t n_tiles = columns(kind);
    return {
        expert_,
        rows_,
        int(local % n_tiles),
        row_offset_,
        scale_offset_,
        row_tile(kind, rows_, int(local / n_tiles))};
  }

 private:
  constexpr int64_t tiles(int rows) const {
    const Kind kind = classify(rows, window_);
    return int64_t(row_tiles(kind, rows)) * columns(kind);
  }

  constexpr int64_t columns(Kind kind) const {
    if (kind == Kind::Tiny) return tiny_n_tiles_;
    return kind == Kind::Mid ? mid_n_tiles_ : normal_n_tiles_;
  }

  const int* counts_;
  const int32_t* tile_prefix_;
  const int32_t* row_prefix_;
  const int32_t* scale_prefix_;
  int groups_;
  MidWindow window_;
  int worker_, workers_;
  bool uniform_;
  int rows_ = 0, expert_ = 0, divisor_shift_ = -1;
  int64_t linear_, wave_ = 0;
  int64_t normal_n_tiles_, mid_n_tiles_, tiny_n_tiles_;
  int64_t expert_tiles_ = 0, total_tiles_ = 0, full_waves_ = 0;
  int64_t row_offset_ = 0, scale_offset_ = 0;
  bool fast_uniform_ = false;
  Kind uniform_kind_ = Kind::Empty;
  int last_row_tile_ = 0;
  RowTile penultimate_tile_{}, last_tile_{};
  cutlass::FastDivmod expert_divmod_, n_divmod_;
};

}  // namespace gpu::cutlass_kernel::grouped_gemm::greedy
