// Device-side-synchronized 2-rank collectives (all-reduce, all-gather) for
// XPU, over peer memory mapped with Level Zero IPC (see p2p_ipc.cpp).
//
// One kernel per collective does everything CUDA's custom all-reduce does
// with Signal/RankSignals: publish the local chunk into the peer-visible
// staging buffer, handshake through per-workgroup flags, reduce.  The host
// only enqueues; it never blocks on the peer.  The all-gather kernel is the
// same protocol without the reduction.
//
// The caller owns every buffer these kernels touch: the staging slots, the
// flag page, the counter page and the peer pointers obtained over the
// dma-buf fd exchange.  They cross this boundary as raw device addresses in
// int64_t, the way vLLM's custom_all_reduce passes `fptr_t`.
//
// Level Zero device addresses sit above 2**63, so they arrive here as the
// two's-complement int64 of the address (torch's `int` schema type cannot
// carry them otherwise). Casting that bit pattern back to a pointer is
// exact; vllm_xpu_kernels/p2p.py::as_fptr is the conversion on the caller
// side.
//
// The kernels are graph-capture safe: they are submitted to whatever queue
// is current (the one recording, if any), and they carry no per-call state
// from the host.  The sequence number that drives the handshake and selects
// the staging slot is a per-workgroup counter in device memory that the
// kernel itself advances, so a replayed launch keeps making progress with
// its arguments frozen at capture time.  An earlier version passed a
// host-incremented sequence number as a kernel argument; it froze at
// capture and silently produced wrong output on replay.

#include <sycl/sycl.hpp>

#include <ATen/DeviceGuard.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <utility>

#include "utils.h"
#include "xpu/p2p/p2p_fptr.h"

namespace vllm {
namespace xpu {
namespace p2p {

// One workgroup per 2048 elements, capped.  The cap also fixes the size of
// the flag and counter pages the caller must allocate: kMaxWorkgroups
// entries at kLineStride uint32 each, i.e. 4096 bytes.  Callers should read
// that size back through xpu_p2p_signal_page_bytes() rather than hardcoding
// it.
constexpr int64_t kMaxWorkgroups = 64;
constexpr int64_t kWorkgroupSize = 256;

// Flags and counters get a 64-byte line each: the peer writes into this
// rank's flag page over PCIe, and false sharing between neighbouring
// workgroups would turn every signal into a line bounce.
constexpr int64_t kLineStride = 16;  // in uint32 elements

// ---------------------------------------------------------------------------
// bf16 <-> fp32, bit-for-bit as the OpenCL C original computed them.
//
// The reduce must stay bit-identical to that version: the e2e evidence for
// this path is that its output matches oneCCL exactly over thousands of
// tokens.  Both the 8-wide and the scalar tail of the original computed
// exactly the expression below per element -- the vector width only ever
// changed the memory access, never the arithmetic -- so one scalar helper
// reproduces both.
// ---------------------------------------------------------------------------

inline float bf2f(uint16_t h) {
  return sycl::bit_cast<float>(static_cast<uint32_t>(h) << 16);
}

inline uint16_t f2bf(float f) {
  uint32_t u = sycl::bit_cast<uint32_t>(f);
  if (sycl::isnan(f)) {
    return static_cast<uint16_t>((u >> 16) | 0x0040u);
  }
  u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
  return static_cast<uint16_t>(u >> 16);
}

// A single two-operand add rounds once, which is bit-identical to fp32
// accumulation for each of these dtypes.  The fp16 path must widen to fp32
// explicitly: `a + b` on sycl::half would add in fp16 and round differently.
struct AddBf16 {
  using T = uint16_t;
  static inline T apply(T a, T b) { return f2bf(bf2f(a) + bf2f(b)); }
};

struct AddF16 {
  using T = sycl::half;
  static inline T apply(T a, T b) {
    return static_cast<sycl::half>(
        static_cast<float>(a) + static_cast<float>(b));
  }
};

struct AddF32 {
  using T = float;
  static inline T apply(T a, T b) { return a + b; }
};

// ---------------------------------------------------------------------------
// Handshake
// ---------------------------------------------------------------------------

// One flag per workgroup, carrying monotonically increasing sequence
// numbers so the flags are never reset.
//
// Both sides of the exchange MUST be system-scope atomics.  A plain
// volatile load spins on a stale cache line forever, because an inbound
// PCIe write from the peer GPU does not invalidate this device's cache;
// only the coherent access a system-scope atomic compiles to observes it.
// Measured on 2x Arc B70: a volatile poll missed a mid-spin arrival even
// after 100 s, an atomic poll saw it immediately.  sycl::memory_scope::
// system is the SYCL spelling of OpenCL's memory_scope_all_svm_devices --
// both lower to SPIR-V CrossDevice.  Nothing narrower is correct here, and
// getting it wrong fails rarely and silently.
using flag_ref = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::acq_rel,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>;

// Work-item 0 publishes this rank's sequence number to the peer and waits
// for the peer's, then a workgroup barrier extends that ordering to the
// rest of the group: the release/acquire pair happens-before every peer
// read the other work-items go on to make.
inline void handshake(
    uint32_t* lflags,
    uint32_t* pflags,
    uint32_t seq,
    const sycl::nd_item<1>& item) {
  const size_t wg = item.get_group(0);
  if (item.get_local_id(0) == 0) {
    flag_ref peer(pflags[wg * kLineStride]);
    peer.store(seq, sycl::memory_order::release);
    flag_ref local(lflags[wg * kLineStride]);
    // Wrapping compare.  The counter is uint32 and never reset, so a plain
    // `<` stops waiting the moment it wraps: the peer sits at 0xFFFFFFFF
    // while seq is 0, the wait falls through and the reduce reads a stale
    // slot.  The signed difference is equivalent whenever the two are
    // within 2**31 of each other, which the protocol guarantees.
    while (static_cast<int32_t>(local.load(sycl::memory_order::acquire) - seq) <
           0) {
    }
  }
  sycl::group_barrier(item.get_group());
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Workgroup wg owns elements [wg*chunk, min(n, wg*chunk+chunk)).  Both
// ranks launch the identical grid (the grid is a function of the element
// count alone, and the collective contract makes the counts equal), so peer
// workgroup wg publishes exactly the chunk this workgroup reads:
// per-workgroup flags are a sufficient handshake and no kernel-wide barrier
// is needed.
//
// seq comes from a per-workgroup counter (own cache line, like the flags)
// that only this rank's kernels touch, so plain loads and stores suffice;
// launches on an in-order queue serialize, so each launch sees the value
// the previous one left.  Its parity picks one of two staging slots.  That
// double buffer needs no release barrier: this rank reuses a slot at seq+2
// only after its seq+1 handshake completed, which required the peer's seq+1
// signal, which the peer sends only after its own kernel seq -- every read
// of this rank's slot included -- finished on its in-order queue.  That
// argument is why the launcher must keep enqueueing on the current stream's
// in-order queue.
template <typename Op, int kVec>
class p2p_all_reduce_kernel {
  using T = typename Op::T;

 public:
  p2p_all_reduce_kernel(
      T* dst_,
      const T* input_,
      T* my_stage_,
      const T* peer_stage_,
      uint32_t* lflags_,
      uint32_t* pflags_,
      uint32_t* counters_,
      size_t n_,
      size_t chunk_,
      size_t slot_,
      sycl::local_accessor<uint32_t, 1> lseq_)
      : dst(dst_),
        input(input_),
        my_stage(my_stage_),
        peer_stage(peer_stage_),
        lflags(lflags_),
        pflags(pflags_),
        counters(counters_),
        n(n_),
        chunk(chunk_),
        slot(slot_),
        lseq(lseq_) {}

  void operator()(sycl::nd_item<1> item) const {
    const size_t wg = item.get_group(0);
    const size_t lid = item.get_local_id(0);
    const size_t ls = item.get_local_range(0);

    if (lid == 0) {
      const uint32_t s = counters[wg * kLineStride] + 1;
      counters[wg * kLineStride] = s;
      lseq[0] = s;
    }
    sycl::group_barrier(item.get_group());
    const uint32_t seq = lseq[0];

    T* mine = my_stage + (seq & 1) * slot;
    const T* peer = peer_stage + (seq & 1) * slot;

    // The grid is sized so that wg*chunk < n for every launched workgroup,
    // which is what keeps end - start below from underflowing.
    const size_t start = wg * chunk;
    const size_t end = (start + chunk < n) ? (start + chunk) : n;
    const size_t vend = start + ((end - start) / kVec) * kVec;

    // Phase 1: stage this rank's chunk where the peer can read it.
    for (size_t i = start + lid * kVec; i < vend; i += ls * kVec) {
      T v[kVec];
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        v[k] = input[i + k];
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        mine[i + k] = v[k];
    }
    if (lid == 0) {
      for (size_t i = vend; i < end; ++i)
        mine[i] = input[i];
    }
    sycl::group_barrier(item.get_group());

    handshake(lflags, pflags, seq, item);

    // Phase 2: reduce the local input against the peer's staged chunk.
    for (size_t i = start + lid * kVec; i < vend; i += ls * kVec) {
      T a[kVec], b[kVec];
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        a[k] = input[i + k];
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        b[k] = peer[i + k];
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        a[k] = Op::apply(a[k], b[k]);
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        dst[i + k] = a[k];
    }
    if (lid == 0) {
      for (size_t i = vend; i < end; ++i)
        dst[i] = Op::apply(input[i], peer[i]);
    }
  }

 private:
  T* dst;
  const T* input;
  T* my_stage;
  const T* peer_stage;
  uint32_t* lflags;
  uint32_t* pflags;
  uint32_t* counters;
  const size_t n;
  const size_t chunk;
  const size_t slot;
  sycl::local_accessor<uint32_t, 1> lseq;
};

// 2-rank all-gather along dim 0 of a contiguous input: pure placement, so it
// works on bytes and serves every dtype.  Phase 1 stages the local chunk for
// the peer and copies it into this rank's slice of dst; phase 2 copies the
// peer's staged chunk into the peer's slice.  Same per-workgroup handshake
// and double buffer as the all-reduce, on its own slots, flags and counters.
template <int kVec>
class p2p_all_gather_kernel {
 public:
  p2p_all_gather_kernel(
      uint8_t* dst_,
      const uint8_t* input_,
      uint8_t* my_stage_,
      const uint8_t* peer_stage_,
      uint32_t* lflags_,
      uint32_t* pflags_,
      uint32_t* counters_,
      size_t n_,
      size_t chunk_,
      size_t slot_,
      size_t my_off_,
      size_t peer_off_,
      sycl::local_accessor<uint32_t, 1> lseq_)
      : dst(dst_),
        input(input_),
        my_stage(my_stage_),
        peer_stage(peer_stage_),
        lflags(lflags_),
        pflags(pflags_),
        counters(counters_),
        n(n_),
        chunk(chunk_),
        slot(slot_),
        my_off(my_off_),
        peer_off(peer_off_),
        lseq(lseq_) {}

  void operator()(sycl::nd_item<1> item) const {
    const size_t wg = item.get_group(0);
    const size_t lid = item.get_local_id(0);
    const size_t ls = item.get_local_range(0);

    if (lid == 0) {
      const uint32_t s = counters[wg * kLineStride] + 1;
      counters[wg * kLineStride] = s;
      lseq[0] = s;
    }
    sycl::group_barrier(item.get_group());
    const uint32_t seq = lseq[0];

    uint8_t* mine_stage = my_stage + (seq & 1) * slot;
    const uint8_t* peer = peer_stage + (seq & 1) * slot;
    uint8_t* mine = dst + my_off;
    uint8_t* theirs = dst + peer_off;

    const size_t start = wg * chunk;
    const size_t end = (start + chunk < n) ? (start + chunk) : n;
    const size_t vend = start + ((end - start) / kVec) * kVec;

    for (size_t i = start + lid * kVec; i < vend; i += ls * kVec) {
      uint8_t v[kVec];
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        v[k] = input[i + k];
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        mine_stage[i + k] = v[k];
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        mine[i + k] = v[k];
    }
    if (lid == 0) {
      for (size_t i = vend; i < end; ++i) {
        const uint8_t v = input[i];
        mine_stage[i] = v;
        mine[i] = v;
      }
    }
    sycl::group_barrier(item.get_group());

    handshake(lflags, pflags, seq, item);

    for (size_t i = start + lid * kVec; i < vend; i += ls * kVec) {
      uint8_t v[kVec];
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        v[k] = peer[i + k];
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        theirs[i + k] = v[k];
    }
    if (lid == 0) {
      for (size_t i = vend; i < end; ++i)
        theirs[i] = peer[i];
    }
  }

 private:
  uint8_t* dst;
  const uint8_t* input;
  uint8_t* my_stage;
  const uint8_t* peer_stage;
  uint32_t* lflags;
  uint32_t* pflags;
  uint32_t* counters;
  const size_t n;
  const size_t chunk;
  const size_t slot;
  const size_t my_off;
  const size_t peer_off;
  sycl::local_accessor<uint32_t, 1> lseq;
};

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

namespace {

// Both ranks must compute the identical grid.  Deriving it here, from the
// element count alone, makes that true by construction rather than by two
// copies of a formula agreeing.  chunk is rounded up to the vector width so
// every workgroup's vector loop is aligned the same way.
inline std::pair<int64_t, int64_t> launch_grid(int64_t n, int64_t vec) {
  const int64_t cap = std::min<int64_t>(kMaxWorkgroups, (n + 2047) / 2048);
  const int64_t nwg0 = std::max<int64_t>(1, cap);
  const int64_t chunk = (((n + nwg0 - 1) / nwg0) + vec - 1) / vec * vec;
  return {(n + chunk - 1) / chunk, chunk};
}

template <typename Op, int kVec>
void submit_all_reduce(
    sycl::queue& q,
    void* dst,
    const void* input,
    int64_t my_stage,
    int64_t peer_stage,
    int64_t lflags,
    int64_t pflags,
    int64_t counters,
    int64_t n,
    int64_t chunk,
    int64_t slot,
    int64_t nwg) {
  using T = typename Op::T;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint32_t, 1> lseq(sycl::range<1>(1), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(nwg * kWorkgroupSize)),
            sycl::range<1>(static_cast<size_t>(kWorkgroupSize))),
        p2p_all_reduce_kernel<Op, kVec>(
            reinterpret_cast<T*>(dst),
            reinterpret_cast<const T*>(input),
            from_fptr<T>(my_stage),
            from_fptr<const T>(peer_stage),
            from_fptr<uint32_t>(lflags),
            from_fptr<uint32_t>(pflags),
            from_fptr<uint32_t>(counters),
            static_cast<size_t>(n),
            static_cast<size_t>(chunk),
            static_cast<size_t>(slot),
            lseq));
  });
}

}  // namespace

}  // namespace p2p
}  // namespace xpu
}  // namespace vllm

// Bytes the caller must allocate for each of the two signal pages (the
// per-workgroup flag page and the per-workgroup counter page).  Reading it
// back beats hardcoding it on the caller side: raising kMaxWorkgroups here
// would otherwise silently overrun a caller-sized page.
int64_t xpu_p2p_signal_page_bytes() {
  return vllm::xpu::p2p::kMaxWorkgroups * vllm::xpu::p2p::kLineStride *
         static_cast<int64_t>(sizeof(uint32_t));
}

void xpu_p2p_all_reduce(
    torch::Tensor& out,
    const torch::Tensor& input,
    int64_t my_stage,
    int64_t peer_stage,
    int64_t local_flags,
    int64_t peer_flags,
    int64_t counters,
    int64_t slot_bytes) {
  namespace p2p = vllm::xpu::p2p;
  const at::DeviceGuard device_guard(input.device());
  CHECK_DEVICE(input);
  CHECK_DEVICE(out);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(out);
  // The staging and signal pointers belong to one device, and the kernel is
  // launched on input's queue; an out on another XPU would be written from
  // the wrong device.
  TORCH_CHECK(
      out.device().index() == input.device().index(),
      "xpu_p2p_all_reduce: out and input must be on the same XPU (out on ",
      out.device(),
      ", input on ",
      input.device(),
      ")");
  TORCH_CHECK(
      out.scalar_type() == input.scalar_type(),
      "xpu_p2p_all_reduce: out and input must have the same dtype");
  TORCH_CHECK(
      out.numel() == input.numel(),
      "xpu_p2p_all_reduce: out and input must have the same element count");

  const int64_t n = input.numel();
  if (n == 0) {
    // Nothing to exchange.  Both ranks see the same count, so both skip the
    // launch and the handshake stays in step.
    return;
  }

  constexpr int kVec = 8;
  const int64_t elem = input.element_size();
  TORCH_CHECK(
      slot_bytes > 0 && slot_bytes % elem == 0,
      "xpu_p2p_all_reduce: slot_bytes (",
      slot_bytes,
      ") must be a positive multiple of the element size (",
      elem,
      ")");
  TORCH_CHECK(
      n * elem <= slot_bytes,
      "xpu_p2p_all_reduce: input of ",
      n * elem,
      " bytes exceeds the ",
      slot_bytes,
      " byte staging slot");

  const auto grid = p2p::launch_grid(n, kVec);
  const int64_t nwg = grid.first;
  const int64_t chunk = grid.second;
  const int64_t slot = slot_bytes / elem;

  // Submitted to the queue behind the *current* stream: during graph capture
  // that is the recording queue, and a launch on any other queue would run
  // eagerly and be missing from the replay.
  sycl::queue& q = vllm::xpu::vllmGetQueue(input.device().index());

  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      p2p::submit_all_reduce<p2p::AddBf16, kVec>(
          q,
          out.data_ptr(),
          input.data_ptr(),
          my_stage,
          peer_stage,
          local_flags,
          peer_flags,
          counters,
          n,
          chunk,
          slot,
          nwg);
      break;
    case at::ScalarType::Half:
      p2p::submit_all_reduce<p2p::AddF16, kVec>(
          q,
          out.data_ptr(),
          input.data_ptr(),
          my_stage,
          peer_stage,
          local_flags,
          peer_flags,
          counters,
          n,
          chunk,
          slot,
          nwg);
      break;
    case at::ScalarType::Float:
      p2p::submit_all_reduce<p2p::AddF32, kVec>(
          q,
          out.data_ptr(),
          input.data_ptr(),
          my_stage,
          peer_stage,
          local_flags,
          peer_flags,
          counters,
          n,
          chunk,
          slot,
          nwg);
      break;
    default:
      TORCH_CHECK(
          false,
          "xpu_p2p_all_reduce: unsupported dtype ",
          input.scalar_type(),
          " (supported: bfloat16, float16, float32)");
  }
}

void xpu_p2p_all_gather(
    torch::Tensor& out,
    const torch::Tensor& input,
    int64_t my_stage,
    int64_t peer_stage,
    int64_t local_flags,
    int64_t peer_flags,
    int64_t counters,
    int64_t slot_bytes,
    int64_t rank) {
  namespace p2p = vllm::xpu::p2p;
  const at::DeviceGuard device_guard(input.device());
  CHECK_DEVICE(input);
  CHECK_DEVICE(out);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(out);
  // The staging and signal pointers belong to one device, and the kernel is
  // launched on input's queue; an out on another XPU would be written from
  // the wrong device.
  TORCH_CHECK(
      out.device().index() == input.device().index(),
      "xpu_p2p_all_gather: out and input must be on the same XPU (out on ",
      out.device(),
      ", input on ",
      input.device(),
      ")");
  TORCH_CHECK(
      out.scalar_type() == input.scalar_type(),
      "xpu_p2p_all_gather: out and input must have the same dtype");
  TORCH_CHECK(
      out.numel() == 2 * input.numel(),
      "xpu_p2p_all_gather: out must hold exactly both ranks' shards");
  // The double-buffered handshake is written, and validated, for exactly two
  // ranks; the same limit is enforced on the vLLM side before a communicator
  // is built.
  TORCH_CHECK(
      rank == 0 || rank == 1,
      "xpu_p2p_all_gather: rank must be 0 or 1; these collectives support "
      "exactly two ranks, got rank ",
      rank);

  const int64_t n = input.numel() * input.element_size();  // bytes
  if (n == 0) {
    return;
  }

  constexpr int kVec = 16;
  TORCH_CHECK(
      n <= slot_bytes,
      "xpu_p2p_all_gather: input of ",
      n,
      " bytes exceeds the ",
      slot_bytes,
      " byte staging slot");

  const auto grid = p2p::launch_grid(n, kVec);
  const int64_t nwg = grid.first;
  const int64_t chunk = grid.second;

  sycl::queue& q = vllm::xpu::vllmGetQueue(input.device().index());
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint32_t, 1> lseq(sycl::range<1>(1), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(nwg * p2p::kWorkgroupSize)),
            sycl::range<1>(static_cast<size_t>(p2p::kWorkgroupSize))),
        p2p::p2p_all_gather_kernel<kVec>(
            reinterpret_cast<uint8_t*>(out.data_ptr()),
            reinterpret_cast<const uint8_t*>(input.data_ptr()),
            p2p::from_fptr<uint8_t>(my_stage),
            p2p::from_fptr<const uint8_t>(peer_stage),
            p2p::from_fptr<uint32_t>(local_flags),
            p2p::from_fptr<uint32_t>(peer_flags),
            p2p::from_fptr<uint32_t>(counters),
            static_cast<size_t>(n),
            static_cast<size_t>(chunk),
            static_cast<size_t>(slot_bytes),
            static_cast<size_t>(rank * n),
            static_cast<size_t>((1 - rank) * n),
            lseq));
  });
}
