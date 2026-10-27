// 2-rank all-reduce for XPU over peer memory mapped with Level Zero IPC
// (p2p_ipc.cpp).
//
// One kernel launch stages the local input where the peer can read it,
// handshakes through flags in device memory and reduces, like CUDA's custom
// all-reduce; the host only enqueues and never blocks on the peer.
//
// Each rank allocates its region with xpu_p2p_alloc_region and maps its
// peer's.  Layout, identical on both ranks, with F = align_up(2 * slot, 4 KiB)
// and one 64-byte line per workgroup on each signal page:
//
//   [0, 2 * slot_bytes)       two staging slots, alternated per launch
//   [F, F + 4 KiB)            flags the peer writes sequence numbers into
//   [F + 4 KiB, F + 8 KiB)    counters only this rank's launches touch
//
// The region is an allocation of its own, and the flag page holds nothing
// this device writes.  On 2x Arc B70, peer writes over PCIe disturbed this
// device's own writes elsewhere in the same allocation: carved out of torch's
// caching allocator, the region made torch ops on neighbouring tensors return
// wrong values now and then, and a slot tail on the flag page was corrupted
// on every full-slot call.
//
// The sequence numbers live in the counters and the kernel advances them
// itself, so a launch recorded into an XPU graph keeps making progress on
// replay.  A sequence number passed from the host would freeze at capture and
// silently produce wrong output.

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

constexpr size_t kWorkgroupSize = 256;
constexpr int kVec = 8;
// Workgroups of at least 2048 elements each, up to kMaxWorkgroups, stage,
// handshake and reduce in parallel; one workgroup made a 64 KiB call about 4x
// slower on 2x Arc B70.  16 is what a 64 KiB bf16 slot reaches.
constexpr int64_t kMaxWorkgroups = 16;
// Each workgroup's flag and counter get a cache line of their own: the peer
// writes flags over PCIe, and a shared line would bounce between workgroups.
constexpr int64_t kLine = 64;
constexpr int64_t kPage = 4096;
static_assert(kMaxWorkgroups * kLine <= kPage);

inline int64_t flags_offset(int64_t slot_bytes) {
  return (2 * slot_bytes + kPage - 1) / kPage * kPage;
}

// bf16 <-> fp32, bit-for-bit as the OpenCL C original computed them: the e2e
// evidence for this path is exact agreement with oneCCL.
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
// accumulation.  fp16 must widen explicitly: `a + b` on sycl::half adds in
// fp16 and rounds differently.
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

// Both sides of the handshake MUST be system-scope atomics.  An inbound PCIe
// write from the peer GPU does not invalidate this device's cache, so a plain
// volatile poll can spin on a stale line forever (on 2x Arc B70 it was still
// stuck after 100 s where an atomic poll saw the write at once).
// memory_scope::system lowers to SPIR-V CrossDevice; nothing narrower is
// correct, and getting it wrong fails rarely and silently.
using flag_ref = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::acq_rel,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>;

// Launches on an in-order queue serialize, so each sees the counters the
// previous one left and plain loads and stores suffice.  The sequence parity
// picks the staging slot.  That double buffer needs no release barrier: this
// rank reuses a slot at seq+2 only after its seq+1 handshake, which needs the
// peer's seq+1 signal, which the peer sends only after its launch seq, every
// read of this rank's slot included, has finished on its in-order queue.  So
// the launcher must keep enqueueing on the current stream's in-order queue.
template <typename Op>
struct p2p_all_reduce_kernel {
  using T = typename Op::T;

  T* dst;
  const T* input;
  T* my_stage;
  const T* peer_stage;
  uint32_t* flags;
  uint32_t* peer_flags;
  uint32_t* counters;
  size_t n;
  size_t chunk;
  size_t slot;
  sycl::local_accessor<uint32_t, 1> seq;

  void operator()(sycl::nd_item<1> item) const {
    const size_t wg = item.get_group(0);
    const size_t lid = item.get_local_id(0);
    const size_t stride = item.get_local_range(0) * kVec;
    const size_t line = wg * (kLine / sizeof(uint32_t));
    // launch_grid keeps wg * chunk < n, so end - start cannot underflow.
    const size_t start = wg * chunk;
    const size_t end = start + chunk < n ? start + chunk : n;
    const size_t vend = start + (end - start) / kVec * kVec;

    if (lid == 0) {
      seq[0] = ++counters[line];
    }
    sycl::group_barrier(item.get_group());
    const uint32_t s = seq[0];
    T* mine = my_stage + (s & 1) * slot;
    const T* peer = peer_stage + (s & 1) * slot;

    // Phase 1: stage this workgroup's chunk where the peer can read it.
    for (size_t i = start + lid * kVec; i < vend; i += stride) {
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

    // Handshake: publish s, then wait for the peer's.  The barrier extends
    // that ordering to every peer read below.  The compare wraps because the
    // counter is never reset: a plain `<` would stop waiting at the rollover
    // and reduce against a stale slot.
    if (lid == 0) {
      flag_ref(peer_flags[line]).store(s, sycl::memory_order::release);
      flag_ref local(flags[line]);
      while (static_cast<int32_t>(local.load(sycl::memory_order::acquire) - s) <
             0) {
      }
    }
    sycl::group_barrier(item.get_group());

    // Phase 2: reduce against the peer's staged chunk.
    for (size_t i = start + lid * kVec; i < vend; i += stride) {
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
};

namespace {

// Both ranks derive the identical grid from the element count alone, so peer
// workgroup wg publishes exactly the chunk this workgroup reads and
// per-workgroup flags are a sufficient handshake.  chunk is rounded up to the
// vector width so every workgroup's vector loop aligns the same way.
inline std::pair<int64_t, int64_t> launch_grid(int64_t n) {
  const int64_t want = std::min<int64_t>(kMaxWorkgroups, (n + 2047) / 2048);
  const int64_t nwg = std::max<int64_t>(1, want);
  const int64_t chunk = ((n + nwg - 1) / nwg + kVec - 1) / kVec * kVec;
  return {(n + chunk - 1) / chunk, chunk};
}

template <typename Op>
void submit_all_reduce(
    sycl::queue& q,
    torch::Tensor& out,
    const torch::Tensor& input,
    int64_t my_region,
    int64_t peer_region,
    int64_t slot_bytes) {
  using T = typename Op::T;
  char* mine = from_fptr<char>(my_region);
  char* peer = from_fptr<char>(peer_region);
  const int64_t f = flags_offset(slot_bytes);
  const std::pair<int64_t, int64_t> grid = launch_grid(input.numel());
  q.submit([&](sycl::handler& cgh) {
    p2p_all_reduce_kernel<Op> kernel{
        static_cast<T*>(out.data_ptr()),
        static_cast<const T*>(input.data_ptr()),
        reinterpret_cast<T*>(mine),
        reinterpret_cast<const T*>(peer),
        reinterpret_cast<uint32_t*>(mine + f),
        reinterpret_cast<uint32_t*>(peer + f),
        reinterpret_cast<uint32_t*>(mine + f + kPage),
        static_cast<size_t>(input.numel()),
        static_cast<size_t>(grid.second),
        static_cast<size_t>(slot_bytes / input.element_size()),
        sycl::local_accessor<uint32_t, 1>(sycl::range<1>(1), cgh)};
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(grid.first) * kWorkgroupSize),
            sycl::range<1>(kWorkgroupSize)),
        kernel);
  });
}

void check_slot_bytes(int64_t slot_bytes) {
  // A multiple of kLine keeps the flags and counters aligned for every dtype.
  TORCH_CHECK(
      slot_bytes > 0 && slot_bytes % kLine == 0,
      "xpu_p2p: slot_bytes (",
      slot_bytes,
      ") must be a positive multiple of ",
      kLine);
}

}  // namespace

int64_t region_bytes(int64_t slot_bytes) {
  check_slot_bytes(slot_bytes);
  return flags_offset(slot_bytes) + 2 * kPage;
}

}  // namespace p2p
}  // namespace xpu
}  // namespace vllm

void xpu_p2p_all_reduce(
    torch::Tensor& out,
    const torch::Tensor& input,
    int64_t my_region,
    int64_t peer_region,
    int64_t slot_bytes) {
  namespace p2p = vllm::xpu::p2p;
  const at::DeviceGuard device_guard(input.device());
  CHECK_DEVICE(input);
  CHECK_DEVICE(out);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(out);
  // The regions belong to one device and the kernel runs on input's queue.
  TORCH_CHECK(
      out.device().index() == input.device().index(),
      "xpu_p2p_all_reduce: out and input must be on the same XPU (out on ",
      out.device(),
      ", input on ",
      input.device(),
      ")");
  TORCH_CHECK(
      out.scalar_type() == input.scalar_type() && out.numel() == input.numel(),
      "xpu_p2p_all_reduce: out and input must have the same dtype and "
      "element count");

  // Both ranks see the same count, so both skip and stay in step.
  if (input.numel() == 0) {
    return;
  }

  p2p::check_slot_bytes(slot_bytes);
  TORCH_CHECK(
      ((my_region | peer_region) & (p2p::kPage - 1)) == 0,
      "xpu_p2p_all_reduce: regions must start on a page, as "
      "xpu_p2p_alloc_region returns them");
  const int64_t nbytes = input.numel() * input.element_size();
  TORCH_CHECK(
      nbytes <= slot_bytes,
      "xpu_p2p_all_reduce: input of ",
      nbytes,
      " bytes exceeds the ",
      slot_bytes,
      " byte staging slot");

  // During graph capture the current stream's queue is the recording one; a
  // launch on any other queue would run eagerly and be missing on replay.
  sycl::queue& q = vllm::xpu::vllmGetQueue(input.device().index());

  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      p2p::submit_all_reduce<p2p::AddBf16>(
          q, out, input, my_region, peer_region, slot_bytes);
      break;
    case at::ScalarType::Half:
      p2p::submit_all_reduce<p2p::AddF16>(
          q, out, input, my_region, peer_region, slot_bytes);
      break;
    case at::ScalarType::Float:
      p2p::submit_all_reduce<p2p::AddF32>(
          q, out, input, my_region, peer_region, slot_bytes);
      break;
    default:
      TORCH_CHECK(
          false,
          "xpu_p2p_all_reduce: unsupported dtype ",
          input.scalar_type(),
          " (supported: bfloat16, float16, float32)");
  }
}
