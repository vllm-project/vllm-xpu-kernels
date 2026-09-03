// Minimal Level Zero IPC interop for torch XPU tensors.  Host only: no
// device kernels here (the collectives live in p2p_collective.cpp).
//
// There is no Python API for cross-process XPU memory sharing, so a peer
// rank's device allocation is reached the Level Zero way: export the
// allocation as an IPC handle (on Linux that handle wraps a dma-buf fd),
// hand the fd to the peer over a unix socket with SCM_RIGHTS, and open it
// there into a device pointer.
//
// The Level Zero context and device are taken from the current XPU stream
// on every call, so memory opened here is valid in torch's own context and
// copies enqueued on that (in-order) queue are ordered against torch ops
// with no extra synchronization.  Deriving them per call rather than
// caching them at an init() call also removes the need for the caller to
// hand a sycl::queue across the language boundary.
//
// Raw device addresses cross this boundary as int64_t, as vLLM's
// custom_all_reduce does with `fptr_t`.
//
// Level Zero device addresses sit above 2**63, so they arrive here as the
// two's-complement int64 of the address (torch's `int` schema type cannot
// carry them otherwise). Casting that bit pattern back to a pointer is
// exact; vllm_xpu_kernels/p2p.py::as_fptr is the conversion on the caller
// side.

#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cstring>
#include <ios>
#include <tuple>

namespace {

#define ZE_CHECK(expr)              \
  do {                              \
    ze_result_t _r = (expr);        \
    TORCH_CHECK(                    \
        _r == ZE_RESULT_SUCCESS,    \
        #expr " failed: 0x",        \
        std::hex,                   \
        static_cast<unsigned>(_r)); \
  } while (0)

sycl::queue& current_queue() { return c10::xpu::getCurrentXPUStream().queue(); }

ze_context_handle_t current_ze_context() {
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      current_queue().get_context());
}

// Decodes the opaque bytes xpu_ipc_export_handle handed out back into a
// Level Zero handle.  Shared by open (which then substitutes the receiver's
// fd) and release (which needs the handle exactly as exported).
ze_ipc_mem_handle_t
handle_from_bytes(const torch::Tensor& handle_bytes, const char* what) {
  TORCH_CHECK(
      handle_bytes.device().is_cpu() && handle_bytes.is_contiguous() &&
          handle_bytes.scalar_type() == torch::kUInt8,
      what,
      ": handle_bytes must be a contiguous uint8 CPU tensor");
  ze_ipc_mem_handle_t handle;
  std::memset(&handle, 0, sizeof(handle));
  const size_t n =
      std::min(static_cast<size_t>(handle_bytes.numel()), sizeof(handle.data));
  std::memcpy(handle.data, handle_bytes.data_ptr(), n);
  return handle;
}

ze_device_handle_t current_ze_device() {
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      c10::xpu::get_raw_device(c10::xpu::current_device()));
}

// Marks the edge of what these collectives were validated on: single-tile
// devices (2x Arc B70).  It is not a claim that multi-tile cannot work.
//
// Why it is here: exporting a multi-tile allocation needs one IPC handle
// per tile, and Level Zero has no plural zeMemGetIpcHandle* to produce
// them.  The single handle below covers only part of such a region, so the
// peer would open a pointer that is valid for some of the range and not
// the rest -- silent corruption instead of an error.  Refusing is the
// honest answer until someone can test the alternative.
//
// To lift it: on a multi-tile part (e.g. PVC under the COMPOSITE device
// hierarchy), delete this function and its one call site in
// xpu_ipc_export_handle, then run tests/test_p2p_collective.py.  A
// per-tile export, if it turns out to be needed, belongs here too.
//
// Under FLAT, the current driver default, each tile is already its own
// root device, so this never fires there.
//
// tests/test_p2p_collective.py matches "single-tile" in the message below
// to skip rather than fail on such a part; keep the phrase if rewording.
void reject_untested_multi_tile_allocation(
    ze_context_handle_t ctx, const void* base) {
  ze_memory_allocation_properties_t alloc_props{};
  alloc_props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
  ze_device_handle_t alloc_device = nullptr;
  ZE_CHECK(zeMemGetAllocProperties(ctx, base, &alloc_props, &alloc_device));
  if (alloc_device == nullptr) {
    return;  // host or shared allocation: no tiles to span
  }
  uint32_t tiles = 0;
  ZE_CHECK(zeDeviceGetSubDevices(alloc_device, &tiles, nullptr));
  TORCH_CHECK(
      tiles <= 1,
      "xpu_ipc_export_handle: this allocation is on a device with ",
      tiles,
      " tiles, and the p2p collectives were only validated on single-tile "
      "devices.  Exporting a multi-tile allocation would need one Level "
      "Zero IPC handle per tile, which this code does not do, so it "
      "refuses rather than hand back a partially mapped pointer.  Set "
      "ZE_FLAT_DEVICE_HIERARCHY=FLAT to expose each tile as its own "
      "device, or lift the limit at "
      "reject_untested_multi_tile_allocation() in "
      "csrc/xpu/p2p/p2p_ipc.cpp.");
}

}  // namespace

// Exports the allocation containing `ptr`, returning
// (handle_bytes, dma_buf_fd, offset_of_ptr_within_the_allocation).
//
// The IPC handle covers the whole allocation, so the offset is what locates
// `ptr` inside it.  On Linux the handle wraps a dma-buf fd in its first 8
// bytes, and that fd is only valid in this process: the caller must pass it
// to the peer via SCM_RIGHTS, not as raw bytes.  It is returned separately
// for exactly that reason.
std::tuple<torch::Tensor, int64_t, int64_t> xpu_ipc_export_handle(int64_t ptr) {
  ze_context_handle_t ctx = current_ze_context();
  void* base = nullptr;
  size_t size = 0;
  ZE_CHECK(
      zeMemGetAddressRange(ctx, reinterpret_cast<void*>(ptr), &base, &size));

  reject_untested_multi_tile_allocation(ctx, base);

  ze_ipc_mem_handle_t handle;
  std::memset(&handle, 0, sizeof(handle));
  ZE_CHECK(zeMemGetIpcHandle(ctx, base, &handle));

  uint64_t fd = 0;
  std::memcpy(&fd, handle.data, sizeof(fd));

  auto out = torch::empty(
      {static_cast<int64_t>(sizeof(handle.data))},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  std::memcpy(out.data_ptr(), handle.data, sizeof(handle.data));

  const int64_t offset = static_cast<char*>(reinterpret_cast<void*>(ptr)) -
                         static_cast<char*>(base);
  return {out, static_cast<int64_t>(fd), offset};
}

// Releases an IPC handle obtained from xpu_ipc_export_handle.  Without this
// the exporting process keeps its dma-buf fd and the driver's export
// bookkeeping for the life of the process.  It does not free the underlying
// allocation, and it is not the counterpart of xpu_ipc_close_handle, which
// unmaps on the importing side.
//
// The driver may close the fd embedded in the handle as part of this, so
// the caller must NOT also close that fd: this call replaces it.
//
// Ordering: it drops the export reference that the peer's
// zeMemOpenIpcHandle resolves against, so it must not run until the peer
// has opened.  The fd itself would survive either way -- SCM_RIGHTS takes
// its own reference when the message is sent -- but the export reference is
// the part that matters.  The right moment is the all-ranks agreement that
// both sides opened successfully, which every caller already needs in order
// to know the mapping is live.
//
// A driver without zeMemPutIpcHandle is tolerated: this is cleanup, and
// failing setup over it would cost the whole p2p path to fix a bounded
// leak.
void xpu_ipc_release_handle(const torch::Tensor& handle_bytes) {
  ze_ipc_mem_handle_t handle =
      handle_from_bytes(handle_bytes, "xpu_ipc_release_handle");
  ze_result_t r = zeMemPutIpcHandle(current_ze_context(), handle);
  if (r == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    return;
  }
  TORCH_CHECK(
      r == ZE_RESULT_SUCCESS,
      "zeMemPutIpcHandle failed: 0x",
      std::hex,
      static_cast<unsigned>(r));
}

// Opens a peer's exported allocation and returns the device address of
// `offset` within it.  `fd` is the dma-buf fd as received over SCM_RIGHTS;
// it replaces the sender's (process-local, meaningless here) fd inside the
// handle before opening.
//
// The base address to hand back to xpu_ipc_close_handle is the return value
// minus `offset`.
int64_t xpu_ipc_open_handle(
    const torch::Tensor& handle_bytes, int64_t fd, int64_t offset) {
  ze_ipc_mem_handle_t handle =
      handle_from_bytes(handle_bytes, "xpu_ipc_open_handle");

  const uint64_t fd64 = static_cast<uint64_t>(fd);
  std::memcpy(handle.data, &fd64, sizeof(fd64));

  void* base = nullptr;
  ZE_CHECK(zeMemOpenIpcHandle(
      current_ze_context(), current_ze_device(), handle, 0, &base));
  return reinterpret_cast<int64_t>(static_cast<char*>(base) + offset);
}

// Closes a mapping opened by xpu_ipc_open_handle.  `base_ptr` must be the
// allocation base, i.e. what open returned minus the offset passed to it.
void xpu_ipc_close_handle(int64_t base_ptr) {
  ZE_CHECK(zeMemCloseIpcHandle(
      current_ze_context(), reinterpret_cast<void*>(base_ptr)));
}

// Enqueues a device-to-device copy on the current stream.  Takes raw
// addresses because the source is usually a peer pointer, which has no
// torch tensor behind it.
void xpu_p2p_memcpy(int64_t dst, int64_t src, int64_t nbytes) {
  TORCH_CHECK(nbytes >= 0, "xpu_p2p_memcpy: nbytes must be non-negative");
  if (nbytes == 0) {
    return;
  }
  current_queue().memcpy(
      reinterpret_cast<void*>(dst),
      reinterpret_cast<const void*>(src),
      static_cast<size_t>(nbytes));
}

// Host-synchronizes the current stream.  Much cheaper than
// torch.xpu.synchronize(), which measurably adds ~19us per call -- enough to
// matter to the host-synchronized p2p path that calls this once per
// collective.
void xpu_p2p_queue_sync() { current_queue().wait(); }
