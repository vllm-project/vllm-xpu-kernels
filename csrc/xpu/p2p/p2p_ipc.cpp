// Minimal Level Zero IPC interop for torch XPU tensors (host only).
//
// There is no Python API for sharing XPU memory across processes, so an
// allocation is exported as a Level Zero IPC handle (a dma-buf fd on Linux),
// the fd travels to the peer over SCM_RIGHTS, and the peer opens it into a
// device pointer.  The context and device come from the current XPU stream
// on every call, so opened memory is valid in torch's own context.

#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstring>
#include <ios>
#include <tuple>

#include "xpu/p2p/p2p_fptr.h"

using vllm::xpu::p2p::from_fptr;
using vllm::xpu::p2p::to_fptr;

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

ze_context_handle_t current_ze_context() {
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      c10::xpu::getCurrentXPUStream().queue().get_context());
}

ze_device_handle_t current_ze_device() {
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      c10::xpu::get_raw_device(c10::xpu::current_device()));
}

ze_ipc_mem_handle_t
handle_from_bytes(const torch::Tensor& handle_bytes, const char* what) {
  // Exactly the handle size: a truncated or padded handle would only fail
  // later, inside open or release, far from its cause.
  TORCH_CHECK(
      handle_bytes.device().is_cpu() && handle_bytes.is_contiguous() &&
          handle_bytes.scalar_type() == torch::kUInt8 &&
          static_cast<size_t>(handle_bytes.numel()) ==
              sizeof(ze_ipc_mem_handle_t::data),
      what,
      ": handle_bytes must be a contiguous uint8 CPU tensor of ",
      sizeof(ze_ipc_mem_handle_t::data),
      " bytes");
  ze_ipc_mem_handle_t handle{};
  std::memcpy(handle.data, handle_bytes.data_ptr(), sizeof(handle.data));
  return handle;
}

// Validated on single-tile devices only.  A multi-tile allocation needs one
// IPC handle per tile, and the single handle exported below would map only
// part of it, so refuse rather than hand the peer a partially valid pointer.
// Under the default FLAT device hierarchy each tile is its own device and
// this does not fire.  tests/test_p2p_collective.py skips on "single-tile".
void reject_multi_tile_allocation(ze_context_handle_t ctx, const void* base) {
  ze_memory_allocation_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
  ze_device_handle_t device = nullptr;
  ZE_CHECK(zeMemGetAllocProperties(ctx, base, &props, &device));
  if (device == nullptr) {
    return;  // host or shared allocation
  }
  uint32_t tiles = 0;
  ZE_CHECK(zeDeviceGetSubDevices(device, &tiles, nullptr));
  TORCH_CHECK(
      tiles <= 1,
      "xpu_ipc_export_handle: allocation on a device with ",
      tiles,
      " tiles; only single-tile devices are supported "
      "(ZE_FLAT_DEVICE_HIERARCHY=FLAT exposes each tile as its own device)");
}

}  // namespace

// Returns (handle_bytes, dma_buf_fd, offset of ptr in its allocation).  The
// handle covers the whole allocation.  The fd is process-local and must reach
// the peer over SCM_RIGHTS, which is why it is returned separately.
std::tuple<torch::Tensor, int64_t, int64_t> xpu_ipc_export_handle(int64_t ptr) {
  ze_context_handle_t ctx = current_ze_context();
  void* base = nullptr;
  size_t size = 0;
  ZE_CHECK(zeMemGetAddressRange(ctx, from_fptr<void>(ptr), &base, &size));
  reject_multi_tile_allocation(ctx, base);

  ze_ipc_mem_handle_t handle{};
  ZE_CHECK(zeMemGetIpcHandle(ctx, base, &handle));
  uint64_t fd = 0;
  std::memcpy(&fd, handle.data, sizeof(fd));

  auto out = torch::empty(
      {static_cast<int64_t>(sizeof(handle.data))},
      torch::TensorOptions().dtype(torch::kUInt8));
  std::memcpy(out.data_ptr(), handle.data, sizeof(handle.data));

  const int64_t offset = from_fptr<char>(ptr) - static_cast<char*>(base);
  return {out, static_cast<int64_t>(fd), offset};
}

// Releases an exported handle once the peer has opened it; earlier would drop
// the export reference the peer's open resolves against.  The driver may
// close the embedded fd here, so the caller must not close it as well.  A
// driver without zeMemPutIpcHandle is tolerated: this is cleanup.
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

// Opens a peer's exported allocation and returns the address of `offset` in
// it.  `fd` is the dma-buf fd as received over SCM_RIGHTS; it replaces the
// sender's fd inside the handle.  xpu_ipc_close_handle takes the result minus
// `offset`.
int64_t xpu_ipc_open_handle(
    const torch::Tensor& handle_bytes, int64_t fd, int64_t offset) {
  ze_ipc_mem_handle_t handle =
      handle_from_bytes(handle_bytes, "xpu_ipc_open_handle");
  const uint64_t fd64 = static_cast<uint64_t>(fd);
  std::memcpy(handle.data, &fd64, sizeof(fd64));

  void* base = nullptr;
  ZE_CHECK(zeMemOpenIpcHandle(
      current_ze_context(), current_ze_device(), handle, 0, &base));
  return to_fptr(static_cast<char*>(base) + offset);
}

void xpu_ipc_close_handle(int64_t base_ptr) {
  ZE_CHECK(
      zeMemCloseIpcHandle(current_ze_context(), from_fptr<void>(base_ptr)));
}

// A zeroed region for a staging slot of slot_bytes, allocated straight from
// Level Zero so that it is an allocation of its own starting on a page:
// neither torch's caching allocator nor the SYCL USM pool may place other
// memory in it (see the layout comment in p2p_collective.cpp).
torch::Tensor xpu_p2p_alloc_region(int64_t slot_bytes) {
  const int64_t nbytes = vllm::xpu::p2p::region_bytes(slot_bytes);
  ze_context_handle_t ctx = current_ze_context();
  ze_device_mem_alloc_desc_t desc{};
  desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  void* ptr = nullptr;
  ZE_CHECK(zeMemAllocDevice(
      ctx,
      &desc,
      static_cast<size_t>(nbytes),
      4096,
      current_ze_device(),
      &ptr));
  auto region = torch::from_blob(
      ptr,
      {nbytes},
      [ctx](void* p) { zeMemFree(ctx, p); },
      torch::TensorOptions()
          .dtype(torch::kUInt8)
          .device(torch::Device(torch::kXPU, c10::xpu::current_device())));
  // Zeroed before any launch: the flags and counters hold sequence numbers
  // that only grow.
  region.zero_();
  c10::xpu::getCurrentXPUStream().synchronize();
  return region;
}
