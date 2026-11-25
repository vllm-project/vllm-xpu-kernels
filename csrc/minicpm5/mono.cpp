// SPDX-License-Identifier: Apache-2.0
#include "minicpm5/mono.h"
#include "minicpm5/sampling.h"

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <unistd.h>

#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

#include "utils.h"

namespace vllm::minicpm5::mono {
using at::Tensor;

static void ze_check(ze_result_t value, const char* operation) {
  TORCH_CHECK(value == ZE_RESULT_SUCCESS, operation, " failed: ", int(value));
}

static bool supported_on(const c10::xpu::XPUStream& stream) {
#if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) ||  \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED) || \
    defined(MINICPM5_MONO_REFERENCE_MOE_ENABLED)
  constexpr size_t RequiredSlm = 128 * 1024;
#else
  constexpr size_t RequiredSlm = MaxRows * H * sizeof(scalar_t);
#endif
  const auto& queue = stream.queue();
  if (queue.get_backend() != sycl::backend::ext_oneapi_level_zero ||
      !vllm::xpu::is_bmg(stream.device_index()) || !queue.is_in_order()) {
    return false;
  }
  const auto device = queue.get_device();
  const auto cores =
      device.get_info<sycl::ext::intel::info::device::gpu_slices>() *
      device
          .get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  return cores >= Groups &&
         device.get_info<sycl::info::device::max_work_group_size>() >= Local &&
         device.get_info<sycl::info::device::local_mem_size>() >= RequiredSlm;
}

bool supports_device(int64_t device_index) {
  if (device_index < 0 || device_index >= c10::xpu::device_count()) {
    return false;
  }
  return supported_on(
      c10::xpu::getCurrentXPUStream(
          static_cast<c10::DeviceIndex>(device_index)));
}

bool supported() {
  return c10::xpu::device_count() > 0 &&
         supported_on(c10::xpu::getCurrentXPUStream());
}

class Region : public torch::CustomClassHolder {
 public:
  explicit Region(int64_t bytes)
      : stream_(c10::xpu::getCurrentXPUStream()), bytes_(0) {
    TORCH_CHECK(
        bytes > 0 && bytes <= std::numeric_limits<int64_t>::max() - 64 &&
            supported_on(stream_) && !stream_.is_capturing(),
        "MiniCPM mono regions require Xe2 >=32 cores outside capture");
    bytes_ = bytes + 64;
#ifdef MINICPM5_MONO_REFERENCE_RUNTIME_ENABLED
    health_ = at::zeros(
        {8},
        at::TensorOptions().dtype(at::kInt).device(
            at::kXPU, stream_.device_index()));
#endif
    auto& queue = stream_.queue();
    context_ = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        queue.get_context());
    device_ = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        queue.get_device());
    ze_device_mem_alloc_desc_t descriptor{
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    void* pointer = nullptr;
    ze_check(
        zeMemAllocDevice(
            context_, &descriptor, bytes_, 4096, device_, &pointer),
        "zeMemAllocDevice");
    own_ = static_cast<char*>(pointer);
    try {
      queue.memset(own_, 0, bytes_).wait_and_throw();
      cookie_ = (uint64_t(getpid()) << 32) ^ uint64_t(uintptr_t(own_)) ^
                0x9e3779b97f4a7c15ull;
      queue.memcpy(own_, &cookie_, sizeof(cookie_)).wait_and_throw();
    } catch (...) {
      zeMemFree(context_, own_);
      own_ = nullptr;
      throw;
    }
  }

  ~Region() override {
    try {
      stream_.queue().wait_and_throw();
      if (peer_) zeMemCloseIpcHandle(context_, peer_);
      if (exported_) zeMemPutIpcHandle(context_, handle_);
      if (own_) zeMemFree(context_, own_);
      if (imported_fd_ >= 0) close(imported_fd_);
    } catch (...) {
      // Retain mappings if their queue failed rather than free live storage.
    }
  }

  std::vector<int64_t> export_info() {
    check_stream();
    TORCH_CHECK(!stream_.is_capturing(), "IPC export inside graph capture");
    if (!exported_) {
      ze_check(
          zeMemGetIpcHandle(context_, own_, &handle_), "zeMemGetIpcHandle");
      exported_ = true;
    }
    uint64_t fd = 0;
    if (zeMemGetFileDescriptorFromIpcHandleExp(context_, handle_, &fd) !=
        ZE_RESULT_SUCCESS) {
      int32_t value;
      std::memcpy(&value, handle_.data, sizeof(value));
      fd = value;
    }
    std::vector<int64_t> result{
        1,
        getpid(),
        int64_t(fd),
        int64_t(bytes_),
        int64_t(uint32_t(cookie_)),
        int64_t(uint32_t(cookie_ >> 32))};
    uint32_t words[ZE_MAX_IPC_HANDLE_SIZE / 4];
    std::memcpy(words, handle_.data, sizeof(words));
    for (auto word : words)
      result.push_back(word);
    return result;
  }

  void open(const std::vector<int64_t>& info, int64_t fd) {
    check_stream();
    TORCH_CHECK(
        !stream_.is_capturing() && !peer_ && fd >= 0 &&
            info.size() == 6 + ZE_MAX_IPC_HANDLE_SIZE / 4 && info[0] == 1 &&
            info[3] == bytes_ && info[1] != getpid(),
        "invalid MiniCPM peer IPC info");
    ze_ipc_mem_handle_t handle{};
    uint32_t words[ZE_MAX_IPC_HANDLE_SIZE / 4];
    for (size_t i = 0; i < ZE_MAX_IPC_HANDLE_SIZE / 4; ++i)
      words[i] = uint32_t(info[6 + i]);
    std::memcpy(handle.data, words, sizeof(words));
    const int32_t local = fd;
    std::memcpy(handle.data, &local, sizeof(local));
    // The exported Intel handle can embed the exporter PID alongside its fd.
    for (size_t i = 1; i < ZE_MAX_IPC_HANDLE_SIZE / 4; ++i)
      if (words[i] == uint32_t(info[1])) words[i] = uint32_t(getpid());
    std::memcpy(handle.data + 4, words + 1, sizeof(words) - 4);
    void* pointer = nullptr;
    ze_check(
        zeMemOpenIpcHandle(context_, device_, handle, 0, &pointer),
        "zeMemOpenIpcHandle");
    const uint64_t expected =
        uint32_t(info[4]) | (uint64_t(uint32_t(info[5])) << 32);
    auto& queue = stream_.queue();
    uint64_t* readback = sycl::malloc_shared<uint64_t>(1, queue);
    if (!readback) {
      zeMemCloseIpcHandle(context_, pointer);
      TORCH_CHECK(false, "peer IPC cookie readback allocation failed");
    }
    try {
      // Validate the actual device load used by the mailbox protocol. A SYCL
      // memcpy can select a copy engine that cannot access the imported peer.
      queue
          .single_task(
              [=]() { readback[0] = *static_cast<const uint64_t*>(pointer); })
          .wait_and_throw();
      const uint64_t actual = readback[0];
      sycl::free(readback, queue);
      readback = nullptr;
      TORCH_CHECK(actual == expected, "peer IPC cookie mismatch");
    } catch (...) {
      if (readback) sycl::free(readback, queue);
      zeMemCloseIpcHandle(context_, pointer);
      throw;
    }
    peer_ = static_cast<char*>(pointer);
    imported_fd_ = fd;
  }

  void check_stream() const {
    TORCH_CHECK(
        c10::xpu::getCurrentXPUStream() == stream_,
        "MiniCPM mono owner belongs to another XPU stream");
  }

  void rebind() {
    const auto current = c10::xpu::getCurrentXPUStream();
    auto& queue = current.queue();
    TORCH_CHECK(
        !current.is_capturing() && queue.is_in_order() &&
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                queue.get_context()) == context_ &&
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                queue.get_device()) == device_,
        "MiniCPM mono region stream transfer requires the same device/context "
        "outside capture");
    if (current != stream_) {
      stream_.queue().wait_and_throw();
      stream_ = current;
    }
  }

  Tensor health() const {
    TORCH_CHECK(
        health_.defined(), "Rebuild with reference mono runtime enabled");
    return health_;
  }

  void set_health(Tensor health) {
    check_stream();
    TORCH_CHECK(
        !stream_.is_capturing() && !health_frozen_ && health_.defined() &&
            health.is_xpu() && health.get_device() == stream_.device_index() &&
            health.scalar_type() == at::kInt && health.is_contiguous() &&
            health.sizes() == at::IntArrayRef({8}),
        "MiniCPM shared health must bind on its owner device outside capture");
    stream_.queue().wait_and_throw();
    health_ = std::move(health);
  }

  uint32_t* health_pointer() const {
    health_frozen_ = true;
    return health_.defined()
               ? reinterpret_cast<uint32_t*>(health_.data_ptr<int>())
               : nullptr;
  }

  uint64_t* pointer(int64_t slot, bool peer) const {
    TORCH_CHECK(
        slot >= 0 && uint64_t(slot) < uint64_t(bytes_ - 64) /
                                          (MailboxWords * sizeof(uint64_t)),
        "MiniCPM mono mailbox outside owned region");
    const size_t offset = size_t(slot) * MailboxWords * sizeof(uint64_t) + 64;
    auto* base = peer ? peer_ : own_;
    TORCH_CHECK(base, "MiniCPM mono peer is not mapped");
    return reinterpret_cast<uint64_t*>(base + offset);
  }

  uint64_t* words(int64_t bytes, bool peer) const {
    check_stream();
    TORCH_CHECK(
        bytes > 0 && bytes <= bytes_ - 64,
        "MiniCPM candidate mailbox exceeds owned region");
    auto* base = peer ? peer_ : own_;
    TORCH_CHECK(base, "MiniCPM candidate peer is not mapped");
    return reinterpret_cast<uint64_t*>(base + 64);
  }

 private:
  Tensor health_;
  mutable bool health_frozen_ = false;
  c10::xpu::XPUStream stream_;
  int64_t bytes_;
  ze_context_handle_t context_ = nullptr;
  ze_device_handle_t device_ = nullptr;
  ze_ipc_mem_handle_t handle_{};
  char *own_ = nullptr, *peer_ = nullptr;
  uint64_t cookie_ = 0;
  int imported_fd_ = -1;
  bool exported_ = false;
};

void check_tensor(
    const Tensor& tensor,
    at::ScalarType dtype,
    at::IntArrayRef shape,
    const char* name,
    int device) {
  TORCH_CHECK(
      tensor.is_xpu() && tensor.get_device() == device &&
          tensor.scalar_type() == dtype && tensor.sizes() == shape &&
          tensor.is_contiguous(),
      "MiniCPM mono ",
      name,
      ": expected contiguous ",
      dtype,
      " ",
      shape,
      " on XPU ",
      device,
      ", got ",
      tensor.scalar_type(),
      " ",
      tensor.sizes());
}

const uint8_t*
nk(const Tensor& tensor, int64_t n, int64_t k, const char* name, int device) {
  // Online XPU FP8 stores a transposed [K,N] view of physical [N,K].
  TORCH_CHECK(
      tensor.is_xpu() && tensor.get_device() == device &&
          tensor.scalar_type() == at::kFloat8_e4m3fn && tensor.dim() == 2 &&
          tensor.size(0) == k && tensor.size(1) == n && tensor.stride(0) == 1 &&
          tensor.stride(1) == k,
      "MiniCPM mono ",
      name,
      " requires [K,N] view of contiguous [N,K]: ",
      tensor.sizes(),
      " strides ",
      tensor.strides());
  return static_cast<const uint8_t*>(tensor.data_ptr());
}

const float* scale(const Tensor& tensor, int64_t elements, int device) {
  TORCH_CHECK(
      tensor.is_xpu() && tensor.get_device() == device &&
          tensor.scalar_type() == at::kFloat && tensor.numel() == elements &&
          tensor.is_contiguous(),
      "MiniCPM mono requires ",
      elements,
      " contiguous FP32 scale elements");
  return tensor.data_ptr<float>();
}

Weights check_weights(
    const std::vector<Tensor>& t,
    int tp,
    bool dense,
    int device,
    at::ScalarType dtype) {
  TORCH_CHECK(
      (t.size() == 16 || t.size() == 17) && (tp == 1 || tp == 2),
      "invalid MiniCPM mono weights/TP");
  const int qdim = 32 / tp * D, kvdim = 2 / tp * D, inter = 512 / tp;
  Weights w{};
  check_tensor(t[0], dtype, {H}, "input norm", device);
  check_tensor(t[1], dtype, {H}, "post norm", device);
  w.input_norm = reinterpret_cast<const scalar_t*>(t[0].data_ptr());
  w.post_norm = reinterpret_cast<const scalar_t*>(t[1].data_ptr());
  w.qkv = nk(t[2], 2 * qdim + 2 * kvdim, H, "QKV", device);
  if (t.size() == 17) nk(t[16], 2 * qdim + 2 * kvdim, H, "tiled QKV", device);
  w.qkv_scale = scale(t[3], 1, device);
  w.out = nk(t[4], H, qdim, "O projection", device);
  w.out_scale = scale(t[5], 1, device);
  if (dense) {
    w.w13 = nk(t[8], 2 * 8192 / tp, H, "dense gate/up", device);
    w.w2 = nk(t[10], H, 8192 / tp, "dense down", device);
    w.w13_scale = scale(t[9], 1, device);
    w.w2_scale = scale(t[11], 1, device);
  } else {
    check_tensor(t[6], dtype, {E, H}, "router", device);
    check_tensor(t[7], at::kFloat, {E}, "correction bias", device);
    check_tensor(
        t[8], at::kFloat8_e4m3fn, {E, H, 2 * inter}, "expert gate/up", device);
    check_tensor(
        t[10], at::kFloat8_e4m3fn, {E, inter, H}, "expert down", device);
    w.router = reinterpret_cast<const scalar_t*>(t[6].data_ptr());
    w.bias = t[7].data_ptr<float>();
    w.w13 = static_cast<const uint8_t*>(t[8].data_ptr());
    w.w2 = static_cast<const uint8_t*>(t[10].data_ptr());
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
    // Subgroup block-2D loads use each expert matrix as a cache-line-aligned
    // surface. Both expert strides are multiples of the same alignment.
    constexpr size_t Block2dAlignment = 64;
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(w.w13) % Block2dAlignment == 0 &&
            reinterpret_cast<uintptr_t>(w.w2) % Block2dAlignment == 0,
        "MiniCPM mono routed FP8 weights require 64-byte aligned bases");
#endif
    w.w13_scale = scale(t[9], E, device);
    w.w2_scale = scale(t[11], E, device);
    w.shared13 = nk(t[12], 2 * inter, H, "shared gate/up", device);
    w.shared13_scale = scale(t[13], 1, device);
    w.shared2 = nk(t[14], H, inter, "shared down", device);
    w.shared2_scale = scale(t[15], 1, device);
  }
  return w;
}

Workspace check_workspace(
    const std::vector<Tensor>& t, int tp, int device, at::ScalarType dtype) {
  TORCH_CHECK(t.size() == 10, "MiniCPM mono workspace needs 10 tensors");
  const int q = 32 / tp, qdim = q * D, inter = 512 / tp;
  const int stride = (Top + 1) * inter;
  check_tensor(t[0], dtype, {MaxRows, H}, "normed scratch", device);
  check_tensor(
      t[1],
      dtype,
      {MaxRows, 2 * qdim + 2 * (2 / tp) * D},
      "QKV scratch",
      device);
  check_tensor(t[2], dtype, {MaxRows, qdim}, "attention scratch", device);
  check_tensor(
      t[3],
      at::kFloat,
      {MaxRows, attention_partial_groups(tp), q, D + 2},
      "attention partials",
      device);
  check_tensor(t[4], dtype, {MaxRows, stride}, "MLP scratch", device);
  check_tensor(t[5], at::kFloat, {MaxRows, E}, "router scratch", device);
  check_tensor(t[6], at::kFloat, {MaxRows, Top}, "routing weights", device);
  check_tensor(t[7], at::kInt, {MaxRows, Top}, "expert IDs", device);
  check_tensor(
      t[8],
      at::kFloat,
      {MaxRows, Top, 2 * inter, 8},
      "gate/up partials",
      device);
  check_tensor(t[9], dtype, {MaxRows, Top + 1, H}, "expert outputs", device);
  return {
      reinterpret_cast<scalar_t*>(t[0].data_ptr()),
      reinterpret_cast<scalar_t*>(t[1].data_ptr()),
      reinterpret_cast<scalar_t*>(t[2].data_ptr()),
      reinterpret_cast<scalar_t*>(t[4].data_ptr()),
      reinterpret_cast<scalar_t*>(t[9].data_ptr()),
      t[3].data_ptr<float>(),
      t[5].data_ptr<float>(),
      t[6].data_ptr<float>(),
      t[8].data_ptr<float>(),
      t[7].data_ptr<int>()};
}

void validate(
    const std::vector<Tensor>& weights,
    const Tensor& cache,
    const std::vector<Tensor>& workspace,
    int64_t tp,
    bool dense) {
  TORCH_CHECK(
      supported(), "MiniCPM mono requires an in-order Xe2 GPU >=32 cores");
  TORCH_CHECK(tp == 1 || tp == 2, "MiniCPM mono TP1/TP2 only");
  const int device = c10::xpu::current_device();
  const auto dtype = cache.scalar_type();
  TORCH_CHECK(
      dtype == at::kBFloat16 || dtype == at::kHalf,
      "MiniCPM mono requires BF16 or FP16 activations/KV");
  check_weights(weights, tp, dense, device, dtype);
  check_workspace(workspace, tp, device, dtype);
  TORCH_CHECK(
      cache.is_xpu() && cache.get_device() == device &&
          cache.scalar_type() == dtype && cache.dim() == 4 &&
          cache.size(0) > 0 &&
          cache.size(0) <= std::numeric_limits<int>::max() &&
          cache.size(1) == 2 / tp && cache.size(2) > 0 &&
          cache.size(2) <= 32768 && cache.size(2) % 64 == 0 &&
          cache.size(3) == 2 * D &&
          (cache.is_contiguous() ||
           cache.permute({0, 2, 1, 3}).is_contiguous()),
      "MiniCPM mono requires packed BF16/FP16 cache "
      "[blocks,2/TP,block_size,256] "
      "with BHND or BNHD physical order, got ",
      cache.sizes(),
      " strides ",
      cache.strides());
}

void check_disjoint(
    const std::vector<Tensor>& writes, const std::vector<Tensor>& reads) {
  for (size_t i = 0; i < writes.size(); ++i) {
    for (size_t j = i + 1; j < writes.size(); ++j)
      TORCH_CHECK(
          at::get_overlap_status(writes[i], writes[j]) ==
              at::MemOverlapStatus::No,
          "MiniCPM mono mutable tensors overlap");
    for (const auto& read : reads)
      TORCH_CHECK(
          at::get_overlap_status(writes[i], read) == at::MemOverlapStatus::No,
          "MiniCPM mono mutable tensor overlaps an input or weight");
  }
}

void decode(
    const Tensor& hidden,
    const c10::optional<Tensor>& residual,
    const Tensor& positions,
    const Tensor& blocks,
    const Tensor& lengths,
    const Tensor& slots,
    Tensor output,
    Tensor res_out,
    std::vector<Tensor> workspace,
    Tensor control,
    std::vector<Tensor> tensors,
    Tensor cache,
    const c10::intrusive_ptr<Region>& region,
    int64_t mailbox_slot,
    int64_t tp,
    bool dense,
    bool ignore_padding,
    const c10::optional<Tensor>& rope_cache) {
  region->check_stream();
  TORCH_CHECK(
      hidden.dim() == 2 && (tp == 1 || tp == 2),
      "invalid MiniCPM decode shape/TP");
  const int rows = hidden.size(0), device = hidden.get_device();
  const auto dtype = hidden.scalar_type();
  TORCH_CHECK(
      dtype == at::kBFloat16 || dtype == at::kHalf,
      "MiniCPM mono requires BF16 or FP16 activations");
  TORCH_CHECK(
      cache.scalar_type() == dtype,
      "MiniCPM mono hidden and KV cache precision must match");
  TORCH_CHECK(rows >= 1 && rows <= 8, "MiniCPM mono rows1..8 only");
  TORCH_CHECK(
      !ignore_padding || (rows == 4 || rows == 8),
      "MiniCPM mono padding mode requires physical rows4 or8");
  check_tensor(hidden, dtype, {rows, H}, "hidden", device);
  check_tensor(output, dtype, {rows, H}, "output", device);
  check_tensor(res_out, dtype, {rows, H}, "residual output", device);
  if (residual) check_tensor(*residual, dtype, {rows, H}, "residual", device);
  check_tensor(positions, at::kLong, {rows}, "positions", device);
  check_tensor(slots, at::kLong, {rows}, "slot mapping", device);
  if (rope_cache)
    TORCH_CHECK(
        rope_cache->is_xpu() && rope_cache->get_device() == device &&
            rope_cache->scalar_type() == dtype && rope_cache->dim() == 2 &&
            rope_cache->size(0) > 0 && rope_cache->size(1) == D &&
            rope_cache->is_contiguous(),
        "MiniCPM mono requires contiguous activation-dtype RoPE cache [N,128] "
        "on the hidden device");
  TORCH_CHECK(
      lengths.is_xpu() && lengths.get_device() == device &&
          lengths.scalar_type() == at::kInt && lengths.is_contiguous() &&
          lengths.numel() >= rows,
      "invalid MiniCPM sequence lengths");
  const bool shared_pages = rows == 8 && dtype == at::kHalf &&
                            blocks.dim() == 2 && blocks.size(0) == 1;
  TORCH_CHECK(
      blocks.is_xpu() && blocks.get_device() == device &&
          blocks.scalar_type() == at::kInt && blocks.dim() == 2 &&
          (blocks.size(0) >= rows || shared_pages) && blocks.size(1) > 0 &&
          blocks.stride(1) == 1 && blocks.stride(0) >= blocks.size(1) &&
          blocks.stride(0) <= std::numeric_limits<int>::max(),
      "invalid MiniCPM block table");
  check_tensor(control, at::kInt, {ControlWords}, "control", device);
  // Validation remains host-only under capture and precedes the first submit.
  validate(tensors, cache, workspace, tp, dense);
  auto weights = check_weights(tensors, tp, dense, device, dtype);
  auto work = check_workspace(workspace, tp, device, dtype);
  auto writes = workspace;
  writes.insert(writes.end(), {output, res_out, control, cache});
  auto reads = tensors;
  reads.insert(reads.end(), {hidden, positions, blocks, lengths, slots});
  if (residual) reads.push_back(*residual);
  if (rope_cache) reads.push_back(*rope_cache);
  check_disjoint(writes, reads);
  Step step{
      reinterpret_cast<const scalar_t*>(hidden.data_ptr()),
      residual ? reinterpret_cast<const scalar_t*>(residual->data_ptr())
               : nullptr,
      reinterpret_cast<scalar_t*>(output.data_ptr()),
      reinterpret_cast<scalar_t*>(res_out.data_ptr()),
      reinterpret_cast<scalar_t*>(cache.data_ptr()),
      positions.data_ptr<int64_t>(),
      slots.data_ptr<int64_t>(),
      blocks.data_ptr<int>(),
      lengths.data_ptr<int>(),
      int(cache.size(2)),
      shared_pages ? 0 : int(blocks.stride(0)),
      cache.size(0),
      blocks.size(1),
      cache.stride(0),
      cache.stride(1),
      cache.stride(2),
      reinterpret_cast<uint32_t*>(control.data_ptr<int>()),
      region->pointer(mailbox_slot, false),
      tp == 2 ? region->pointer(mailbox_slot, true) : nullptr,
      region->health_pointer(),
      rope_cache ? reinterpret_cast<const scalar_t*>(rope_cache->data_ptr())
                 : nullptr,
      rope_cache ? rope_cache->size(0) : 0};
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  if (dtype == at::kHalf) {
    if (tensors.size() == 17 && rows == 1) {
      weights.qkv =
          nk(tensors[16], (32 + 2) * 2 * D / tp, H, "tiled QKV", device);
      mono_fp16::dispatch_tiled_qkv(
          queue,
          weights.as<sycl::half>(),
          work.as<sycl::half>(),
          step.as<sycl::half>(),
          tp,
          dense);
      return;
    }
    mono_fp16::dispatch(
        queue,
        weights.as<sycl::half>(),
        work.as<sycl::half>(),
        step.as<sycl::half>(),
        tp,
        rows,
        dense,
        ignore_padding);
    return;
  }
  if (tensors.size() == 17 && tp == 2 && rows == 1) {
    weights.qkv = nk(tensors[16], 4352, H, "tiled QKV", device);
    if (dense)
      launch<2, 1, true, false, true>(queue, weights, work, step);
    else
      launch<2, 1, false, false, true>(queue, weights, work, step);
    return;
  }
#define LAUNCH(T, R) \
  dispatch_layer<T, R>(queue, weights, work, step, dense, ignore_padding)
#define ROWS(T)     \
  switch (rows) {   \
    case 1: {       \
      LAUNCH(T, 1); \
      break;        \
    }               \
    case 2: {       \
      LAUNCH(T, 2); \
      break;        \
    }               \
    case 4: {       \
      LAUNCH(T, 4); \
      break;        \
    }               \
    case 3: {       \
      LAUNCH(T, 3); \
      break;        \
    }               \
    case 5: {       \
      LAUNCH(T, 5); \
      break;        \
    }               \
    case 6: {       \
      LAUNCH(T, 6); \
      break;        \
    }               \
    case 7: {       \
      LAUNCH(T, 7); \
      break;        \
    }               \
    case 8: {       \
      LAUNCH(T, 8); \
      break;        \
    }               \
  }
  if (tp == 1) {
    ROWS(1);
  } else {
    ROWS(2);
  }
#undef ROWS
#undef LAUNCH
}
#if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) &&  \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED) && \
    defined(MINICPM5_MONO_REFERENCE_MOE_ENABLED) &&        \
    defined(MINICPM5_MONO_DPAS_ENABLED)
template <int Rows>
void decode_tail(
    const Tensor& normed,
    const Tensor& slots,
    Tensor output,
    std::vector<Tensor> workspace,
    Tensor control,
    const std::vector<Tensor>& tensors,
    const c10::intrusive_ptr<Region>& region,
    int64_t mailbox_slot) {
  region->check_stream();
  TORCH_CHECK(
      supported(), "MiniCPM mono tail requires an in-order Xe2 GPU >=32 cores");
  const int device = c10::xpu::current_device();
  constexpr auto dtype = at::kHalf;
  static_assert(Rows == 4 || Rows == MaxRows);
  check_tensor(normed, dtype, {Rows, H}, "post-attention activation", device);
  check_tensor(slots, at::kLong, {Rows}, "slot mapping", device);
  check_tensor(output, dtype, {Rows, H}, "output", device);
  check_tensor(control, at::kInt, {ControlWords}, "control", device);
  auto weights = check_weights(tensors, 1, false, device, dtype);
  auto work = check_workspace(workspace, 1, device, dtype);
  auto writes = workspace;
  writes.insert(writes.end(), {output, control});
  auto reads = tensors;
  reads.insert(reads.end(), {normed, slots});
  check_disjoint(writes, reads);
  Step step{};
  step.hidden = reinterpret_cast<const scalar_t*>(normed.data_ptr());
  step.output = reinterpret_cast<scalar_t*>(output.data_ptr());
  step.slots = slots.data_ptr<int64_t>();
  step.control = reinterpret_cast<uint32_t*>(control.data_ptr<int>());
  step.mailbox = region->pointer(mailbox_slot, false);
  step.health = region->health_pointer();
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  if constexpr (Rows == 4)
    mono_fp16::dispatch_tail_rows4(
        queue,
        weights.as<sycl::half>(),
        work.as<sycl::half>(),
        step.as<sycl::half>());
  else
    mono_fp16::dispatch_tail(
        queue,
        weights.as<sycl::half>(),
        work.as<sycl::half>(),
        step.as<sycl::half>());
}

template <bool PostO>
void decode_attention_tail(
    const Tensor& activation,
    const Tensor& residual,
    const Tensor& slots,
    Tensor output,
    Tensor res_out,
    std::vector<Tensor> workspace,
    Tensor control,
    const std::vector<Tensor>& tensors,
    const c10::intrusive_ptr<Region>& region,
    int64_t mailbox_slot) {
  region->check_stream();
  TORCH_CHECK(
      supported(),
      "MiniCPM mono attention tail requires an in-order Xe2 GPU >=32 cores");
  const int device = c10::xpu::current_device();
  constexpr auto dtype = at::kHalf;
  check_tensor(
      activation,
      dtype,
      {MaxRows, H},
      PostO ? "O projection partial" : "gated attention",
      device);
  check_tensor(residual, dtype, {MaxRows, H}, "rounded input residual", device);
  check_tensor(slots, at::kLong, {MaxRows}, "slot mapping", device);
  check_tensor(output, dtype, {MaxRows, H}, "output", device);
  check_tensor(res_out, dtype, {MaxRows, H}, "output residual", device);
  check_tensor(control, at::kInt, {ControlWords}, "control", device);
  auto weights = check_weights(tensors, 2, false, device, dtype);
  auto work = check_workspace(workspace, 2, device, dtype);
  auto writes = workspace;
  writes.insert(writes.end(), {output, res_out, control});
  auto reads = tensors;
  reads.insert(reads.end(), {activation, residual, slots});
  check_disjoint(writes, reads);
  Step step{};
  // The split specialization reads hidden only as the supplied activation.
  // Residual is the already rounded base added after AR1.
  step.hidden = reinterpret_cast<const scalar_t*>(activation.data_ptr());
  step.residual = reinterpret_cast<const scalar_t*>(residual.data_ptr());
  step.slots = slots.data_ptr<int64_t>();
  step.output = reinterpret_cast<scalar_t*>(output.data_ptr());
  step.res_out = reinterpret_cast<scalar_t*>(res_out.data_ptr());
  step.control = reinterpret_cast<uint32_t*>(control.data_ptr<int>());
  step.mailbox = region->pointer(mailbox_slot, false);
  step.peer = region->pointer(mailbox_slot, true);
  step.health = region->health_pointer();
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  if constexpr (PostO)
    mono_fp16::dispatch_post_o(
        queue,
        weights.as<sycl::half>(),
        work.as<sycl::half>(),
        step.as<sycl::half>());
  else
    mono_fp16::dispatch_pre_o(
        queue,
        weights.as<sycl::half>(),
        work.as<sycl::half>(),
        step.as<sycl::half>());
}
#endif

#if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) && \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED)
void draft_decode(
    const Tensor& hidden,
    const c10::optional<Tensor>& residual,
    const Tensor& positions,
    const Tensor& blocks,
    const Tensor& lengths,
    const Tensor& slots,
    Tensor output,
    Tensor res_out,
    std::vector<Tensor> workspace,
    Tensor control,
    const std::vector<Tensor>& tensors,
    Tensor cache,
    const c10::intrusive_ptr<Region>& region,
    int64_t mailbox_slot,
    const Tensor& rope) {
  region->check_stream();
  TORCH_CHECK(
      cache.dim() == 4 && (cache.size(1) == 1 || cache.size(1) == 2),
      "MiniCPM draft KV heads must match TP1 or TP2");
  const int tp = 2 / cache.size(1);
  const int QDim = 32 / tp * D, QKV = (32 + 4) / tp * D;
  const int Intermediate = 8192 / tp;
  constexpr int R = DraftRows;
  const int device = hidden.get_device();
  constexpr auto dtype = at::kHalf;
  check_tensor(hidden, dtype, {R, H}, "draft hidden", device);
  if (residual)
    check_tensor(*residual, dtype, {R, H}, "draft residual", device);
  check_tensor(output, dtype, {R, H}, "draft output", device);
  check_tensor(res_out, dtype, {R, H}, "draft residual output", device);
  check_tensor(positions, at::kLong, {R}, "draft positions", device);
  check_tensor(slots, at::kLong, {R}, "draft slot mapping", device);
  check_tensor(lengths, at::kInt, {R}, "draft context lengths", device);
  check_tensor(control, at::kInt, {ControlWords}, "draft control", device);
  TORCH_CHECK(
      blocks.is_xpu() && blocks.get_device() == device &&
          blocks.scalar_type() == at::kInt && blocks.dim() == 2 &&
          blocks.size(0) == R && blocks.size(1) > 0 && blocks.stride(1) == 1 &&
          blocks.stride(0) >= blocks.size(1) &&
          blocks.stride(0) <= std::numeric_limits<int>::max(),
      "MiniCPM mono draft requires one page table per query row");
  TORCH_CHECK(
      cache.is_xpu() && cache.get_device() == device &&
          cache.scalar_type() == dtype && cache.dim() == 4 &&
          cache.size(0) > 0 &&
          cache.size(0) <= std::numeric_limits<int>::max() &&
          cache.size(1) == 2 / tp && cache.size(2) >= 64 &&
          cache.size(2) <= 32768 && cache.size(2) % 64 == 0 &&
          cache.size(3) == 2 * D &&
          (cache.is_contiguous() ||
           cache.permute({0, 2, 1, 3}).is_contiguous()),
      "MiniCPM mono draft requires a contiguous TP1/TP2 FP16 K/V cache");
  TORCH_CHECK(
      rope.is_xpu() && rope.get_device() == device &&
          rope.scalar_type() == dtype && rope.dim() == 2 && rope.size(0) > 0 &&
          rope.size(1) == D && rope.is_contiguous(),
      "invalid MiniCPM draft RoPE cache");
  TORCH_CHECK(tensors.size() == 8, "MiniCPM mono draft needs eight weights");
  check_tensor(tensors[0], dtype, {H}, "draft input norm", device);
  check_tensor(tensors[1], dtype, {H}, "draft post norm", device);
  check_tensor(tensors[2], dtype, {D}, "draft query norm", device);
  check_tensor(tensors[3], dtype, {D}, "draft key norm", device);
  check_tensor(
      tensors[4], dtype, {QKV / 16, H / 16, 8, 16, 2}, "draft QKV", device);
  check_tensor(
      tensors[5],
      dtype,
      {H / 16, QDim / 16, 8, 16, 2},
      "draft out projection",
      device);
  check_tensor(
      tensors[6],
      dtype,
      {2 * Intermediate / 16, H / 16, 8, 16, 2},
      "draft gate up",
      device);
  check_tensor(
      tensors[7],
      dtype,
      {H / 16, Intermediate / 16, 8, 16, 2},
      "draft down",
      device);
  const auto half_data = [](const Tensor& tensor) {
    return static_cast<const sycl::half*>(tensor.data_ptr());
  };
  for (size_t i = 4; i < tensors.size(); ++i)
    TORCH_CHECK(
        uintptr_t(tensors[i].data_ptr()) % 32 == 0,
        "MiniCPM mono draft projection weights must be 32-byte aligned");
  const DraftWeights<sycl::half> weights{
      half_data(tensors[0]),
      half_data(tensors[1]),
      half_data(tensors[2]),
      half_data(tensors[3]),
      half_data(tensors[4]),
      half_data(tensors[5]),
      half_data(tensors[6]),
      half_data(tensors[7])};
  auto work = check_workspace(workspace, tp, device, dtype).as<sycl::half>();
  auto writes = workspace;
  writes.insert(writes.end(), {output, res_out, control, cache});
  auto reads = tensors;
  reads.insert(reads.end(), {hidden, positions, blocks, lengths, slots, rope});
  if (residual) reads.push_back(*residual);
  check_disjoint(writes, reads);
  const StepPointers<sycl::half> step{
      half_data(hidden),
      residual ? half_data(*residual) : nullptr,
      static_cast<sycl::half*>(output.data_ptr()),
      static_cast<sycl::half*>(res_out.data_ptr()),
      static_cast<sycl::half*>(cache.data_ptr()),
      positions.data_ptr<int64_t>(),
      slots.data_ptr<int64_t>(),
      blocks.data_ptr<int>(),
      lengths.data_ptr<int>(),
      int(cache.size(2)),
      int(blocks.stride(0)),
      cache.size(0),
      blocks.size(1),
      cache.stride(0),
      cache.stride(1),
      cache.stride(2),
      reinterpret_cast<uint32_t*>(control.data_ptr<int>()),
      region->pointer(mailbox_slot, false),
      tp == 2 ? region->pointer(mailbox_slot, true) : nullptr,
      region->health_pointer(),
      half_data(rope),
      rope.size(0)};
  mono_fp16::dispatch_draft(
      c10::xpu::getCurrentXPUStream().queue(), weights, work, step, tp);
}
#endif

int64_t candidate_mailbox_bytes() {
  return 2 * sampling::BMAX * sampling::KMAX * 2 * sizeof(uint64_t);
}

void candidate_topk_impl(
    Tensor& values,
    Tensor& indices,
    const Tensor& logits,
    int64_t k,
    int64_t rank,
    Tensor& partial,
    Tensor& control,
    const c10::intrusive_ptr<Region>& region,
    Tensor* sampled = nullptr,
    Tensor* num_sampled = nullptr,
    Tensor* num_rejected = nullptr,
    const Tensor* bias = nullptr) {
  TORCH_CHECK(
      logits.dim() == 2 && logits.size(0) >= 1 &&
          logits.size(0) <= sampling::BMAX && logits.size(1) == 65280,
      "MiniCPM candidate logits require [1..8, 65280] TP2 shards");
  TORCH_CHECK(
      k >= 1 && k <= sampling::KMAX && (rank == 0 || rank == 1),
      "MiniCPM candidates require k1..64 and TP2 rank0/1");
  const int device = logits.get_device();
  TORCH_CHECK(
      device == c10::xpu::getCurrentXPUStream().device_index() && supported(),
      "MiniCPM candidates require the current qualified XPU");
  TORCH_CHECK(
      logits.scalar_type() == at::kBFloat16 ||
          logits.scalar_type() == at::kHalf,
      "MiniCPM candidates require BF16/FP16 logits");
  check_tensor(
      logits, logits.scalar_type(), logits.sizes(), "candidate logits", device);
  check_tensor(
      values, at::kFloat, {logits.size(0), 128}, "candidate values", device);
  check_tensor(indices, at::kInt, values.sizes(), "candidate indices", device);
  check_tensor(
      partial,
      at::kLong,
      {sampling::BMAX, sampling::NW, sampling::KMAX},
      "candidate partial",
      device);
  check_tensor(control, at::kInt, {ControlWords}, "candidate control", device);
  if (bias) {
    TORCH_CHECK(
        sampled && logits.scalar_type() == at::kHalf,
        "MiniCPM draft candidate bias requires direct FP16 greedy sampling");
    check_tensor(
        *bias, at::kHalf, logits.sizes(), "draft candidate bias", device);
  }
  if (sampled) {
    TORCH_CHECK(
        k == 1 && num_sampled && num_rejected,
        "MiniCPM direct greedy requires K1 and all output tensors");
    check_tensor(
        *sampled, at::kLong, {logits.size(0), 1}, "greedy tokens", device);
    check_tensor(
        *num_sampled, at::kInt, {logits.size(0)}, "greedy counts", device);
    check_tensor(
        *num_rejected, at::kInt, {logits.size(0)}, "greedy rejected", device);
    check_disjoint(
        {values,
         indices,
         partial,
         control,
         *sampled,
         *num_sampled,
         *num_rejected},
        {logits});
    if (bias)
      check_disjoint(
          {values,
           indices,
           partial,
           control,
           *sampled,
           *num_sampled,
           *num_rejected},
          {*bias});
  } else {
    check_disjoint({values, indices, partial, control}, {logits});
  }
  sampling::Args args{};
  args.lg = reinterpret_cast<const uint16_t*>(logits.data_ptr());
  args.bias =
      bias ? reinterpret_cast<const uint16_t*>(bias->data_ptr()) : nullptr;
  args.vs = int(logits.size(1));
  args.ld = int(logits.stride(0));
  args.k = int(k);
  args.rank = int(rank);
  args.lo = 128;
  args.mine = region->words(candidate_mailbox_bytes(), false);
  args.peer = region->words(candidate_mailbox_bytes(), true);
  args.tag = reinterpret_cast<uint32_t*>(control.data_ptr<int>());
  args.health = region->health_pointer()
                    ? reinterpret_cast<int*>(region->health_pointer())
                    : control.data_ptr<int>() + Failed;
  args.part = reinterpret_cast<uint64_t*>(partial.data_ptr<int64_t>());
  args.ov = values.data_ptr<float>();
  args.oi = indices.data_ptr<int32_t>();
  if (sampled) {
    args.sampled = sampled->data_ptr<int64_t>();
    args.num_sampled = num_sampled->data_ptr<int32_t>();
    args.num_rejected = num_rejected->data_ptr<int32_t>();
  }
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  if (sampled) {
    if (bias)
      sampling::shard_topk<true, sycl::half, true, true>(
          queue, args, logits.size(0));
    else if (logits.scalar_type() == at::kHalf)
      sampling::shard_topk<true, sycl::half, true>(queue, args, logits.size(0));
    else
      sampling::shard_topk<true, scalar_t, true>(queue, args, logits.size(0));
    return;
  }
  if (logits.scalar_type() == at::kHalf) {
    if (k == 1)
      sampling::shard_topk<true, sycl::half>(queue, args, logits.size(0));
    else
      sampling::shard_topk<false, sycl::half>(queue, args, logits.size(0));
  } else {
    if (k == 1)
      sampling::shard_topk<true, scalar_t>(queue, args, logits.size(0));
    else
      sampling::shard_topk<false, scalar_t>(queue, args, logits.size(0));
  }
}
void candidate_topk(
    Tensor& values,
    Tensor& indices,
    const Tensor& logits,
    int64_t k,
    int64_t rank,
    Tensor& partial,
    Tensor& control,
    const c10::intrusive_ptr<Region>& region) {
  candidate_topk_impl(
      values, indices, logits, k, rank, partial, control, region);
}

void candidate_greedy(
    Tensor& values,
    Tensor& indices,
    const Tensor& logits,
    int64_t rank,
    Tensor& partial,
    Tensor& control,
    const c10::intrusive_ptr<Region>& region,
    Tensor& sampled,
    Tensor& num_sampled,
    Tensor& num_rejected) {
  candidate_topk_impl(
      values,
      indices,
      logits,
      1,
      rank,
      partial,
      control,
      region,
      &sampled,
      &num_sampled,
      &num_rejected);
}

void candidate_greedy_bias(
    Tensor& values,
    Tensor& indices,
    const Tensor& logits,
    int64_t rank,
    Tensor& partial,
    Tensor& control,
    const c10::intrusive_ptr<Region>& region,
    Tensor& sampled,
    Tensor& num_sampled,
    Tensor& num_rejected,
    const Tensor& bias) {
  candidate_topk_impl(
      values,
      indices,
      logits,
      1,
      rank,
      partial,
      control,
      region,
      &sampled,
      &num_sampled,
      &num_rejected,
      &bias);
}
// Check the exact decoder mailbox protocol independently of layer math.
// The optional tag shift supports a bounded peer-fault sensitivity check.
void mailbox_exchange(
    const Tensor& input,
    Tensor output,
    Tensor control,
    c10::intrusive_ptr<Region> region,
    int64_t send_tag_delta) {
  TORCH_CHECK(supported(), "MiniCPM mailbox requires qualified Xe2");
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= MaxRows,
      "MiniCPM mailbox requires 1..8 rows");
  const int device = c10::xpu::current_device();
  check_tensor(input, at::kFloat, {input.size(0), H}, "mailbox input", device);
  check_tensor(output, at::kFloat, input.sizes(), "mailbox output", device);
  check_tensor(control, at::kInt, {ControlWords}, "mailbox control", device);
  check_disjoint({output, control}, {input});
  region->check_stream();
  Step step{};
  step.control = reinterpret_cast<uint32_t*>(control.data_ptr<int>());
  step.health = region->health_pointer();
  step.mailbox = region->pointer(0, false);
  step.peer = region->pointer(0, true);
  const auto* source = input.data_ptr<float>();
  auto* target = output.data_ptr<float>();
  const int count = input.numel();
  const uint32_t delta = send_tag_delta;
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  queue.submit([&](sycl::handler& handler) {
    handler.parallel_for(
        sycl::nd_range<1>(Groups * Local, Local),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const uint32_t call = Flag(step.control[Epoch]).load() + 1;
          const int n = item.get_global_linear_id();
          if (!grid_barrier(item, step.control, 14, call, step.health)) return;
          if (n < count) {
            const size_t offset = size_t(call & 1u) * MaxRows * H + n;
            Mail(step.peer[offset]).store(encode_mail(source[n], call + delta));
            target[n] = source[n] + receive(
                                        step.mailbox + offset,
                                        call,
                                        step.control,
                                        offset,
                                        step.health);
          }
          if (!grid_barrier(item, step.control, 15, call, step.health)) return;
          if (n == 0) Flag(step.control[Epoch]).store(call);
        });
  });
}
}  // namespace vllm::minicpm5::mono

TORCH_LIBRARY(minicpm5_mono, m) {
  using namespace vllm::minicpm5::mono;
  m.class_<Region>("Region")
      .def(torch::init<int64_t>())
      .def("export_info", &Region::export_info)
      .def("open", &Region::open)
      .def("rebind", &Region::rebind)
      .def("health", &Region::health)
      .def("set_health", &Region::set_health);
  m.def("abi_version() -> int", []() -> int64_t { return 3; });
  m.def("max_context_len() -> int", []() -> int64_t {
    return vllm::minicpm5::MaxContext;
  });
  m.def("rotary_cache_version() -> int", []() -> int64_t { return 1; });
  m.def("padding_mask_version() -> int", []() -> int64_t { return 1; });
  m.def("padding_mask_rows() -> int[]", []() -> std::vector<int64_t> {
    return {4, 8};
  });
  m.def("padding_mask_supported_tps() -> int[]", []() -> std::vector<int64_t> {
    return {1, 2};
  });
  m.def("supported_rows() -> int[]", []() -> std::vector<int64_t> {
    return {1, 2, 3, 4, 5, 6, 7, 8};
  });
  m.def("stage_timing_enabled() -> bool", []() -> bool { return false; });
  m.def("runtime_variant() -> str", []() -> std::string {
#ifdef MINICPM5_MONO_REFERENCE_RUNTIME_ENABLED
    // SpecializedDecode uses this value as its shared-FailStop contract.
    return "reference-shared-health-parallel-page-validation";
#else
    return "per-control-health-serial-page-validation";
#endif
  });
  m.def("handoff_variant() -> str", []() -> std::string {
    return "bf16-tp2-row1-leader-device-acquire-wg-global-barrier;"
           "moe-kernel-exit;other-cases-original-acquire";
  });
  m.def("projection_variant() -> str", []() -> std::string {
#ifdef MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED
  #ifdef MINICPM5_MONO_DPAS_ROW1_ENABLED
    return "bf16-tp2-row1-balanced-eight-k-slices;"
           "tp2-bfloat16-row4-float16-row8-pr-early-full-qkv-op-preload;tp2-"
           "row1-reference-full-"
           "op-"
           "gemv-local-ar1-prt;other-cases-reference-bf16-slm-k-slices-rows1-2-"
           "4-8";
  #else
    return "reference-bf16-slm-k-slices-rows2-4-8;fp32-gemv-pipeline-row1";
  #endif
#else
  #ifdef MINICPM5_MONO_DPAS_ENABLED
    #ifdef MINICPM5_MONO_DPAS_ROW1_ENABLED
    return "bf16-dpas-rows1-2-4-8";
    #else
    return "bf16-dpas-rows2-4-8;scalar-fp32-row1";
    #endif
  #else
    return "scalar-fp32-rows1-2-4-8";
  #endif
#endif
  });
  m.def("mlp_phase_variant() -> str", []() -> std::string {
#if defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED) && \
    defined(MINICPM5_MONO_DPAS_ROW1_ENABLED)
    return "shared-activation-local-router-slm-no-phase14-moe";
#else
    return "shared-activation-global-phase14";
#endif
  });
  m.def("activation_dtypes() -> str[]", []() -> std::vector<std::string> {
    return {"bfloat16", "float16"};
  });
  m.def("attention_overlap_enabled() -> bool", []() -> bool { return false; });
  m.def("supported() -> bool", &supported);
  m.def("supports_device(int device_index) -> bool", &supports_device);
  m.def("attention_variant() -> str", []() -> std::string {
#ifdef MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED
    return "tp2-row2-row4-pr-parallel-merge-8chains;reference-sg-context-"
           "chunks-tickets-slm-q-block2d-qk-pv-4tile-"
           "single-body-row4-8-packed-probability-row1-merge8parts-8chains-"
           "default-head8;full64-unchecked-kv;score-mask-rounded;"
           "fp16-tp2-row1-merge-load8";
#else
  #ifdef MINICPM5_MONO_DPAS_ENABLED
    return "bf16-dpas-qk8-gqa-reused-values-fp32-softmax-pv";
  #else
    return "scalar-qk-per-head-fp32-softmax-pv";
  #endif
#endif
  });
  m.def("attention_partial_groups(int tp) -> int", [](int64_t tp) -> int64_t {
    TORCH_CHECK(tp == 1 || tp == 2, "MiniCPM mono TP1/TP2 only");
    return attention_partial_groups(tp);
  });
  m.def("expert_union_variant() -> str", []() -> std::string {
#ifdef MINICPM5_MONO_REFERENCE_MOE_ENABLED
  #ifdef MINICPM5_MONO_DPAS_ROW1_ENABLED
    return "reference-union-native-kn32-kn64-row1-2-gateup-kn64-kparts8-slm-"
           "down-topk-order-rows1-2-4-8";
  #else
    return "reference-union-native-kn32-kn64-slm-down-topk-order;axpy-row1";
  #endif
#else
    return "repeated-global-expert-scans-for-matrix-path";
#endif
  });
  m.def("fp8_decode_variant() -> str", []() -> std::string {
#ifdef MINICPM5_MONO_BF16_FP8_HALF_ENABLED
    return "exact-native-half-conversion-bf16;scaled-half-fp16";
#elif defined(MINICPM5_MONO_FAST_FP8_ENABLED)
    return "packed-exact-bf16-fixed-shift-subnormals";
#else
    return "exact-bf16-clz-subnormals";
#endif
  });
  m.def("control_words() -> int", []() -> int64_t { return ControlWords; });
  m.def("router_logits_dtype() -> str", []() -> std::string {
    return "float32";
  });
  m.def("router_variant() -> str", []() -> std::string {
#if defined(MINICPM5_MONO_REFERENCE_MOE_ENABLED) && \
    defined(MINICPM5_MONO_DPAS_ENABLED)
  #ifdef MINICPM5_MONO_DPAS_ROW1_ENABLED
    return "tp2-row1-reference-prt-canonical-top16-unroll32;other-cases-"
           "activation-dpas-k64-wg-local-top16-no-phase7-rows1-2-4-8";
  #else
    return "activation-dpas-k64-wg-local-top16-no-phase7-rows2-4-8;scalar-row1";
  #endif
#else
    return "scalar-bf16-weight-fp32-accumulate";
#endif
  });
  m.def("health_offset() -> int", []() -> int64_t { return Failed; });
  m.def("mailbox_bytes() -> int", []() -> int64_t {
    return MailboxWords * sizeof(uint64_t);
  });
  m.def("candidate_mailbox_bytes() -> int", &candidate_mailbox_bytes);
  m.def("candidate_greedy_version() -> int", []() -> int64_t { return 1; });
  m.def("shared_verify_pages_version() -> int", []() -> int64_t { return 1; });
  m.def(
      "shared_verify_pages_supported_tps() -> int[]",
      []() -> std::vector<int64_t> { return {1, 2}; });
  m.def("qkv_layout_version() -> int", []() -> int64_t { return 1; });
  m.def("fp16_qkv_layout_version() -> int", []() -> int64_t { return 1; });
  m.def("fp16_qkv_supported_tps() -> int[]", []() -> std::vector<int64_t> {
    return {1, 2};
  });
  m.def(
      "candidate_greedy(Tensor(a!) values, Tensor(b!) indices, Tensor logits, "
      "int rank, Tensor(c!) partial, Tensor(d!) control, "
      "__torch__.torch.classes.minicpm5_mono.Region region, "
      "Tensor(e!) sampled, Tensor(f!) num_sampled, Tensor(g!) num_rejected) -> "
      "()",
      &candidate_greedy);
  m.def(
      "candidate_greedy_bias(Tensor(a!) values, Tensor(b!) indices, Tensor "
      "logits, "
      "int rank, Tensor(c!) partial, Tensor(d!) control, "
      "__torch__.torch.classes.minicpm5_mono.Region region, "
      "Tensor(e!) sampled, Tensor(f!) num_sampled, Tensor(g!) num_rejected, "
      "Tensor bias) -> ()",
      &candidate_greedy_bias);
  m.def(
      "mailbox_exchange(Tensor input, Tensor(a!) output, Tensor(b!) control, "
      "__torch__.torch.classes.minicpm5_mono.Region region, int "
      "send_tag_delta=0) -> ()",
      &mailbox_exchange);
  m.def(
      "candidate_topk(Tensor(a!) values, Tensor(b!) indices, Tensor logits, "
      "int k, int rank, Tensor(c!) partial, Tensor(d!) control, "
      "__torch__.torch.classes.minicpm5_mono.Region region) -> ()",
      &candidate_topk);
#if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) && \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED)
  m.def("draft_weight_layout_version() -> int", []() -> int64_t { return 1; });
  m.def("draft_tensor_parallel_sizes() -> int[]", []() -> std::vector<int64_t> {
    return {1, 2};
  });
  m.def(
      "draft_decode(Tensor hidden, Tensor? residual, Tensor positions, Tensor "
      "blocks, "
      "Tensor lengths, Tensor slots, Tensor(a!) output, Tensor(b!) res_out, "
      "Tensor(c!)[] workspace, Tensor(d!) control, Tensor[] weights, "
      "Tensor(e!) cache, "
      "__torch__.torch.classes.minicpm5_mono.Region region, int mailbox_slot, "
      "Tensor rope_cache) -> ()",
      &draft_decode);
#endif
  m.def(
      "validate(Tensor[] weights, Tensor cache, Tensor[] workspace, int tp, "
      "bool dense) -> ()",
      &validate);
  m.def(
      "decode(Tensor hidden, Tensor? residual, Tensor positions, Tensor "
      "blocks, "
      "Tensor lengths, Tensor slots, Tensor(a!) output, Tensor(b!) res_out, "
      "Tensor(c!)[] workspace, Tensor(d!) control, Tensor[] weights, "
      "Tensor(e!) cache, "
      "__torch__.torch.classes.minicpm5_mono.Region region, int mailbox_slot, "
      "int tp, bool dense, bool ignore_padding=False, Tensor? rope_cache=None) "
      "-> ()",
      &decode);
#if defined(MINICPM5_MONO_REFERENCE_ATTENTION_ENABLED) &&  \
    defined(MINICPM5_MONO_REFERENCE_PROJECTION_ENABLED) && \
    defined(MINICPM5_MONO_REFERENCE_MOE_ENABLED) &&        \
    defined(MINICPM5_MONO_DPAS_ENABLED)
  m.def("tail_only_abi_version() -> int", []() -> int64_t { return 1; });
  m.def(
      "decode_tail(Tensor normed, Tensor slots, Tensor(a!) output, "
      "Tensor(b!)[] workspace, Tensor(c!) control, Tensor[] weights, "
      "__torch__.torch.classes.minicpm5_mono.Region region, int mailbox_slot) "
      "-> ()",
      &decode_tail<MaxRows>);
  m.def("tail_rows4_abi_version() -> int", []() -> int64_t { return 1; });
  m.def(
      "decode_tail_rows4(Tensor normed, Tensor slots, Tensor(a!) output, "
      "Tensor(b!)[] workspace, Tensor(c!) control, Tensor[] weights, "
      "__torch__.torch.classes.minicpm5_mono.Region region, int mailbox_slot) "
      "-> ()",
      &decode_tail<4>);

  m.def("pre_o_abi_version() -> int", []() -> int64_t { return 1; });
  m.def(
      "decode_pre_o(Tensor attention, Tensor residual, Tensor slots, "
      "Tensor(a!) output, Tensor(b!) res_out, Tensor(c!)[] workspace, "
      "Tensor(d!) control, Tensor[] weights, "
      "__torch__.torch.classes.minicpm5_mono.Region region, int mailbox_slot) "
      "-> ()",
      &decode_attention_tail<false>);
  m.def("post_o_abi_version() -> int", []() -> int64_t { return 1; });
  m.def(
      "decode_post_o(Tensor projection, Tensor residual, Tensor slots, "
      "Tensor(a!) output, Tensor(b!) res_out, Tensor(c!)[] workspace, "
      "Tensor(d!) control, Tensor[] weights, "
      "__torch__.torch.classes.minicpm5_mono.Region region, int mailbox_slot) "
      "-> ()",
      &decode_attention_tail<true>);
#endif
}
