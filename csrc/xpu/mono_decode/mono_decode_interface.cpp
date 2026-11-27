// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <atomic>

#include "utils.h"
#include "mono_decode_interface.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/mono_decode_xe2.h"
#endif

// The kernels are written for Xe2 (Battlemage: 2D block loads, DPAS shapes,
// 256-GRF work-groups, the sub-group clock); every op that touches a device
// checks its architecture first. The layer kernels also keep 32 work-groups
// resident, one per Xe core, that wait on each other: a Battlemage part with
// fewer Xe cores (BMG-G21: 20) is not supported. The result is cached per
// device: the candidate exchange is called once a decode step, outside any
// graph.

namespace vllm::mono_decode {

#ifdef VLLM_XPU_ENABLE_XE2

namespace {

// work-groups of a layer kernel, all resident (xe_2/mono_layers.hpp: NWG)
constexpr int kResidentWorkGroups = 32;

bool supported_device(at::DeviceIndex d) {
  if (!vllm::xpu::is_bmg(d)) return false;
  const auto& dev = c10::xpu::get_raw_device(d);
  const int xe_cores =
      dev.get_info<sycl::ext::intel::info::device::gpu_slices>() *
      dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  return xe_cores >= kResidentWorkGroups;
}

bool xe2_device(int64_t device) {
  static std::array<std::atomic<int8_t>, 64> known{};  // 0 unknown, 1 / -1
  const at::DeviceIndex d =
      device < 0 ? c10::xpu::current_device() : (at::DeviceIndex)device;
  if (d < 0 || d >= (at::DeviceIndex)known.size()) return supported_device(d);
  int8_t v = known[d].load(std::memory_order_relaxed);
  if (v == 0) {
    v = supported_device(d) ? 1 : -1;
    known[d].store(v, std::memory_order_relaxed);
  }
  return v == 1;
}

void check_device(const char* op, int64_t device = -1) {
  TORCH_CHECK(
      xe2_device(device),
      op,
      ": unsupported device arch (the mono decode kernels need an Xe2 "
      "(Battlemage) GPU with at least 32 Xe cores; check "
      "mono_decode_supported() first).");
}

void check_xpu(const char* op, const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_xpu(), op, ": ", name, " must be an XPU tensor");
}

}  // namespace

bool mono_decode_supported(int64_t device) { return xe2_device(device); }

int64_t mono_max_context() { return xe2::mono_max_context(); }

torch::Tensor mono_health(int64_t device) {
  check_device("mono_health", device);
  return xe2::mono_health(device);
}

void mono_fault_inject(int64_t layer, int64_t delta) {
  check_device("mono_fault_inject");
  xe2::mono_fault_inject(layer, delta);
}

int64_t mono_ipc_create(int64_t bytes) {
  check_device("mono_ipc_create");
  TORCH_CHECK(bytes > 0, "mono_ipc_create: bytes must be positive");
  return xe2::mono_ipc_create(bytes);
}

std::vector<int64_t> mono_ipc_export(int64_t handle) {
  return xe2::mono_ipc_export(handle);
}

int64_t mono_ipc_open(
    int64_t handle, std::vector<int64_t> peer_info, int64_t local_fd) {
  check_device("mono_ipc_open");
  return xe2::mono_ipc_open(handle, std::move(peer_info), local_fd);
}

int64_t mono_ipc_ptr(int64_t handle, bool peer) {
  return xe2::mono_ipc_ptr(handle, peer);
}

int64_t mono_mbox_bytes() { return xe2::mono_mbox_bytes(); }

void mono_register_layer(
    int64_t layer,
    bool attn,
    int64_t mbox,
    int64_t peer,
    const std::vector<torch::Tensor>& tensors) {
  check_device("mono_register_layer");
  for (const auto& t : tensors)
    check_xpu("mono_register_layer", t, "tensors");
  xe2::mono_register_layer(layer, attn, mbox, peer, tensors);
}

void mono_decode_layer(
    int64_t layer,
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    torch::Tensor& hid_out,
    torch::Tensor& res_out,
    const torch::Tensor& positions,
    const torch::Tensor& m0,
    const std::optional<torch::Tensor>& m1,
    const std::optional<torch::Tensor>& m2) {
  check_xpu("mono_decode_layer", hidden, "hidden");
  check_device("mono_decode_layer", hidden.get_device());
  xe2::mono_decode_layer(
      layer, hidden, residual, hid_out, res_out, positions, m0, m1, m2);
}

bool mono_decode_layer_spec(
    int64_t layer,
    int64_t tokens,
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    torch::Tensor& hid_out,
    torch::Tensor& res_out,
    const torch::Tensor& positions,
    const torch::Tensor& m0,
    const torch::Tensor& m1,
    const std::optional<torch::Tensor>& m2) {
  check_xpu("mono_decode_layer_spec", hidden, "hidden");
  check_device("mono_decode_layer_spec", hidden.get_device());
  TORCH_CHECK(tokens >= 1, "mono_decode_layer_spec: tokens must be positive");
  return xe2::mono_decode_layer_spec(
      layer, tokens, hidden, residual, hid_out, res_out, positions, m0, m1, m2);
}

void mono_register_mtp(
    int64_t layer,
    int64_t mbox,
    int64_t peer,
    const std::vector<torch::Tensor>& tensors,
    int64_t rank) {
  check_device("mono_register_mtp");
  for (const auto& t : tensors)
    check_xpu("mono_register_mtp", t, "tensors");
  xe2::mono_register_mtp(layer, mbox, peer, tensors, rank);
}

bool mono_mtp_prologue(
    int64_t layer, const torch::Tensor& x, torch::Tensor& out) {
  check_xpu("mono_mtp_prologue", x, "x");
  check_device("mono_mtp_prologue", x.get_device());
  return xe2::mono_mtp_prologue(layer, x, out);
}

int64_t mono_lmhead_mbox_bytes() { return xe2::mono_lmhead_mbox_bytes(); }

int64_t mono_shard_topk_max_k() { return xe2::mono_shard_topk_max_k(); }

void mono_shard_topk(
    torch::Tensor& vals,
    torch::Tensor& idx,
    const torch::Tensor& logits,
    int64_t k,
    int64_t rank,
    int64_t mbox,
    int64_t peer) {
  check_xpu("mono_shard_topk", logits, "logits");
  check_device("mono_shard_topk", logits.get_device());
  xe2::mono_shard_topk(vals, idx, logits, k, rank, mbox, peer);
}

#endif  // VLLM_XPU_ENABLE_XE2

}  // namespace vllm::mono_decode
