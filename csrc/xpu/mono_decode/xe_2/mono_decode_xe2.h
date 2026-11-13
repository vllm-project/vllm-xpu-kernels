// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

#include <cstdint>
#include <optional>
#include <vector>

// The Xe2 (Battlemage) implementation of the mono decode ops
// (mono_decode_kernels_xe_2). See ../mono_decode_interface.h for the contracts;
// the interface checks the device architecture before calling in.
namespace vllm::mono_decode::xe2 {

int64_t mono_max_context();
torch::Tensor mono_health(int64_t device);
void mono_fault_inject(int64_t layer, int64_t delta);
// The device's health words (mono_health), allocated on first use.
int* health_words(int device);

int64_t mono_ipc_create(int64_t bytes);
std::vector<int64_t> mono_ipc_export(int64_t handle);
int64_t
mono_ipc_open(int64_t handle, std::vector<int64_t> peer_info, int64_t local_fd);
int64_t mono_ipc_ptr(int64_t handle, bool peer);
int64_t mono_mbox_bytes();

void mono_register_layer(
    int64_t layer,
    bool attn,
    int64_t mbox,
    int64_t peer,
    const std::vector<torch::Tensor>& tensors);
void mono_decode_layer(
    int64_t layer,
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    torch::Tensor& hid_out,
    torch::Tensor& res_out,
    const torch::Tensor& positions,
    const torch::Tensor& m0,
    const std::optional<torch::Tensor>& m1,
    const std::optional<torch::Tensor>& m2);
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
    const std::optional<torch::Tensor>& m2);

void mono_register_mtp(
    int64_t layer,
    int64_t mbox,
    int64_t peer,
    const std::vector<torch::Tensor>& tensors,
    int64_t rank);
bool mono_mtp_prologue(
    int64_t layer, const torch::Tensor& x, torch::Tensor& out);

int64_t mono_lmhead_mbox_bytes();
int64_t mono_shard_topk_max_k();
void mono_shard_topk(
    torch::Tensor& vals,
    torch::Tensor& idx,
    const torch::Tensor& logits,
    int64_t k,
    int64_t rank,
    int64_t mbox,
    int64_t peer);

}  // namespace vllm::mono_decode::xe2
