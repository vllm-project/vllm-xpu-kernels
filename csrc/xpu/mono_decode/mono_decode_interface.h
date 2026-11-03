// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <torch/all.h>

#include <cstdint>
#include <optional>
#include <vector>

// Decode-only "mono" decoder layers for Qwen3.5 / Qwen3.6 MoE (35B-A3B shapes)
// at TP2 on Xe2: one resident kernel per decoder layer, the TP all-reduces done
// inside the kernel through a P2P mailbox in the peer card's memory.
// Implemented in xe_2/ (mono_decode_kernels_xe_2).
namespace vllm::mono_decode {

// Whether the device has the kernels' architecture (Xe2: Battlemage). The other
// ops raise an error on any other device.
bool mono_decode_supported(int64_t device);

// Longest context (cached tokens per request) the attention kernels are
// validated for. The kernels take sequence lengths from device memory and do
// not check them: callers must not exceed this (vLLM refuses a larger
// max_model_len). The batched kernels' dynamic schedule packs per-row ticket
// numbers into 16 bits.
int64_t mono_max_context();

// Mailbox memory (one region per rank, shared with the peer through Level Zero
// IPC).
int64_t mono_ipc_create(int64_t bytes);
// [pid, fd, cookie lo, cookie hi, 16 raw ze_ipc_mem_handle_t words]
std::vector<int64_t> mono_ipc_export(int64_t handle);
// Imports the peer's region; returns the import mode that worked.
// local_fd: the peer's dma-buf fd received into this process (-1: import
// through the driver / pidfd).
int64_t
mono_ipc_open(int64_t handle, std::vector<int64_t> peer_info, int64_t local_fd);
// Device address of the own (peer=false) or the imported peer (peer=true)
// region.
int64_t mono_ipc_ptr(int64_t handle, bool peer);
// Bytes of mailbox per layer.
int64_t mono_mbox_bytes();

// Registers layer `layer` (weights, caches; vLLM tensors, used in place).
// `attn`: full_attention layer, else linear_attention. mbox / peer: addresses
// of this layer's mailbox in the own / peer region. tensors (common, then per
// kind):
//   input_norm_w, post_norm_w, router_w [256,2048] f16, shared_gate_w [1,2048]
//   f16, out_w (o_proj / out_proj, logical [2048,2048] = t() of [N][K]),
//   out_scale, w13 [E,2048,512] e4m3 + scale [E], w2 [E,256,2048] + scale [E],
//   shared w13 (logical [2048,512]) + scale, shared w2 (logical [256,2048]) +
//   scale, attn: qkv_w (logical [2048,4608]) + scale, q_norm_w, k_norm_w,
//   kv_cache [blocks,1,bs,512] f16 gdn : qkvz_w (logical [2048,6144]) + scale,
//   ba_w (logical [2048,32]) + scale, conv_w [4096,1,4], A_log,
//         dt_bias, norm_w, conv_state cache [slots,3,4096] f16, ssm_state cache
//         [slots,16,128,128] f32
void mono_register_layer(
    int64_t layer,
    bool attn,
    int64_t mbox,
    int64_t peer,
    const std::vector<torch::Tensor>& tensors);

// One decode step (one token) of layer `layer` on the current XPU stream.
// hid_out = MoE output (all-reduced), res_out = residual after the mixer. m0:
// attn block_table [B, n] int32 | gdn state indices [B] int32; m1: attn
// seq_lens [B] int32; m2: attn slot_mapping [T] int64.
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

// Top-k candidates of the vocabulary-parallel logits (TP2, B <= 8 rows,
// k <= mono_shard_topk_max_k()) instead of all-gathering them: vals / idx
// [B, w >= 2k] f32 / int32 = rank 0's k largest logits of its shard, then
// rank 1's (each part in ascending token id order, idx = vocabulary token
// ids), then -inf / 0 padding; the same on both ranks. logits: this rank's [B,
// V/2] fp16 shard; mbox / peer: own / peer mailbox of mono_lmhead_mbox_bytes()
// bytes.
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
}  // namespace vllm::mono_decode
