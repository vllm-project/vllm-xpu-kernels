// SPDX-License-Identifier: Apache-2.0
//
// Host side of the decode mono layers: IPC mailbox regions, the per-layer
// registry (vLLM weights / caches used in place, one scratch per layer), and
// the per-step launch on the current torch XPU queue.
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "mono_mtp.hpp"
#include "../mono_decode_interface.h"
#include "../mono_arch.h"

namespace mono_decode {
void launch_gdn(sycl::queue& q, const Ptrs& P, const Scratch& S);
void launch_attn(sycl::queue& q, const Ptrs& P, const Scratch& S);
void launch_gdn_b(
    sycl::queue& q,
    int nb,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB);
void launch_attn_b(
    sycl::queue& q,
    int nb,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB);
bool launch_gdn_s(
    sycl::queue& q,
    int nb,
    int t,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB);
bool launch_attn_s(
    sycl::queue& q,
    int nb,
    int t,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB);
bool launch_mtp(
    sycl::queue& q, int nb, const Ptrs& P, const Scratch& S, int rank);
}  // namespace mono_decode

namespace vllm::mono_decode {

using ::mono_decode::Ptrs;
using ::mono_decode::Scratch;
using ::mono_decode::ScratchB;

#define ZE_CHECK(call)                 \
  do {                                 \
    ze_result_t _r = (call);           \
    TORCH_CHECK(                       \
        _r == ZE_RESULT_SUCCESS,       \
        #call " failed: ze_result 0x", \
        std::hex,                      \
        static_cast<uint32_t>(_r));    \
  } while (0)

// -----------------------------------------------------------------------------
// IPC regions
namespace {
constexpr int64_t kCookieOff =
    0;  // first 64 B of a region: cookie (mailboxes start at 64 B)
constexpr int64_t kHdr = 64;

struct Region {
  ze_context_handle_t ctx = nullptr;
  ze_device_handle_t dev = nullptr;
  char* own = nullptr;
  char* peer = nullptr;
  int64_t bytes = 0;
  ze_ipc_mem_handle_t ipc{};
  bool exported = false;
  uint64_t cookie = 0;
  int imported_fd = -1;
};
std::mutex g_mu;
std::vector<std::unique_ptr<Region>> g_regions;

Region& region(int64_t h) {
  std::lock_guard<std::mutex> lk(g_mu);
  TORCH_CHECK(
      h >= 0 && h < (int64_t)g_regions.size() && g_regions[h],
      "mono: invalid IPC handle ",
      h);
  return *g_regions[h];
}
}  // namespace

bool mono_decode_supported(int64_t device) { return xe2_device(device); }

int64_t mono_max_context() { return 512 * 1024; }

int64_t mono_ipc_create(int64_t bytes) {
  require_xe2("mono_ipc_create");
  auto r = std::make_unique<Region>();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  r->ctx =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
  r->dev =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
  r->bytes = kHdr + bytes;
  ze_device_mem_alloc_desc_t desc{
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  void* p = nullptr;
  ZE_CHECK(zeMemAllocDevice(r->ctx, &desc, r->bytes, 4096, r->dev, &p));
  r->own = static_cast<char*>(p);
  q.memset(r->own, 0, r->bytes).wait();
  r->cookie = ((uint64_t)getpid() << 32) ^ (uint64_t)(uintptr_t)r->own ^
              0x9e3779b97f4a7c15ull;
  q.memcpy(r->own + kCookieOff, &r->cookie, sizeof(uint64_t)).wait();
  std::lock_guard<std::mutex> lk(g_mu);
  g_regions.push_back(std::move(r));
  return (int64_t)g_regions.size() - 1;
}

std::vector<int64_t> mono_ipc_export(int64_t h) {
  Region& r = region(h);
  if (!r.exported) {
    ZE_CHECK(zeMemGetIpcHandle(r.ctx, r.own, &r.ipc));
    r.exported = true;
  }
  uint64_t fd = 0;
  if (zeMemGetFileDescriptorFromIpcHandleExp(r.ctx, r.ipc, &fd) !=
      ZE_RESULT_SUCCESS) {
    int fd32 = 0;  // the Intel handle starts with the dma-buf fd
    std::memcpy(&fd32, r.ipc.data, sizeof(fd32));
    fd = (uint64_t)fd32;
  }
  std::vector<int64_t> out{
      (int64_t)getpid(),
      (int64_t)fd,
      (int64_t)(r.cookie & 0xffffffffu),
      (int64_t)(r.cookie >> 32)};
  uint32_t w[ZE_MAX_IPC_HANDLE_SIZE / 4];
  std::memcpy(w, r.ipc.data, sizeof(w));
  for (uint32_t v : w)
    out.push_back(v);
  return out;
}

// local_fd >= 0: the peer's dma-buf fd, already passed into this process (unix
// socket SCM_RIGHTS); the handle is opened with its fd field patched. Otherwise
// tries (0) the raw peer handle (the driver imports the fd itself), then (1)
// pidfd_getfd (needs ptrace permission). A mapping is accepted only if the
// peer's cookie reads back.
int64_t mono_ipc_open(int64_t h, std::vector<int64_t> peer, int64_t local_fd) {
  Region& r = region(h);
  TORCH_CHECK(r.peer == nullptr, "mono: peer already opened");
  TORCH_CHECK(
      peer.size() == 4 + ZE_MAX_IPC_HANDLE_SIZE / 4,
      "mono: bad peer handle info");
  const uint64_t peer_cookie =
      (uint64_t)(uint32_t)peer[2] | ((uint64_t)(uint32_t)peer[3] << 32);
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  auto verify = [&](void* p) {
    uint64_t c = 0;
    q.memcpy(&c, static_cast<char*>(p) + kCookieOff, sizeof(c)).wait();
    if (c == peer_cookie) return true;
    zeMemCloseIpcHandle(r.ctx, p);
    return false;
  };
  ze_ipc_mem_handle_t raw{};
  uint32_t w[ZE_MAX_IPC_HANDLE_SIZE / 4];
  for (size_t i = 0; i < ZE_MAX_IPC_HANDLE_SIZE / 4; ++i)
    w[i] = (uint32_t)peer[4 + i];
  std::memcpy(raw.data, w, sizeof(w));
  void* p = nullptr;
  int64_t mode = -1;
  // the handle with its fd (and the exporter pid) patched to a local fd
  auto open_with_fd = [&](int fd) {
    ze_ipc_mem_handle_t ph = raw;
    int fd32 = fd;
    std::memcpy(ph.data, &fd32, sizeof(fd32));
    uint32_t pw[ZE_MAX_IPC_HANDLE_SIZE / 4];
    std::memcpy(pw, ph.data, sizeof(pw));
    for (int i = 1; i < ZE_MAX_IPC_HANDLE_SIZE / 4; ++i)
      if (pw[i] == (uint32_t)peer[0]) pw[i] = (uint32_t)getpid();
    std::memcpy(ph.data + 4, pw + 1, sizeof(pw) - 4);
    const ze_result_t rr = zeMemOpenIpcHandle(r.ctx, r.dev, ph, 0, &p);
    return rr == ZE_RESULT_SUCCESS && verify(p);
  };
  if (local_fd >= 0) {  // fd received over a unix socket (SCM_RIGHTS): no
                        // ptrace permission needed
    TORCH_CHECK(
        open_with_fd((int)local_fd),
        "mono: could not open the peer IPC handle with the received fd");
    r.imported_fd = (int)local_fd;
    r.peer = static_cast<char*>(p);
    return 2;
  }
  ze_result_t r0 = zeMemOpenIpcHandle(r.ctx, r.dev, raw, 0, &p);
  if (r0 == ZE_RESULT_SUCCESS && verify(p)) mode = 0;
  if (mode < 0) {
    int pidfd = (int)syscall(SYS_pidfd_open, (pid_t)peer[0], 0);
    TORCH_CHECK(
        pidfd >= 0, "mono: pidfd_open(", peer[0], ") failed, errno ", errno);
    int fd = (int)syscall(SYS_pidfd_getfd, pidfd, (int)peer[1], 0);
    const int err = errno;
    close(pidfd);
    TORCH_CHECK(fd >= 0, "mono: pidfd_getfd failed, errno ", err);
    if (open_with_fd(fd)) {
      mode = 1;
      r.imported_fd = fd;
    } else
      close(fd);
    TORCH_CHECK(
        mode >= 0,
        "mono: could not open the peer IPC handle (raw 0x",
        std::hex,
        (uint32_t)r0,
        ")");
  }
  r.peer = static_cast<char*>(p);
  return mode;
}

int64_t mono_ipc_ptr(int64_t h, bool peer) {
  Region& r = region(h);
  char* p = peer ? r.peer : r.own;
  TORCH_CHECK(p != nullptr, "mono: region not opened");
  return (int64_t)(uintptr_t)(p + kHdr);
}

int64_t mono_mbox_bytes() {
  return (int64_t)(::mono_decode::MBOX_WORDS_B * sizeof(uint64_t));
}

// -----------------------------------------------------------------------------
// layers
namespace {
struct Layer {
  bool attn = false;
  bool mtp = false;  // MTP drafter prologue (mono_register_mtp)
  int rank = 0;
  Ptrs P{};
  Scratch S{};
  ScratchB SB{};
  void* scratch = nullptr;
  void* scratch_b = nullptr;
  int device = -1;
};
std::unordered_map<int64_t, Layer> g_layers;
// batched attention chunk partials, one buffer per device shared by all
// attention layers (they run one at a time on the device's queue)
std::unordered_map<int, void*> g_attn_part;

const void* ptr_of(const torch::Tensor& t) { return t.data_ptr(); }

// fp8 linear weight: vLLM keeps W^T as a [K, N] view of a row-major [N, K]
// tensor
const uint8_t*
w_nk(const torch::Tensor& w, int64_t n, int64_t k, const char* name) {
  TORCH_CHECK(
      w.scalar_type() == at::kFloat8_e4m3fn,
      "mono: ",
      name,
      " must be float8_e4m3fn");
  TORCH_CHECK(
      w.dim() == 2 && w.size(0) == k && w.size(1) == n && w.stride(0) == 1 &&
          w.stride(1) == k,
      "mono: ",
      name,
      " must be a [",
      k,
      ", ",
      n,
      "] view of a row-major [",
      n,
      ", ",
      k,
      "] tensor, got ",
      w.sizes(),
      " strides ",
      w.strides());
  return static_cast<const uint8_t*>(w.data_ptr());
}
const float* scale1(const torch::Tensor& s, const char* name) {
  TORCH_CHECK(
      s.scalar_type() == at::kFloat && s.numel() == 1,
      "mono: ",
      name,
      " must be one fp32 (per-tensor)");
  return s.data_ptr<float>();
}
const sycl::half* f16(const torch::Tensor& t, int64_t numel, const char* name) {
  TORCH_CHECK(
      t.scalar_type() == at::kHalf && t.numel() == numel && t.is_contiguous(),
      "mono: ",
      name,
      " must be contiguous fp16 with ",
      numel,
      " elements, got ",
      t.scalar_type(),
      " ",
      t.sizes());
  return reinterpret_cast<const sycl::half*>(t.data_ptr());
}
}  // namespace

void mono_register_layer(
    int64_t layer,
    bool attn,
    int64_t mbox,
    int64_t peer,
    const std::vector<torch::Tensor>& t) {
  require_xe2("mono_register_layer");
  using namespace ::mono_decode;
  TORCH_CHECK(
      t.size() == (attn ? 19u : 24u),
      "mono: wrong number of tensors for layer ",
      layer);
  Layer L;
  L.attn = attn;
  L.device = t[0].get_device();
  Ptrs& P = L.P;
  P.in_norm_w = f16(t[0], HID, "input_norm_w");
  P.post_norm_w = f16(t[1], HID, "post_norm_w");
  P.router_w = f16(t[2], (int64_t)NE * HID, "router_w");
  P.sgate_w = f16(t[3], HID, "shared_gate_w");
  P.out_w = w_nk(t[4], HID, attn ? NQ * HD : VD, "out_w");
  P.out_s = scale1(t[5], "out_scale");
  TORCH_CHECK(
      t[6].scalar_type() == at::kFloat8_e4m3fn && t[6].is_contiguous() &&
          t[6].dim() == 3 && t[6].size(1) == HID && t[6].size(2) == 2 * INTER,
      "mono: w13 must be contiguous [E, 2048, 512] e4m3 (XPU MoE layout), got ",
      t[6].sizes());
  TORCH_CHECK(
      t[8].scalar_type() == at::kFloat8_e4m3fn && t[8].is_contiguous() &&
          t[8].dim() == 3 && t[8].size(1) == INTER && t[8].size(2) == HID &&
          t[8].size(0) == t[6].size(0) && t[6].size(0) == NE,
      "mono: w2 must be contiguous [256, 256, 2048] e4m3, got ",
      t[8].sizes());
  TORCH_CHECK(
      t[7].scalar_type() == at::kFloat && t[7].numel() == NE &&
          t[9].scalar_type() == at::kFloat && t[9].numel() == NE,
      "mono: MoE scales must be per-expert fp32");
  P.w13 = static_cast<const uint8_t*>(t[6].data_ptr());
  P.w13_s = t[7].data_ptr<float>();
  P.w2 = static_cast<const uint8_t*>(t[8].data_ptr());
  P.w2_s = t[9].data_ptr<float>();
  P.sw13 = w_nk(t[10], 2 * INTER, HID, "shared_w13");
  P.sw13_s = scale1(t[11], "shared_w13_scale");
  P.sw2 = w_nk(t[12], HID, INTER, "shared_w2");
  P.sw2_s = scale1(t[13], "shared_w2_scale");
  if (attn) {
    P.qkv_w = w_nk(t[14], QKV_ROWS, HID, "qkv_w");
    P.qkv_s = scale1(t[15], "qkv_scale");
    P.q_norm_w = f16(t[16], HD, "q_norm_w");
    P.k_norm_w = f16(t[17], HD, "k_norm_w");
    const torch::Tensor& kv = t[18];
    TORCH_CHECK(
        kv.scalar_type() == at::kHalf && kv.dim() == 4 && kv.size(1) == 1 &&
            kv.size(3) == 2 * HD && kv.stride(3) == 1 &&
            kv.stride(2) == 2 * HD && kv.size(2) % 64 == 0,
        "mono: kv_cache must be [blocks, 1, bs (multiple of 64), 512] fp16 "
        "rows, got ",
        kv.sizes(),
        " ",
        kv.strides());
    P.kv = reinterpret_cast<sycl::half*>(kv.data_ptr());
    P.blk_stride = kv.stride(0);
    P.bs = (int)kv.size(2);
  } else {
    P.qkvz_w = w_nk(t[14], QKVZ, HID, "qkvz_w");
    P.qkvz_s = scale1(t[15], "qkvz_scale");
    P.ba_w = w_nk(t[16], 2 * NVH, HID, "ba_w");
    P.ba_s = scale1(t[17], "ba_scale");
    P.conv_w = f16(t[18], (int64_t)CONVD * 4, "conv_w");
    TORCH_CHECK(
        t[19].scalar_type() == at::kFloat && t[19].numel() == NVH,
        "mono: A_log must be fp32 [16]");
    P.A_log = t[19].data_ptr<float>();
    P.dt_bias = f16(t[20], NVH, "dt_bias");
    P.gnorm_w = f16(t[21], DV, "norm_w");
    const torch::Tensor& cs = t[22];
    const torch::Tensor& ss = t[23];
    TORCH_CHECK(
        cs.scalar_type() == at::kHalf && cs.dim() == 3 && cs.size(1) >= 3 &&
            cs.size(2) == CONVD && cs.stride(2) == 1 && cs.stride(1) == CONVD,
        "mono: conv state cache must be [slots, 3 (+ speculative tokens), "
        "4096] fp16 (dim-first channels), got ",
        cs.sizes(),
        " ",
        cs.strides());
    TORCH_CHECK(
        ss.scalar_type() == at::kFloat && ss.dim() == 4 && ss.size(1) == NVH &&
            ss.size(2) == DV && ss.size(3) == DK && ss.stride(3) == 1 &&
            ss.stride(2) == DK && ss.stride(1) == DV * DK,
        "mono: ssm state cache must be [slots, 16, 128, 128] fp32, got ",
        ss.sizes(),
        " ",
        ss.strides());
    P.conv_cache = reinterpret_cast<sycl::half*>(cs.data_ptr());
    P.conv_stride = cs.stride(0);
    P.ssm_cache = ss.data_ptr<float>();
    P.ssm_stride = ss.stride(0);
  }
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  auto it = g_layers.find(layer);
  if (it != g_layers.end() &&
      it->second.scratch) {  // re-registration (new KV cache binding): keep the
                             // scratch
    L.scratch = it->second.scratch;
    L.scratch_b = it->second.scratch_b;
  } else {
    L.scratch = sycl::malloc_device(scratch_bytes(), q);
    q.memset(L.scratch, 0, scratch_bytes()).wait();
    L.scratch_b = sycl::malloc_device(scratch_b_bytes(), q);
    q.memset(L.scratch_b, 0, scratch_b_bytes()).wait();
  }
  L.SB = scratch_b_carve(L.scratch_b);
  if (attn) {
    void*& pa = g_attn_part[L.device];
    if (!pa) pa = sycl::malloc_device(attn_part_bytes(), q);
    L.SB.pa_o = static_cast<sycl::half*>(pa);
    L.SB.pa_ml = reinterpret_cast<float*>(
        static_cast<char*>(pa) + (size_t)APASS * APAIRS * NQ * HD * 2);
  }
  L.S = scratch_carve(
      L.scratch,
      reinterpret_cast<uint64_t*>(mbox),
      reinterpret_cast<uint64_t*>(peer));
  g_layers[layer] = L;
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
  using namespace ::mono_decode;
  auto it = g_layers.find(layer);
  TORCH_CHECK(it != g_layers.end(), "mono: layer ", layer, " not registered");
  Layer& L = it->second;
  TORCH_CHECK(
      hidden.get_device() == L.device,
      "mono: layer ",
      layer,
      " registered on another device");
  // rows: 1 (single-token kernels) or a padded batch of 2, 4, 8 (batched)
  const int64_t nb = hidden.dim() == 2 ? hidden.size(0) : 1;
  TORCH_CHECK(
      nb == 1 || nb == 2 || nb == 4 || nb == 8,
      "mono: batch of ",
      nb,
      " rows (supported: 1, 2, 4, 8)");
  Ptrs P = L.P;
  P.hs_in = f16(hidden, nb * HID, "hidden");
  P.res_in = f16(residual, nb * HID, "residual");
  P.hid_out = const_cast<sycl::half*>(f16(hid_out, nb * HID, "hid_out"));
  P.res_out = const_cast<sycl::half*>(f16(res_out, nb * HID, "res_out"));
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  if (L.attn) {
    TORCH_CHECK(
        m1.has_value() && m2.has_value(),
        "mono: attention layer needs seq_lens and slot_mapping");
    TORCH_CHECK(
        m0.scalar_type() == at::kInt && m0.stride(-1) == 1,
        "mono: block_table must be int32");
    TORCH_CHECK(m1->scalar_type() == at::kInt, "mono: seq_lens must be int32");
    TORCH_CHECK(
        m2->scalar_type() == at::kLong, "mono: slot_mapping must be int64");
    TORCH_CHECK(
        positions.scalar_type() == at::kLong, "mono: positions must be int64");
    TORCH_CHECK(
        m0.dim() == 2 && m0.size(0) >= nb && m1->numel() >= nb &&
            m2->numel() >= nb && positions.size(-1) >= nb,
        "mono: attention metadata smaller than the batch");
    P.block_table = m0.data_ptr<int>();
    P.bt_stride = (int)m0.stride(0);
    P.seq_lens = m1->data_ptr<int>();
    P.slot = m2->data_ptr<int64_t>();
    P.pos = positions.data_ptr<int64_t>();  // [T] or mrope [3, T]: row 0
    if (nb == 1)
      launch_attn(q, P, L.S);
    else
      launch_attn_b(q, (int)nb, P, L.S, L.SB);
  } else {
    TORCH_CHECK(
        m0.scalar_type() == at::kInt, "mono: state indices must be int32");
    TORCH_CHECK(m0.numel() >= nb, "mono: state indices smaller than the batch");
    P.state_idx = m0.data_ptr<int>();
    if (nb == 1)
      launch_gdn(q, P, L.S);
    else
      launch_gdn_b(q, (int)nb, P, L.S, L.SB);
  }
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
  using namespace ::mono_decode;
  auto it = g_layers.find(layer);
  TORCH_CHECK(it != g_layers.end(), "mono: layer ", layer, " not registered");
  Layer& L = it->second;
  TORCH_CHECK(
      hidden.get_device() == L.device,
      "mono: layer ",
      layer,
      " registered on another device");
  const int64_t nb = hidden.dim() == 2 ? hidden.size(0) : 1;
  TORCH_CHECK(
      tokens >= 2 && nb % tokens == 0,
      "mono: spec verify needs rows = requests x tokens, got ",
      nb,
      " rows of ",
      tokens,
      " tokens");
  const int64_t nrq = nb / tokens;
  Ptrs P = L.P;
  P.hs_in = f16(hidden, nb * HID, "hidden");
  P.res_in = f16(residual, nb * HID, "residual");
  P.hid_out = const_cast<sycl::half*>(f16(hid_out, nb * HID, "hid_out"));
  P.res_out = const_cast<sycl::half*>(f16(res_out, nb * HID, "res_out"));
  TORCH_CHECK(
      m0.scalar_type() == at::kInt && m0.dim() == 2 && m0.stride(1) == 1 &&
          m0.size(0) >= nrq,
      "mono: spec block table / state indices must be int32 [requests, n]");
  TORCH_CHECK(
      m1.scalar_type() == at::kInt && m1.numel() >= nrq,
      "mono: spec seq_lens / num_accepted_tokens must be int32 [requests]");
  P.block_table = m0.data_ptr<int>();
  P.bt_stride = (int)m0.stride(0);
  P.seq_lens = m1.data_ptr<int>();
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  if (L.attn) {
    TORCH_CHECK(
        m2.has_value() && m2->scalar_type() == at::kLong && m2->numel() >= nb,
        "mono: spec attention needs slot_mapping int64 [rows]");
    TORCH_CHECK(
        positions.scalar_type() == at::kLong && positions.size(-1) >= nb,
        "mono: positions must be int64 [rows]");
    P.slot = m2->data_ptr<int64_t>();
    P.pos = positions.data_ptr<int64_t>();
    return launch_attn_s(q, (int)nb, (int)tokens, P, L.S, L.SB);
  }
  TORCH_CHECK(m0.size(1) >= tokens, "mono: spec state indices narrower than T");
  return launch_gdn_s(q, (int)nb, (int)tokens, P, L.S, L.SB);
}

void mono_register_mtp(
    int64_t layer,
    int64_t mbox,
    int64_t peer,
    const std::vector<torch::Tensor>& t,
    int64_t rank) {
  require_xe2("mono_register_mtp");
  using namespace ::mono_decode;
  TORCH_CHECK(t.size() == 4u, "mono: MTP prologue takes 4 tensors");
  TORCH_CHECK(rank == 0 || rank == 1, "mono: MTP prologue is TP2 only");
  Layer L;
  L.mtp = true;
  L.rank = (int)rank;
  L.device = t[0].get_device();
  Ptrs& P = L.P;
  P.in_norm_w = f16(t[0], HID, "pre_fc_norm_embedding");
  P.post_norm_w = f16(t[1], HID, "pre_fc_norm_hidden");
  P.qkv_w = w_nk(t[2], MTP_FC_OUT, MTP_FC_IN, "fc_w");
  P.qkv_s = scale1(t[3], "fc_scale");
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  auto it = g_layers.find(layer);
  if (it != g_layers.end() && it->second.scratch) {
    L.scratch = it->second.scratch;
  } else {
    L.scratch = sycl::malloc_device(scratch_bytes(), q);
    q.memset(L.scratch, 0, scratch_bytes()).wait();
  }
  L.S = scratch_carve(
      L.scratch,
      reinterpret_cast<uint64_t*>(mbox),
      reinterpret_cast<uint64_t*>(peer));
  g_layers[layer] = L;
}

bool mono_mtp_prologue(
    int64_t layer, const torch::Tensor& x, torch::Tensor& out) {
  using namespace ::mono_decode;
  auto it = g_layers.find(layer);
  TORCH_CHECK(
      it != g_layers.end() && it->second.mtp,
      "mono: MTP prologue ",
      layer,
      " not registered");
  Layer& L = it->second;
  const int64_t nb = x.dim() == 2 ? x.size(0) : 1;
  Ptrs P = L.P;
  P.hs_in = f16(x, nb * MTP_FC_IN, "x");
  P.hid_out = const_cast<sycl::half*>(f16(out, nb * HID, "out"));
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  return launch_mtp(q, (int)nb, P, L.S, L.rank);
}

}  // namespace vllm::mono_decode
