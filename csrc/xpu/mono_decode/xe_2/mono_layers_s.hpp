// SPDX-License-Identifier: Apache-2.0
//
// Speculative-decoding verify kernels (mono decode, TP2): NB = NRQ requests x
// T tokens (T = 1 + draft tokens), rows request-major (row = b * T + t),
// separate from the batched kernels (one request per row). The projections,
// the tail and the MoE are the batched code for NB rows; the GDN mixer runs
// each request's tokens in order.
//
// GDN state layout (vLLM spec decode, as the XPU gdn_attention spec kernels):
// - P.block_table [NRQ][P.bt_stride]: state slots of request b for tokens
//   0 .. T-1 (spec_state_indices_tensor).
// - P.seq_lens [NRQ]: num_accepted_tokens of the previous step (>= 1).
// - Recurrent state: read from slot [b][acc - 1]; after token t, written to
//   slot [b][t].
// - Conv state: one line per request (slot [b][0]) of T + 2 rows of CONVD.
//   The window starts at row acc - 1; the line is written back as rows
//   (acc, acc + 1) of the old line followed by the T raw inputs.
#pragma once
#include "mono_layers_b.hpp"

namespace mono_decode {

template <int NB, int T>
inline void gdn_layer_s(
    sycl::nd_item<1> it,
    uint8_t* slm,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB) {
  constexpr int NRQ = NB / T;
  static_assert(NRQ * T == NB && T >= 2, "spec verify: NB = requests x tokens");
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;
  const SlmB L = slm_b(slm);
  float* qk = reinterpret_cast<float*>(
      slm +
      B_QK);  // per row r: q[0..127] k[128..255] v[256..319] beta g [320, 321]

  const int h = wg >> 1, c = wg & 1,
            kh = h >> 1;  // S2 role: v head h (k head h/2), v rows c*64..+64
  int acc0[NRQ];          // window row / initial state column (acc - 1)
#pragma unroll
  for (int b = 0; b < NRQ; ++b)
    acc0[b] = sycl::max(P.seq_lens[b] - 1, 0);
  auto slot = [&](int b, int t) { return P.block_table[b * P.bt_stride + t]; };
  auto line_of = [&](int b) {
    return P.conv_cache + (size_t)slot(b, 0) * P.conv_stride;
  };
  auto ssm_at = [&](int sl) { return P.ssm_cache + (size_t)sl * P.ssm_stride; };

  // ---- kernel start: S1 weights in flight, the requests' initial
  // recurrent-state blocks of this WG -> L2
  const int tile1 = s >> 2, ks1 = s & 3;
  sycl::uint4 w1[32];
  ld_rows<32>(
      w1, P.qkvz_w + (size_t)(wg * 128 + tile1 * 16 + lane) * HID + ks1 * 512);
#pragma unroll
  for (int b = 0; b < NRQ; ++b)
    syclex::prefetch(
        (void*)(ssm_at(slot(b, acc0[b])) + ((size_t)h * DV + c * 64) * DK +
                lid * 16),
        syclex::properties{syclex::prefetch_hint_L2});
  const int cch = lid < 128   ? kh * DK + lid
                  : lid < 256 ? KD + kh * DK + (lid - 128)
                              : 2 * KD + h * DV + c * 64 + (lid - 256);
  half cwt[4];
  half* csl = reinterpret_cast<half*>(
      slm + B_CS);  // [b][i][320]: window row i of request b, the WG's channels
  for (int v = lid; v < NRQ * 3 * 40; v += WGS) {
    const int bi = v / 40, j8 = (v % 40) * 8, b = bi / 3, i = bi % 3;
    const int ch = j8 < 128   ? kh * DK + j8
                   : j8 < 256 ? KD + kh * DK + (j8 - 128)
                              : 2 * KD + h * DV + c * 64 + (j8 - 256);
    *reinterpret_cast<sycl::vec<half, 8>*>(csl + bi * 320 + j8) =
        *reinterpret_cast<const sycl::vec<half, 8>*>(
            line_of(b) + (size_t)(acc0[b] + i) * CONVD + ch);
  }
  if (lid < 320) {
    const sycl::vec<half, 4> w4 = *reinterpret_cast<const sycl::vec<half, 4>*>(
        P.conv_w + (size_t)cch * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      cwt[i] = w4[i];
  }
  input_norm_b<NB>(it, P, L);
  wg_bar(grp);

  // --- S1: q|k|v rows (128 per WG), z, b|a: as the batched kernel ------------
  const int tz = s >> 3, kz = s & 7;
  sycl::uint4 zw[16], bw[8];
  {
    v8f cc = v8f(0.f);
    mm_rows<32>(cc, w1, L.xa, ks1 * 32, lane);
    ld_rows<16>(
        zw,
        P.qkvz_w + (size_t)(CONVD + h * DV + c * 64 + tz * 16 + lane) * HID +
            kz * 256);
    ld_rows<8>(
        bw, P.ba_w + (size_t)((s >> 4) * 16 + lane) * HID + (s & 15) * 128);
    red_put(L.red, s, lane, cc);
  }
  wg_bar(grp);
  {
    const float sc = 256.f * P.qkvz_s[0];
    for (int idx = lid; idx < NB * 128; idx += WGS) {
      const int b = idx >> 7, r = idx & 127, t = r >> 4, l = r & 15;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 4; ++k2)
        v += red_get(L.red, t * 4 + k2, b, l);
      SB.qkv[b * CONVD + wg * 128 + r] = (half)(v * sc);
    }
  }
  signal(it, S.cnt + C_A * 16);
  {
    v8f cz = v8f(0.f), cb = v8f(0.f);
    mm_rows<16>(cz, zw, L.xa, kz * 16, lane);
    mm_rows<8>(cb, bw, L.xa, (s & 15) * 8, lane);
    wg_bar(grp);
    red_put(L.red, s, lane, cz);
    if (lane == h) {
#pragma unroll
      for (int b = 0; b < NB; ++b)
        L.sm[256 + (s >> 4) * 128 + b * 16 + (s & 15)] = cb[b];
    }
  }
  wg_bar(grp);
  {
    const float sc = 256.f * P.qkvz_s[0];
    for (int idx = lid; idx < NB * 64; idx += WGS) {
      const int b = idx >> 6, r = idx & 63, t = r >> 4, l = r & 15;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 8; ++k2)
        v += red_get(L.red, t * 8 + k2, b, l);
      SB.z[b * VD + h * DV + c * 64 + r] = (half)(v * sc);
    }
    if (lid < 2 * NB) {
      const int which = lid / NB, b = lid % NB;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 16; ++k2)
        v += L.sm[256 + which * 128 + b * 16 + k2];
      L.sm[512 + which * 8 + b] = (float)(half)(v * 256.f * P.ba_s[0]);
    }
  }
  // initial recurrent state of each request (from L2)
  sycl::vec<float, 8> st[NRQ][2];
#pragma unroll
  for (int b = 0; b < NRQ; ++b)
#pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      st[b][rr] = *reinterpret_cast<const sycl::vec<float, 8>*>(
          ssm_at(slot(b, acc0[b])) +
          ((size_t)h * DV + c * 64 + 2 * s + rr) * DK + lane * 8);

  // --- S2: conv1d over each request's window (all tokens at once), gated delta
  // rule in token order ---
  wait_for(it, S.cnt + C_A * 16, NWG * call);
  if (lid < 320) {
#pragma unroll
    for (int b = 0; b < NRQ; ++b) {
      half win[T + 3];
      win[0] = csl[(b * 3) * 320 + lid];
      win[1] = csl[(b * 3 + 1) * 320 + lid];
      win[2] = csl[(b * 3 + 2) * 320 + lid];
#pragma unroll
      for (int t = 0; t < T; ++t)
        win[3 + t] = SB.qkv[(b * T + t) * CONVD + cch];
#pragma unroll
      for (int t = 0; t < T; ++t) {
        float acc = (float)win[t] * (float)cwt[0];
        acc += (float)win[t + 1] * (float)cwt[1];
        acc += (float)win[t + 2] * (float)cwt[2];
        acc += (float)win[t + 3] * (float)cwt[3];
        qk[(b * T + t) * 384 + lid] =
            (float)(half)(acc / (1.f + sycl::exp(-acc)));
      }
      if (lid >= 256 ||
          ((h & 1) == 0 &&
           c == 0)) {  // rolled line: old rows acc, acc + 1, then the T inputs
        half* ln = line_of(b);
#pragma unroll
        for (int j = 0; j < T + 2; ++j)
          ln[(size_t)j * CONVD + cch] = win[j + 1];
      }
    }
  }
  if (lid < NB) {  // beta, g of v head h per row
    const float bb = L.sm[512 + lid], a = L.sm[520 + lid] + (float)P.dt_bias[h];
    const float sp = a < 20.f ? sycl::log(1.f + sycl::exp(a)) : a;
    qk[lid * 384 + 320] = 1.f / (1.f + sycl::exp(-bb));
    qk[lid * 384 + 321] = sycl::exp(-sycl::exp(P.A_log[h]) * sp);
  }
  wg_bar(grp);
  if (s < 2 * NB) {  // L2 norms: SG (row s/2, q | k)
    const int b = s >> 1, which = s & 1;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float v = qk[b * 384 + which * 128 + lane * 8 + i];
      ss += v * v;
    }
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    if (lane == 0) L.sm[600 + b * 2 + which] = ss;
  }
  wg_bar(grp);
#pragma unroll
  for (int b = 0; b < NRQ; ++b) {
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const int row = b * T + t;
      const float* q_ = qk + row * 384;
      const float iq = 1.f / sycl::sqrt(L.sm[600 + row * 2] + 1e-6f),
                  ik = 1.f / sycl::sqrt(L.sm[600 + row * 2 + 1] + 1e-6f);
      const float scale = 0.08838834764831845f;
      float qv[8], kv8[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        qv[i] = q_[lane * 8 + i] * iq * scale;
        kv8[i] = q_[128 + lane * 8 + i] * ik;
      }
      const float beta = q_[320], g = q_[321];
      float kvm[2], o[2];
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          st[b][rr][i] *= g;
          d += st[b][rr][i] * kv8[i];
        }
        kvm[rr] = d;
      }
#pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        kvm[rr] = sycl::reduce_over_group(sg, kvm[rr], sycl::plus<float>());
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const float delta = (q_[256 + 2 * s + rr] - kvm[rr]) * beta;
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          st[b][rr][i] += kv8[i] * delta;
          d += st[b][rr][i] * qv[i];
        }
        o[rr] = d;
      }
#pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        o[rr] = sycl::reduce_over_group(sg, o[rr], sycl::plus<float>());
      if (lane < 2)
        SB.core[row * VD + h * DV + c * 64 + 2 * s + lane] =
            (half)(lane == 0 ? o[0] : o[1]);
      float* dst =
          ssm_at(slot(b, t));  // this token's state slot (rollback position t)
#pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        st_store8(
            dst + ((size_t)h * DV + c * 64 + 2 * s + rr) * DK + lane * 8,
            st[b][rr]);
    }
  }
  signal(it, S.cnt + C_B * 16);
  sycl::uint4 ow[16], prt_r[4], prt_s[4];
  ld_rows<16>(
      ow,
      P.out_w + (size_t)(wg * 64 + (s >> 3) * 16 + lane) * VD + (s & 7) * 256);
  load_prt_b(P, wg, s, lane, prt_r, prt_s);

  // --- S3: gated RMSNorm (all heads, all rows) -> perm A tiles ; tail --------
  wait_for(it, S.cnt + C_B * 16, NWG * call);
  {
    sycl::vec<half, 4> o4[NB], z4[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      o4[b] = *reinterpret_cast<const sycl::vec<half, 4>*>(
          SB.core + b * VD + lid * 4);
      z4[b] =
          *reinterpret_cast<const sycl::vec<half, 4>*>(SB.z + b * VD + lid * 4);
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ss += (float)o4[b][i] * (float)o4[b][i];
      ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
      if (lane == 0) L.sm[640 + b * 32 + s] = ss;
    }
    wg_bar(grp);
    const int hh = s >> 1;
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float rs = sycl::rsqrt(
          (L.sm[640 + b * 32 + 2 * hh] + L.sm[640 + b * 32 + 2 * hh + 1]) / DV +
          EPS);
      half y[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float zf = (float)z4[b][i];
        y[i] = (half)((float)o4[b][i] * rs *
                      (float)P.gnorm_w[(lid * 4 + i) & (DV - 1)] *
                      (zf / (1.f + sycl::exp(-zf))));
      }
      a_put4_perm(L.xa, b, lid * 4, y);
    }
    wg_bar(grp);
  }
  tail_b<NB>(it, P, S, SB, L, call, C_O, ow, prt_r, prt_s);
}

template <int NB, int T>
sycl::event launch_gdn_s(
    sycl::queue& q, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(B_BYTES), h);
    syclex::properties props{intelex::grf_size<256>};
    h.parallel_for(
        sycl::nd_range<1>(NWG * WGS, WGS),
        props,
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          gdn_layer_s<NB, T>(
              it,
              slm.get_multi_ptr<sycl::access::decorated::no>().get(),
              P,
              S,
              SB);
        });
  });
}

// ------------------------------------------------ full attention (spec verify)
// The batched attention with row r = (request r / T, token r % T):
// - the context chunks of a row are its request's cached tokens
//   (P.seq_lens[request] - T, P.block_table row = request); the T rows of a
//   request read the same KV, mostly from L2;
// - the T new k / v are written to the cache (P.slot[row]) and merged
//   causally: row r takes the new tokens 0 .. r % T of its request.
template <int NB, int T>
inline void attn_layer_s(
    sycl::nd_item<1> it,
    uint8_t* slm,
    const Ptrs& P,
    const Scratch& S,
    const ScratchB& SB) {
  static_assert(NB % T == 0 && T >= 2, "spec verify: NB = requests x tokens");
  const int wg = it.get_group(0);
  const int lid = it.get_local_linear_id();
  sycl::sub_group sg = it.get_sub_group();
  const int s = sg.get_group_linear_id();
  const int lane = sg.get_local_linear_id();
  auto grp = it.get_group();
  const uint32_t call = uc_load32(S.cnt + EPOCH) + 1u;
  const SlmB L = slm_b(slm);
  // the single-row kernel takes gate/k/v chunks from a ticket counter whose
  // base it derives from the call number: every call must consume the same
  // number of tickets
  if (wg == 0 && lid == 0)
    cnt_ref(S.cnt[C_TKT * 16]).fetch_add((uint32_t)GKV_CHUNKS + NSG * NWG);
  uint32_t* qpk_all = reinterpret_cast<uint32_t*>(slm + AB_QPK);
  float* rope_all = reinterpret_cast<float*>(slm + AB_CS);
  half* kvn_all = reinterpret_cast<half*>(slm + AB_KVN);
  half* qs_all = reinterpret_cast<half*>(slm + AB_QS);
  float* wgt = reinterpret_cast<float*>(slm + AB_W);
  float* sm3 = reinterpret_cast<float*>(slm + AB_SM3);
  auto qrow = [](int qi) { return (qi >> 8) * 512 + (qi & 255); };
  auto grow = [](int gi) {
    return gi < NQ * HD ? (gi >> 8) * 512 + 256 + (gi & 255)
                        : NQ * 512 + (gi - NQ * HD);
  };
  const int bs = P.bs;
  // context chunks: one per SG pair at most; the chunk length ach (a multiple
  // of 16) is sized per call so that the rows' chunks (row b: ceil(lc_b / ach),
  // chunks coff[b] .. coff[b + 1] - 1) fit the APAIRS pairs
  int lcs[NB], coff[NB + 1];
  int tot = 0;
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    lcs[b] = sycl::max(P.seq_lens[b / T] - T, 0);
    tot += lcs[b];
  }
  // long steps (many cached tokens): row groups with dynamic units; short ones:
  // static chunks (no ticket atomics)
  const bool dyn = tot >= ADYN_MIN;
  int ach = 0;
  if (dyn) {  // row groups: coff[b] = first pair of row b, by length; at least
              // one pair per non-empty row
    {
      int ne = 0;
#pragma unroll
      for (int b = 0; b < NB; ++b)
        ne += lcs[b] > 0;
      coff[0] = 0;
      long cum = 0;
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        cum += lcs[b];
        ne -= lcs[b] > 0;
        int e = tot > 0 ? (int)((cum * APAIRS + tot / 2) / tot) : 0;
        e = sycl::max(e, coff[b] + (lcs[b] > 0 ? 1 : 0));
        e = sycl::min(e, APAIRS - ne);
        coff[b + 1] = e;
      }
    }
  } else {  // chunks: coff[b] = first chunk of row b
    ach = sycl::max(16, ((tot + APAIRS - NB - 1) / (APAIRS - NB) + 15) & ~15);
    coff[0] = 0;
#pragma unroll
    for (int b = 0; b < NB; ++b)
      coff[b + 1] = coff[b] + (lcs[b] + ach - 1) / ach;
  }

  // ---- S1: q rows (64 per WG: 4 tiles x 8 K slices), weights in flight before
  // the input norm
  sycl::uint4 wq[16];
  ld_rows<16>(
      wq,
      P.qkv_w + (size_t)qrow(wg * 64 + (s >> 3) * 16 + lane) * HID +
          (s & 7) * 256);
  input_norm_b<NB>(it, P, L);
  wg_bar(grp);
  {
    v8f cc = v8f(0.f);
    mm_rows<16>(cc, wq, L.xa, (s & 7) * 16, lane);
    red_put(L.red, s, lane, cc);
  }
  wg_bar(grp);
  {
    const float sc = 256.f * P.qkv_s[0];
    for (int idx = lid; idx < NB * 64; idx += WGS) {
      const int b = idx >> 6, r = idx & 63, t = r >> 4, l = r & 15;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 8; ++k2)
        v += red_get(L.red, t * 8 + k2, b, l);
      SB.qkv_raw[b * QKV_ROWS + qrow(wg * 64 + r)] = v * sc;
    }
  }
  signal(it, S.cnt + C_A * 16);
  if (lid < 32 * NB) {  // RoPE tables per row: Cody-Waite reduction by 2*pi,
                        // then cos/sin of |r| <= pi
    const int b = lid >> 5, i = lid & 31;
    const float inv = 1.0f / sycl::pow(THETA, (float)(2 * i) / 64.f);
    const float f = (float)P.pos[b] * inv;
    const float k = sycl::rint(f * 0.15915494309189535f);
    float r = sycl::fma(-k, 6.28125f, f);
    r = sycl::fma(-k, 1.9353071795864769e-3f, r);
    rope_all[b * 64 + i] = sycl::cos(r);
    rope_all[b * 64 + 32 + i] = sycl::sin(r);
  }

  if (lid < 48)
    reinterpret_cast<int*>(slm + AB_TK)[lid] =
        -1;  // ticket ring / acks: nothing yet
  // ---- S2: q norm + RoPE per row, attention over the flattened context chunks
  // -> partials (global)
  wait_for(it, S.cnt + C_A * 16, NWG * call);
  for (int p = s; p < NB * NQ; p += NSG) {  // (row, head) per SG: q norm + RoPE
                                            // straight into the dpas A order
    // qpk[(kt*16 + l)*4 + j] = (q[2j + l/8][kt*16 + 2(l%8)], +1): head hh, dim
    // lane*16 + i -> half
    // ((lane*16 + (hh&1)*8 + i/2)*4 + hh/2)*2 + i%2
    const int b = p / NQ, hh = p % NQ;
    const float* src = SB.qkv_raw + b * QKV_ROWS + hh * 512;
    const float* cs = rope_all + b * 64;
    float y[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
      y[i] = (float)(half)src[lane * 16 + i];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i)
      ss += y[i] * y[i];
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    const float rs = sycl::rsqrt(ss / HD + EPS);
#pragma unroll
    for (int i = 0; i < 16; ++i)
      y[i] = y[i] * rs * (1.f + (float)P.q_norm_w[lane * 16 + i]);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float other = sycl::permute_group_by_xor(sg, y[i], 2);
      if (lane < 4) {
        const int fi = (lane & 1) * 16 + i;
        y[i] = lane < 2 ? y[i] * cs[fi] - other * cs[32 + fi]
                        : y[i] * cs[fi] + other * cs[32 + fi];
      }
    }
    half* qh = reinterpret_cast<half*>(qpk_all + b * 1024);
#pragma unroll
    for (int i = 0; i < 16; ++i)
      qh[((lane * 16 + (hh & 1) * 8 + (i >> 1)) * 4 + (hh >> 1)) * 2 +
         (i & 1)] = (half)y[i];
  }
  wg_bar(grp);
  {
    if (dyn) {
      // row groups: row b owns the pairs coff[b] .. coff[b + 1] - 1 (in
      // proportion to its length, at least one if not empty); the group takes
      // units of 16 tokens as tickets from the row's counter (tickets of 4
      // units for the first 3/4, then single units), so the pairs' speed
      // differences even out inside the group. One partial per pair (slot =
      // pair): the merge is the chunk one.
      const int dh = s & 1, pl = s >> 1, pr = wg * 16 + pl;
      int* tkr = reinterpret_cast<int*>(slm + AB_TK);  // [16][2]
      int* ack = tkr + 32;                             // [16]
      int b = 0;
#pragma unroll
      for (int bb = 1; bb < NB; ++bb)
        if (coff[bb] <= pr) b = bb;
      if (pr < coff[NB]) {
        const int lc = lcs[b];
        const int* btab = P.block_table + (size_t)(b / T) * P.bt_stride;
        const uint32_t* qpk = qpk_all + b * 1024;
        v8f O[8];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
          O[nt] = v8f(0.f);
        float m[8], lp[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          m[i] = -INFINITY;
          lp[i] = 0.f;
        }
        using lat = sycl::atomic_ref<
            int,
            sycl::memory_order::relaxed,
            sycl::memory_scope::work_group,
            sycl::access::address_space::local_space>;
        const int nu = (lc + 15) / 16, na = nu * 3 / 16;
#pragma unroll 1
        for (int k = 0;; ++k) {
          int t = 0;
          if (lane == 0) {
            if (dh == 0) {
              if (k >= 2)
                while (lat(ack[pl]).load(sycl::memory_order::acquire) < k - 2) {
                }
              t = (int)cnt_ref(S.cnt[ATK + b])
                      .fetch_add(1u, sycl::memory_order::relaxed);
              lat(tkr[pl * 2 + (k & 1)])
                  .store(
                      (k << 16) | sycl::min(t, 65535),
                      sycl::memory_order::release);
            } else {
              int v;
              do {
                v = lat(tkr[pl * 2 + (k & 1)])
                        .load(sycl::memory_order::acquire);
              } while ((v >> 16) != k);
              t = v & 0xffff;
              lat(ack[pl]).store(k, sycl::memory_order::release);
            }
          }
          t = sycl::select_from_group(sg, t, 0);
          const int u = t < na ? 4 * t : 4 * na + (t - na);
          if (u >= nu) break;
          const int t0 = u * 16,
                    te = sycl::min(lc, (u + (t < na ? 4 : 1)) * 16);
          auto trow = [&](int tk) {  // K row of token tk of row b (a 16-token
                                     // tile never crosses a block); past the
            if (tk >= te)
              tk = 0;  // segment: token 0 (masked by the score select)
            return P.kv + (size_t)btab[tk / bs] * P.blk_stride +
                   (size_t)(tk % bs) * KROW;
          };
          if ((lc & 15) && (lc & ~15) >= t0 &&
              (lc & ~15) < te) {  // stale V rows of the partial tile -> 0
            half* vz = const_cast<half*>(trow(lc & ~15)) + HD + dh * 128;
            for (int r = lc & 15; r < 16; ++r)
              *reinterpret_cast<sycl::vec<half, 8>*>(
                  vz + (size_t)r * KROW + lane * 8) = sycl::vec<half, 8>(0.f);
            sycl::atomic_fence(
                sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
          }
          {
            const int lc = te;  // the step masks columns >= lc
            int tq = t0;
#pragma unroll 1
            for (; tq + 48 < lc; tq += 64)
              MONO_ATTN_STEP(tq, 4);
#pragma unroll 1
            for (; tq < lc; tq += 16)
              MONO_ATTN_STEP(tq, 1);
          }
        }
        const int g = pr;
#pragma unroll
        for (int i = 0; i < 8;
             ++i) {  // partial: normalized o of this dims half, (m, l)
          const float l =
              sycl::reduce_over_group(sg, lp[i], sycl::plus<float>());
          const float il = l > 0.f ? 1.f / l : 0.f;
          half* dst = SB.pa_o + ((size_t)g * NQ + i) * HD + dh * 128;
#pragma unroll
          for (int nt = 0; nt < 8; ++nt)
            dst[nt * 16 + lane] = (half)(O[nt][i] * il);
          if (dh == 0 && lane == 0) {
            SB.pa_ml[((size_t)g * NQ + i) * 2] = m[i];
            SB.pa_ml[((size_t)g * NQ + i) * 2 + 1] = l;
          }
        }
      }
    } else {
      const int dh = s & 1, g = wg * 16 + (s >> 1);
      if (g < coff[NB]) {
        int b = 0;
#pragma unroll
        for (int bb = 1; bb < NB; ++bb)
          if (coff[bb] <= g) b = bb;
        const int t0 = (g - coff[b]) * ach;
        const int lcr = lcs[b];
        const int lc = sycl::min(
            lcr,
            t0 + ach);  // end of this chunk (masks the tokens of the next one)
        const int* btab = P.block_table + (size_t)(b / T) * P.bt_stride;
        const uint32_t* qpk = qpk_all + b * 1024;
        auto trow = [&](int t) {  // K row of token t of row b (a 16-token tile
                                  // never crosses a block); past the
          if (t >= lc) t = 0;     // chunk: token 0 (masked by the score select)
          return P.kv + (size_t)btab[t / bs] * P.blk_stride +
                 (size_t)(t % bs) * KROW;
        };
        v8f O[8];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
          O[nt] = v8f(0.f);
        float m[8], lp[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          m[i] = -INFINITY;
          lp[i] = 0.f;
        }
        if ((lcr & 15) && (lcr & ~15) >= t0 &&
            (lcr & ~15) < lc) {  // stale V rows of the partial tile -> 0
          half* vz = const_cast<half*>(trow(lcr & ~15)) + HD + dh * 128;
          for (int r = lcr & 15; r < 16; ++r)
            *reinterpret_cast<sycl::vec<half, 8>*>(
                vz + (size_t)r * KROW + lane * 8) = sycl::vec<half, 8>(0.f);
          sycl::atomic_fence(
              sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
        }
        int t = t0;
#pragma unroll 1
        for (; t + 48 < lc; t += 64)
          MONO_ATTN_STEP(t, 4);
#pragma unroll 1
        for (; t < lc; t += 16)
          MONO_ATTN_STEP(t, 1);
#pragma unroll
        for (int i = 0; i < 8;
             ++i) {  // chunk partial: normalized o of this dims half, (m, l)
          const float l =
              sycl::reduce_over_group(sg, lp[i], sycl::plus<float>());
          const float il = l > 0.f ? 1.f / l : 0.f;
          half* dst = SB.pa_o + ((size_t)g * NQ + i) * HD + dh * 128;
#pragma unroll
          for (int nt = 0; nt < 8; ++nt)
            dst[nt * 16 + lane] = (half)(O[nt][i] * il);
          if (dh == 0 && lane == 0) {
            SB.pa_ml[((size_t)g * NQ + i) * 2] = m[i];
            SB.pa_ml[((size_t)g * NQ + i) * 2 + 1] = l;
          }
        }
      }
    }
  }
  // ---- gate / k / v rows (80 per WG: 5 tiles x 4 K slices, SGs 0..19)
  {
    sycl::uint4 wg4[32];
    const int t = s >> 2, ks = s & 3;
    if (s < 20)
      ld_rows<32>(
          wg4,
          P.qkv_w + (size_t)grow(wg * 80 + t * 16 + lane) * HID + ks * 512);
    v8f cc = v8f(0.f);
    if (s < 20) mm_rows<32>(cc, wg4, L.xa, ks * 32, lane);
    red_put(L.red, s, lane, cc);
  }
  wg_bar(grp);
  {
    const float sc = 256.f * P.qkv_s[0];
    for (int idx = lid; idx < NB * 80; idx += WGS) {
      const int b = idx / 80, r = idx % 80, t = r >> 4, l = r & 15;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 4; ++k2)
        v += red_get(L.red, t * 4 + k2, b, l);
      SB.qkv_raw[b * QKV_ROWS + grow(wg * 80 + r)] = v * sc;
    }
  }
  signal(it, S.cnt + C_B * 16);
  sycl::uint4 ow[16], prt_r[4], prt_s[4];
  ld_rows<16>(
      ow,
      P.out_w + (size_t)(wg * 64 + (s >> 3) * 16 + lane) * (NQ * HD) +
          (s & 7) * 256);

  // ---- S3: per row: new k (norm + RoPE) / v, KV write (WG 0), merge the row's
  // chunk partials + the new token, gate
  wait_for(it, S.cnt + C_B * 16, NWG * call);
  {
    const int hh = wg >> 2, d0 = (wg & 3) * 64;
    if (s < NB) {
      const int b = s;
      const int64_t slot = P.slot[b];
      half* row = (wg == 0 && slot >= 0)
                      ? P.kv + (size_t)(slot / bs) * P.blk_stride +
                            (size_t)(slot % bs) * KROW
                      : nullptr;
      head_norm_rope(
          sg,
          lane,
          SB.qkv_raw + b * QKV_ROWS + NQ * 512,
          P.k_norm_w,
          rope_all + b * 64,
          kvn_all + b * 512,
          row);
    } else if (s >= 16 && s < 16 + NB) {
      const int b = s - 16;
      const int64_t slot = P.slot[b];
      half* row = (wg == 0 && slot >= 0)
                      ? P.kv + (size_t)(slot / bs) * P.blk_stride +
                            (size_t)(slot % bs) * KROW + HD
                      : nullptr;
      head_norm_rope(
          sg,
          lane,
          SB.qkv_raw + b * QKV_ROWS + NQ * 512 + HD,
          nullptr,
          rope_all + b * 64,
          kvn_all + b * 512 + HD,
          row);
    }
    wg_bar(grp);
    float* sn = sm3 + 128;  // [NB][T] new-token scores of head hh (row r, token
                            // j of its request; -inf: j > r % T)
    for (int p = s; p < NB * T; p += NSG) {
      const int r = p / T, j = p % T, kr = (r / T) * T + j;
      float d = 0.f;
      if (j <= r % T) {
#pragma unroll
        for (int i = 0; i < 16;
             ++i) {  // q[hh][dd] from the packed q: kt = dd/16, l = (hh&1)*8 +
                     // (dd%16)/2, j = hh/2
          const int dd = lane * 16 + i, kt = dd >> 4,
                    l = (hh & 1) * 8 + ((dd & 15) >> 1);
          const half qv = reinterpret_cast<const half*>(
              qpk_all +
              r * 1024)[((kt * 16 + l) * 4 + (hh >> 1)) * 2 + (dd & 1)];
          d += (float)qv * (float)kvn_all[kr * 512 + dd];
        }
        d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
      }
      if (lane == 0) sn[p] = j <= r % T ? d * SM_SCALE : -INFINITY;
    }
    wg_bar(grp);
    {  // merge, all rows at once: a work-item per chunk for the maxima /
       // weights, TW work-items per row
      // for the weighted sums (deterministic: atomic max only, fixed-order
      // sums)
      int* mkey = reinterpret_cast<int*>(sm3 + 16);
      float* lwp = sm3 + 32;  // [NB][TW / 16] per-SG weight sums
      auto key = [](float f) {
        const int k = sycl::bit_cast<int>(f);
        return k >= 0 ? k : k ^ 0x7fffffff;
      };
      auto unkey = [](int k) {
        return sycl::bit_cast<float>(k >= 0 ? k : k ^ 0x7fffffff);
      };
      if (lid < NB) {
        float mx = sn[lid * T];
#pragma unroll
        for (int j = 1; j < T; ++j)
          mx = sycl::fmax(mx, sn[lid * T + j]);
        mkey[lid] = key(mx);
      }
      wg_bar(grp);
      const int g = lid;
      int gb = 0;
#pragma unroll
      for (int bb = 1; bb < NB; ++bb)
        if (coff[bb] <= g) gb = bb;
      const bool live = g < coff[NB];
      float mg = -INFINITY, lg = 0.f;
      if (live) {
        mg = SB.pa_ml[((size_t)g * NQ + hh) * 2];
        lg = SB.pa_ml[((size_t)g * NQ + hh) * 2 + 1];
      }
      if (live && lg > 0.f) {
        sycl::atomic_ref<
            int,
            sycl::memory_order::relaxed,
            sycl::memory_scope::work_group,
            sycl::access::address_space::local_space>
            a(mkey[gb]);
        a.fetch_max(key(mg));
      }
      wg_bar(grp);
      if (live)
        wgt[g] = lg > 0.f ? lg * sycl::native::exp2(mg - unkey(mkey[gb])) : 0.f;
      wg_bar(grp);
      constexpr int TW = WGS / NB / 64 * 64,
                    PARTS =
                        TW / 64;  // whole SGs per row (NB = 3, 6: idle tail)
      const int b = lid / TW, r = lid % TW, part = r >> 6, d = r & 63;
      if (b < NB) {
        const int g0 = coff[b], n = coff[b + 1] - coff[b];
        {
          float lw = 0.f;
          for (int i2 = r; i2 < n; i2 += TW)
            lw += wgt[g0 + i2];
          lw = sycl::reduce_over_group(sg, lw, sycl::plus<float>());
          if (lane == 0) lwp[b * (TW / 16) + (r >> 4)] = lw;
        }
        float acc = 0.f;
        {  // 8 independent loads in flight (the loop is latency bound
           // otherwise)
          float a8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
          int i2 = part;
          for (; i2 + 7 * PARTS < n; i2 += 8 * PARTS) {
#pragma unroll
            for (int q = 0; q < 8; ++q)
              a8[q] +=
                  wgt[g0 + i2 + q * PARTS] *
                  (float)SB.pa_o
                      [((size_t)(g0 + i2 + q * PARTS) * NQ + hh) * HD + d0 + d];
          }
          for (; i2 < n; i2 += PARTS)
            a8[0] +=
                wgt[g0 + i2] *
                (float)SB.pa_o[((size_t)(g0 + i2) * NQ + hh) * HD + d0 + d];
#pragma unroll
          for (int q = 0; q < 8; ++q)
            acc += a8[q];
        }
        L.red[(b * PARTS + part) * 64 + d] = acc;
      }
      wg_bar(grp);
      if (lid < NB * 64) {
        const int b2 = lid >> 6, d2 = lid & 63;
        const float M = unkey(mkey[b2]);
        float Lw = 0.f, a = 0.f;
#pragma unroll
        for (int j = 0; j < T; ++j) {  // the request's new tokens 0 .. b2 % T
                                       // (others: exp2(-inf) = 0)
          const float en = sycl::native::exp2(sn[b2 * T + j] - M);
          Lw += en;
          a += en * (float)kvn_all[((b2 / T) * T + j) * 512 + HD + d0 + d2];
        }
        for (int k2 = 0; k2 < TW / 16; ++k2)
          Lw += lwp[b2 * (TW / 16) + k2];
#pragma unroll
        for (int pp = 0; pp < PARTS; ++pp)
          a += L.red[(b2 * PARTS + pp) * 64 + d2];
        const float o = a / Lw;
        const float gate = SB.qkv_raw[b2 * QKV_ROWS + hh * 512 + 256 + d0 + d2];
        SB.attn_out[b2 * NQ * HD + hh * HD + d0 + d2] =
            (half)(o / (1.f + sycl::exp(-gate)));
      }
    }
  }
  signal(it, S.cnt + C_C * 16);
  load_prt_b(P, wg, s, lane, prt_r, prt_s);

  // ---- S4: o (all heads, all rows) -> perm A tiles ; tail
  wait_for(it, S.cnt + C_C * 16, NWG * call);
  if (wg == 0 && lid < NB)
    cnt_ref(S.cnt[ATK + lid]).store(0u);  // every pair is done with the tickets
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    const sycl::vec<half, 4> o4 = *reinterpret_cast<const sycl::vec<half, 4>*>(
        SB.attn_out + b * NQ * HD + lid * 4);
    const half y[4] = {o4[0], o4[1], o4[2], o4[3]};
    a_put4_perm(L.xa, b, lid * 4, y);
  }
  wg_bar(grp);
  tail_b<NB>(it, P, S, SB, L, call, C_O, ow, prt_r, prt_s);
}

template <int NB, int T>
sycl::event launch_attn_s(
    sycl::queue& q, const Ptrs& P, const Scratch& S, const ScratchB& SB) {
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(AB_SLM), h);
    syclex::properties props{intelex::grf_size<256>};
    h.parallel_for(
        sycl::nd_range<1>(NWG * WGS, WGS),
        props,
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          attn_layer_s<NB, T>(
              it,
              slm.get_multi_ptr<sycl::access::decorated::no>().get(),
              P,
              S,
              SB);
        });
  });
}

}  // namespace mono_decode
