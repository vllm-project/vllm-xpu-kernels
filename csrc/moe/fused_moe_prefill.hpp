#pragma once
#include "fused_moe_common.hpp"

template <
    bool HAS_W13_BIAS,
    bool HAS_W2_BIAS,
    bool HAS_CLAMP_LIMIT = false,
    typename DTYPE = bf16_t,
    typename INTER_DTYPE = bf16_t,
    int TM = ::TM,
    int N_TM = ::N_TM,
    int N_TK = ::N_TK,
    int N_TN = ::N_TN>
struct FusedMoePrefill : FusedMoeCommon<
                             HAS_W13_BIAS,
                             HAS_W2_BIAS,
                             HAS_CLAMP_LIMIT,
                             /*IS_DECODE=*/false,
                             DTYPE,
                             INTER_DTYPE,
                             TM,
                             N_TM,
                             N_TK,
                             N_TN> {
  using Base = FusedMoeCommon<
      HAS_W13_BIAS,
      HAS_W2_BIAS,
      HAS_CLAMP_LIMIT,
      false,
      DTYPE,
      INTER_DTYPE,
      TM,
      N_TM,
      N_TK,
      N_TN>;

  DTYPE* act_buf;  // [num_wgs * N_TM * TM, I]

  // ── grid ─────────────────────────────────────────────────────────────────
  static sycl::nd_range<2>
  get_nd_range(int total_tokens, int num_experts, int /*K*/ = 0) {
    const int num_blocks = (total_tokens + TM - 1) / TM;
    const int64_t super_m_blocks =
        ((int64_t)num_blocks + N_TM - 1) / N_TM + num_experts;
    return sycl::nd_range<2>(
        sycl::range<2>(1, (size_t)(super_m_blocks * WG_SIZE)),
        sycl::range<2>(1, (size_t)WG_SIZE));
  }

  static int64_t get_num_wgs(int total_tokens, int num_experts) {
    const int num_blocks = (total_tokens + TM - 1) / TM;
    return (int64_t)((num_blocks + N_TM - 1) / N_TM) + num_experts;
  }

  __attribute__((always_inline)) int compute_tile_info(
      int wg_id,
      bool tile_active[N_TM],
      int tile_token_offset[N_TM],
      int tile_token_count[N_TM]) const {
    int expert_id = -1;
    int expert_block = 0;
    int expert_remainer = wg_id;
    for (int e_idx = 0; e_idx < this->num_experts; ++e_idx) {
      const int expert_num_tokens =
          (int)(this->expert_offset[e_idx + 1] - this->expert_offset[e_idx]);
      const int m_blocks_expert = (expert_num_tokens + TM - 1) / TM;
      const int super_blocks_expert = (m_blocks_expert + N_TM - 1) / N_TM;
      if (expert_remainer < super_blocks_expert) {
        expert_id = e_idx;
        expert_block = m_blocks_expert;
        break;
      }
      expert_remainer -= super_blocks_expert;
    }
    const int first_m_block = expert_remainer * N_TM;
    CUTE_UNROLL
    for (int mt = 0; mt < N_TM; ++mt) {
      const int m_block_idx = first_m_block + mt;
      tile_active[mt] = (expert_id >= 0) && (m_block_idx < expert_block);
      if (tile_active[mt]) {
        const int tok_start = (int)this->expert_offset[expert_id];
        const int tok_count = (int)(this->expert_offset[expert_id + 1] -
                                    this->expert_offset[expert_id]);
        tile_token_offset[mt] = tok_start + m_block_idx * TM;
        tile_token_count[mt] = sycl::min(TM, tok_count - m_block_idx * TM);
      } else {
        tile_token_offset[mt] = 0;
        tile_token_count[mt] = 0;
      }
    }
    return expert_id;
  }

  __attribute__((always_inline)) void load_hidden_states(
      int sg_id,
      int lane,
      bool const tile_active[N_TM],
      int const tile_token_offset[N_TM],
      int k_idx,
      DTYPE* slm_ptr) const {
    const bool is_partial = (k_idx + SG_COUNT * TK > this->H);

    CUTE_UNROLL
    for (int tm_idx = 0; tm_idx < N_TM; ++tm_idx) {
      const int slm_row_idx = tm_idx * TM + sg_id;
      const int topk_row_idx = tile_token_offset[tm_idx] + sg_id;
      const bool valid = tile_active[tm_idx] && (sg_id < TM) &&
                         (topk_row_idx < this->total_tokens);
      if (valid) {
        const int row_idx = (int)this->source_row[topk_row_idx];
        if (!is_partial) {
          auto slm_coord =
              make_identity_tensor(make_shape(Int<TK>{}, Int<SG_COUNT>{}));
          auto gA = make_tensor(
              make_gmem_ptr(this->tokens + (int64_t)row_idx * this->H + k_idx),
              make_layout(
                  make_shape(Int<TK>{}, Int<SG_COUNT>{}),
                  make_stride(_1{}, Int<TK>{})));
          auto load_copy = make_block_2d_copy(
              XE_LOAD_2D<cute::sizeof_bits_v<DTYPE>, SG_COUNT, TK>{}, gA);
          auto thr_lc = load_copy.get_slice(lane);
          auto tSrS = thr_lc.partition_fragment_D(slm_coord);
          const int next_tm_idx = tm_idx + 1;
          if (next_tm_idx < N_TM && tile_active[next_tm_idx]) {
            const int next_topk_row = tile_token_offset[next_tm_idx] + sg_id;
            if (next_topk_row < this->total_tokens) {
              const int next_row_idx = (int)this->source_row[next_topk_row];
              auto gA_next = make_tensor(
                  make_gmem_ptr(
                      this->tokens + (int64_t)next_row_idx * this->H + k_idx),
                  make_layout(
                      make_shape(Int<TK>{}, Int<SG_COUNT>{}),
                      make_stride(_1{}, Int<TK>{})));
              auto pref_copy = make_block_2d_prefetch(make_block_2d_copy(
                  XE_LOAD_2D<cute::sizeof_bits_v<DTYPE>, SG_COUNT, TK>{},
                  gA_next));
              prefetch(
                  pref_copy, pref_copy.get_slice(lane).partition_S(slm_coord));
            }
          }
          copy(load_copy, thr_lc.partition_S(slm_coord), tSrS);
          CUTE_UNROLL
          for (int j = 0; j < SG_COUNT; ++j)
            slm_ptr[slm_row_idx * SLM_STRIDE + j * TK + lane] =
                static_cast<DTYPE>(tSrS(j));
        } else {
          CUTE_UNROLL
          for (int j = 0; j < SG_COUNT * TK / SG_SIZE; ++j) {
            const int src_col = k_idx + j * SG_SIZE + lane;
            slm_ptr[slm_row_idx * SLM_STRIDE + j * SG_SIZE + lane] =
                (src_col < this->H)
                    ? this->tokens[(int64_t)row_idx * this->H + src_col]
                    : DTYPE(0);
          }
        }
      } else if (sg_id < TM) {
        CUTE_UNROLL
        for (int j = 0; j < SG_COUNT * TK / SG_SIZE; ++j)
          slm_ptr[slm_row_idx * SLM_STRIDE + j * SG_SIZE + lane] = DTYPE(0);
      }
    }
  }

  template <typename TiledMMA>
  __attribute__((always_inline)) void compute_gemm13_silu(
      sycl::nd_item<2> const& it,
      TiledMMA const& mma,
      int local_id,
      int sg_id,
      int lane,
      int n_idx,
      int expert_id,
      const DTYPE* w13_expert,
      bool const tile_active[N_TM],
      int const tile_token_offset[N_TM],
      DTYPE* slm_ptr,
      DTYPE* wg_act_buf) const {
    Tensor cW1 = make_tensor(
        make_gmem_ptr(w13_expert + n_idx),
        make_layout(
            make_shape(Int<SG_COUNT * TN>{}, this->H),
            make_stride(_1{}, (int)(2 * this->I))));
    Tensor cW3 = make_tensor(
        make_gmem_ptr(w13_expert + this->I + n_idx),
        make_layout(
            make_shape(Int<SG_COUNT * TN>{}, this->H),
            make_stride(_1{}, (int)(2 * this->I))));

    auto wg_tile = mma.tile_mnk();
    auto tCrGate = partition_fragment_C(mma, select<0, 1>(wg_tile));
    auto tCrUp = partition_fragment_C(mma, select<0, 1>(wg_tile));
    clear(tCrGate);
    clear(tCrUp);

    auto cBw_ktile = make_identity_tensor(select<1, 2>(wg_tile));
    auto cBw1_full =
        make_identity_tensor(make_shape(Int<SG_COUNT * TN>{}, this->H));
    auto cBw3_full =
        make_identity_tensor(make_shape(Int<SG_COUNT * TN>{}, this->H));
    auto gBw1_coord =
        local_tile(cBw1_full, select<1, 2>(wg_tile), make_coord(Int<0>{}, _));
    auto gBw3_coord =
        local_tile(cBw3_full, select<1, 2>(wg_tile), make_coord(Int<0>{}, _));
    auto copy_w1 = make_block_2d_copy_B(mma, cW1);
    auto copy_w3 = make_block_2d_copy_B(mma, cW3);
    auto prefetch_w1 = make_block_2d_prefetch(copy_w1);
    auto prefetch_w3 = make_block_2d_prefetch(copy_w3);
    auto thr_mma = mma.get_slice(local_id);
    auto thr_copy_w1 = copy_w1.get_slice(local_id);
    auto thr_copy_w3 = copy_w3.get_slice(local_id);
    auto tCrW1 = thr_mma.partition_sg_fragment_B(cBw_ktile);
    auto tCrW3 = thr_mma.partition_sg_fragment_B(cBw_ktile);
    auto tBrBw1 = thr_copy_w1.partition_sg_fragment_D(cBw_ktile);
    auto tBrBw3 = thr_copy_w3.partition_sg_fragment_D(cBw_ktile);
    auto tBgBw1 = thr_copy_w1.partition_S(gBw1_coord);
    auto tBgBw3 = thr_copy_w3.partition_S(gBw3_coord);
    auto pBgBw1 = prefetch_w1.get_slice(local_id).partition_S(gBw1_coord);
    auto pBgBw3 = prefetch_w3.get_slice(local_id).partition_S(gBw3_coord);

    auto copy_slm = make_block_2d_slm_copy_A(mma);
    auto thr_copy_slm = copy_slm.get_slice(local_id);

    auto cSlm_full = make_tensor(
        make_smem_ptr(slm_ptr),
        make_shape(Int<N_TM * TM>{}, Int<SG_COUNT * TK>{}),
        make_stride(Int<SG_COUNT * TK>{}, _1{}));
    Tensor gSlm =
        local_tile(cSlm_full, select<0, 2>(wg_tile), make_coord(0, _));

    constexpr int K_STEPS = SG_COUNT / N_TK;
    const int k1_tile_count = this->H / (SG_COUNT * TK);
    const int k_tiles_total = k1_tile_count * K_STEPS;

    for (int k1_tile_idx = 0; k1_tile_idx < k1_tile_count; ++k1_tile_idx) {
      const int col_base = k1_tile_idx * (SG_COUNT * TK);
      load_hidden_states(
          sg_id, lane, tile_active, tile_token_offset, col_base, slm_ptr);
      it.barrier(sycl::access::fence_space::local_space);
      CUTE_UNROLL
      for (int k_step = 0; k_step < K_STEPS; ++k_step) {
        const int k_step_idx = k1_tile_idx * K_STEPS + k_step;
        auto gSlm_k = gSlm(_, _, k_step);
        auto tCrSlm = thr_mma.partition_sg_fragment_A(gSlm_k);
        auto tCrSlm_in = thr_copy_slm.retile_D(tCrSlm);
        auto tAsSlm_in = thr_copy_slm.partition_S(gSlm_k);
        copy(copy_slm, tAsSlm_in, tCrSlm_in);
        if (k_step_idx + 1 < k_tiles_total) {
          prefetch(prefetch_w1, pBgBw1(_, _, _, k_step_idx + 1));
          prefetch(prefetch_w3, pBgBw3(_, _, _, k_step_idx + 1));
        }
        copy(copy_w1, tBgBw1(_, _, _, k_step_idx), tBrBw1);
        reorder(tBrBw1, tCrW1);
        cute::gemm(mma, tCrSlm, tCrW1, tCrGate);
        copy(copy_w3, tBgBw3(_, _, _, k_step_idx), tBrBw3);
        reorder(tBrBw3, tCrW3);
        cute::gemm(mma, tCrSlm, tCrW3, tCrUp);
      }
      it.barrier(sycl::access::fence_space::local_space);
    }

    {
      const int col_abs = n_idx + sg_id * TN + lane;
      float gate_bias_val = 0.0f, up_bias_val = 0.0f;
      if constexpr (HAS_W13_BIAS) {
        gate_bias_val =
            this->w13_bias[(int64_t)expert_id * (2 * this->I) + col_abs];
        up_bias_val =
            this->w13_bias
                [(int64_t)expert_id * (2 * this->I) + this->I + col_abs];
      }
      auto coord_atom = make_identity_tensor(make_shape(Int<TM>{}, Int<TN>{}));
      CUTE_UNROLL
      for (int m_tile = 0; m_tile < N_TM; ++m_tile) {
        Tensor gActStore = make_tensor(
            make_gmem_ptr(
                wg_act_buf + (int64_t)m_tile * TM * this->I + n_idx +
                sg_id * TN),
            make_layout(
                make_shape(Int<TM>{}, Int<TN>{}),
                make_stride((int)this->I, _1{})));
        auto copy_act_store = make_block_2d_copy(
            XE_STORE_2D<cute::sizeof_bits_v<DTYPE>, TM, TN>{}, gActStore);
        auto thr_cas = copy_act_store.get_slice(lane);
        auto tSrAct = thr_cas.partition_fragment_S(coord_atom);
        CUTE_UNROLL
        for (int r = 0; r < TM; ++r) {
          float g = float(tCrGate(m_tile * TM + r)) + gate_bias_val;
          float u = float(tCrUp(m_tile * TM + r)) + up_bias_val;
          if constexpr (HAS_CLAMP_LIMIT) {
            g = sycl::fmin(g, this->gemm1_clamp_limit);
            u = sycl::fmax(
                sycl::fmin(u, this->gemm1_clamp_limit),
                -this->gemm1_clamp_limit);
          }
          tSrAct(r) = DTYPE(silu_apply(g) * u);
        }
        copy(copy_act_store, tSrAct, thr_cas.partition_D(coord_atom));
      }
    }
  }

  template <typename TiledMMA>
  __attribute__((always_inline)) void compute_gemm2(
      sycl::nd_item<2> const& it,
      TiledMMA const& mma,
      int local_id,
      int sg_id,
      int lane,
      int expert_id,
      bool const tile_active[N_TM],
      int const tile_token_offset[N_TM],
      int const tile_token_count[N_TM],
      const DTYPE* w2_expert,
      DTYPE* wg_act_buf) const {
    auto wg_tile = mma.tile_mnk();
    auto thr_mma = mma.get_slice(local_id);

    constexpr int WG_N_TILE = N_TN * SG_COUNT * TN;
    constexpr int K_STEPS = SG_COUNT / N_TK;

    auto cActStatic = make_identity_tensor(
        make_shape(Int<N_TM * TM>{}, Int<SG_COUNT * TK>{}));
    auto gActStatic =
        local_tile(cActStatic, select<0, 2>(wg_tile), make_coord(0, _));

    auto cBw2_ktile = make_identity_tensor(select<1, 2>(wg_tile));
    auto cBw2_full = make_identity_tensor(
        make_shape(Int<WG_N_TILE>{}, Int<SG_COUNT * TK>{}));
    auto gBw2_coord =
        local_tile(cBw2_full, select<1, 2>(wg_tile), make_coord(Int<0>{}, _));

    for (int h_tile_idx = 0; h_tile_idx < this->H; h_tile_idx += WG_N_TILE) {
      auto tCrDown = partition_fragment_C(mma, select<0, 1>(wg_tile));
      clear(tCrDown);

      for (int n_idx = 0; n_idx < this->I; n_idx += SG_COUNT * TK) {
        Tensor cW2 = make_tensor(
            make_gmem_ptr(w2_expert + h_tile_idx + (int64_t)n_idx * this->H),
            make_layout(
                make_shape(Int<WG_N_TILE>{}, Int<SG_COUNT * TK>{}),
                make_stride(_1{}, (int)this->H)));
        auto copy_w2 = make_block_2d_copy_B(mma, cW2);
        auto thr_copy_w2 = copy_w2.get_slice(local_id);
        auto tCrW2 = thr_mma.partition_sg_fragment_B(cBw2_ktile);
        auto tBrBw2 = thr_copy_w2.partition_sg_fragment_D(cBw2_ktile);
        auto tBgBw2 = thr_copy_w2.partition_S(gBw2_coord);

        Tensor cActTile = make_tensor(
            make_gmem_ptr(wg_act_buf + n_idx),
            make_layout(
                make_shape(Int<N_TM * TM>{}, Int<SG_COUNT * TK>{}),
                make_stride((int)this->I, _1{})));
        auto copy_act = make_block_2d_copy_A(mma, cActTile);
        auto thr_copy_act = copy_act.get_slice(local_id);

        const int n_next = n_idx + SG_COUNT * TK;
        if (n_next < this->I) {
          Tensor cActNext = make_tensor(
              make_gmem_ptr(wg_act_buf + n_next),
              make_layout(
                  make_shape(Int<N_TM * TM>{}, Int<SG_COUNT * TK>{}),
                  make_stride((int)this->I, _1{})));
          auto pref_act =
              make_block_2d_prefetch(make_block_2d_copy_A(mma, cActNext));
          auto pAg = pref_act.get_slice(local_id).partition_S(gActStatic);
          CUTE_UNROLL
          for (int k_pf = 0; k_pf < K_STEPS; ++k_pf)
            prefetch(pref_act, pAg(_, _, _, k_pf));
        }

        if (n_next < this->I) {
          Tensor cW2_next = make_tensor(
              make_gmem_ptr(w2_expert + h_tile_idx + (int64_t)n_next * this->H),
              make_layout(
                  make_shape(Int<WG_N_TILE>{}, Int<SG_COUNT * TK>{}),
                  make_stride(_1{}, (int)this->H)));
          auto pref_w2 =
              make_block_2d_prefetch(make_block_2d_copy_B(mma, cW2_next));
          auto pW2g = pref_w2.get_slice(local_id).partition_S(gBw2_coord);
          CUTE_UNROLL
          for (int k_pf = 0; k_pf < K_STEPS; ++k_pf)
            prefetch(pref_w2, pW2g(_, _, _, k_pf));
        }

        CUTE_UNROLL
        for (int k_step = 0; k_step < K_STEPS; ++k_step) {
          auto wgActS = gActStatic(_, _, k_step);
          auto tCrAct = thr_mma.partition_sg_fragment_A(wgActS);
          auto tArAct = thr_copy_act.partition_sg_fragment_D(wgActS);
          auto tAgAct = thr_copy_act.partition_S(wgActS);
          copy(copy_act, tAgAct, tArAct);
          reorder(tArAct, tCrAct);
          copy(copy_w2, tBgBw2(_, _, _, k_step), tBrBw2);
          reorder(tBrBw2, tCrW2);
          cute::gemm(mma, tCrAct, tCrW2, tCrDown);
        }
      }

      {
        auto cC_id = make_identity_tensor(select<0, 1>(wg_tile));
        auto tCcC = thr_mma.partition_C(cC_id);
        store_output(
            tCrDown,
            tCcC,
            sg_id,
            lane,
            h_tile_idx,
            expert_id,
            /*n_idx=*/0,
            tile_active,
            tile_token_offset,
            tile_token_count);
      }
    }
  }

  template <typename TAccum, typename TCoord>
  __attribute__((always_inline)) void store_output(
      TAccum const& tCrDown,
      TCoord const& tCcC,
      int sg_id,
      int lane,
      int h_tile_idx,
      int expert_id,
      int /*n_idx*/,
      bool const tile_active[N_TM],
      int const tile_token_offset[N_TM],
      int const tile_token_count[N_TM]) const {
    int slots[N_TM][TM];
    float weights[N_TM][TM];
    CUTE_UNROLL
    for (int mt = 0; mt < N_TM; ++mt) {
      CUTE_UNROLL
      for (int m = 0; m < TM; ++m) {
        if (tile_active[mt] && m < tile_token_count[mt]) {
          const int idx = tile_token_offset[mt] + m;
          slots[mt][m] = this->topk_ids[idx];
          weights[mt][m] = this->topk_weights[idx];
        } else {
          slots[mt][m] = 0;
          weights[mt][m] = 0.0f;
        }
      }
    }

    auto coord_row = make_identity_tensor(make_shape(_1{}, Int<TN>{}));
    CUTE_UNROLL
    for (int i = 0; i < size(tCrDown); ++i) {
      const int row = cute::get<0>(tCcC(i));
      const int col = cute::get<1>(tCcC(i));
      const int mt = row / TM;
      const int m = row % TM;
      if (!tile_active[mt]) continue;
      if (m >= tile_token_count[mt]) continue;

      const int col_base = h_tile_idx + col - lane;
      const int slot_id = slots[mt][m];
      const float weight = weights[mt][m];

      Tensor gInter = make_tensor(
          make_gmem_ptr(
              this->intermediate + (int64_t)slot_id * this->H + col_base),
          make_layout(
              make_shape(_1{}, Int<TN>{}), make_stride((int)this->H, _1{})));
      auto copy_inter = make_block_2d_copy(
          XE_STORE_2D<cute::sizeof_bits_v<INTER_DTYPE>, 1, TN>{}, gInter);
      auto thr_cinter = copy_inter.get_slice(lane);
      auto tSrInter = thr_cinter.partition_fragment_S(coord_row);

      float val = float(tCrDown(i)) * weight;
      if constexpr (HAS_W2_BIAS) {
        const int abs_col = col_base + lane;
        val += this->w2_bias[(int64_t)expert_id * this->H + abs_col] * weight;
      }
      tSrInter(0) = INTER_DTYPE(val);
      copy(copy_inter, tSrInter, thr_cinter.partition_D(coord_row));
    }
  }

  __attribute__((always_inline)) void reduce_output(
      sycl::nd_item<2> const& it,
      int local_id,
      int sg_id,
      int lane,
      bool const tile_active[N_TM],
      int const tile_token_offset[N_TM],
      int const tile_token_count[N_TM]) const {
    int32_t* slm_count = reinterpret_cast<int32_t*>(
        this->slm.template get_multi_ptr<sycl::access::decorated::no>().get());
    CUTE_UNROLL
    for (int mt = 0; mt < N_TM; ++mt) {
      if (!tile_active[mt]) continue;
      for (int m = 0; m < tile_token_count[mt]; ++m) {
        int count = 0;
        if (local_id == 0) {
          sycl::atomic_ref<
              int32_t,
              sycl::memory_order_acq_rel,
              sycl::memory_scope_device,
              sycl::access::address_space::global_space>
              ctr(this->row_counter
                      [(int)this->source_row[tile_token_offset[mt] + m]]);
          slm_count[0] = ctr.fetch_add(1);
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (slm_count[0] != this->K - 1) continue;
        const int out_row = (int)this->source_row[tile_token_offset[mt] + m];
        this->reduce_one_row(out_row, sg_id, lane, this->K);
      }
    }
  }

  void operator()(sycl::nd_item<2> it) const {
    const int wg_id = (int)it.get_group(1);

    auto sg = it.get_sub_group();
    const int sg_id = (int)sg.get_group_id()[0];
    const int lane = (int)sg.get_local_id()[0];
    const int local_id = sg_id * SG_SIZE + lane;

    bool tile_active[N_TM];
    int tile_token_offset[N_TM], tile_token_count[N_TM];
    int expert_id = compute_tile_info(
        wg_id, tile_active, tile_token_offset, tile_token_count);
    if (expert_id < 0) return;

    DTYPE* slm_ptr =
        this->slm.template get_multi_ptr<sycl::access::decorated::no>().get();
    DTYPE* wg_act_buf = act_buf + (int64_t)wg_id * N_TM * TM * this->I;
    const DTYPE* w13_expert =
        this->w13 + (int64_t)expert_id * this->H * (2 * this->I);
    const DTYPE* w2_expert = this->w2 + (int64_t)expert_id * this->I * this->H;

    auto gate_up_mma = Base::make_gate_up_tiled_mma();
    auto down_mma = Base::make_down_tiled_mma();

    for (int n_idx = 0; n_idx < this->I; n_idx += SG_COUNT * TN) {
      compute_gemm13_silu(
          it,
          gate_up_mma,
          local_id,
          sg_id,
          lane,
          n_idx,
          expert_id,
          w13_expert,
          tile_active,
          tile_token_offset,
          slm_ptr,
          wg_act_buf);
    }
    it.barrier(sycl::access::fence_space::global_space);

    compute_gemm2(
        it,
        down_mma,
        local_id,
        sg_id,
        lane,
        expert_id,
        tile_active,
        tile_token_offset,
        tile_token_count,
        w2_expert,
        wg_act_buf);
    it.barrier(sycl::access::fence_space::global_space);

    reduce_output(
        it,
        local_id,
        sg_id,
        lane,
        tile_active,
        tile_token_offset,
        tile_token_count);
  }
};  // struct FusedMoePrefill
