#pragma once
#include "fused_moe_common.hpp"

template <
    bool HAS_W13_BIAS,
    bool HAS_W2_BIAS,
    bool HAS_CLAMP_LIMIT = false,
    typename DTYPE = bf16_t,
    typename INTER_DTYPE = float,
    int N_TK = ::N_TK,
    int N_TN = ::N_TN>
struct FusedMoeDecode : FusedMoeCommon<
                            HAS_W13_BIAS,
                            HAS_W2_BIAS,
                            HAS_CLAMP_LIMIT,
                            /*IS_DECODE=*/true,
                            DTYPE,
                            INTER_DTYPE,
                            /*TM=*/1,
                            /*N_TM=*/1,
                            N_TK,
                            N_TN> {
  using Base = FusedMoeCommon<
      HAS_W13_BIAS,
      HAS_W2_BIAS,
      HAS_CLAMP_LIMIT,
      true,
      DTYPE,
      INTER_DTYPE,
      1,
      1,
      N_TK,
      N_TN>;

  static sycl::nd_range<2>
  get_nd_range(int total_tokens, int /*num_experts*/, int K) {
    return sycl::nd_range<2>(
        sycl::range<2>((size_t)total_tokens, (size_t)(K * WG_SIZE)),
        sycl::range<2>(1, (size_t)WG_SIZE));
  }

  __attribute__((always_inline)) int compute_tile_info(
      int token_id,
      int slot_local_id,
      bool tile_active[1],
      int tile_token_offset[1],
      int tile_token_count[1]) const {
    const int wg_id = token_id * this->K + slot_local_id;
    const int expert_id = (int)this->topk_ids[wg_id];
    tile_active[0] = true;
    tile_token_offset[0] = wg_id;
    tile_token_count[0] = 1;
    return expert_id;
  }

  template <typename TiledMMA>
  __attribute__((always_inline)) void compute_gemm2(
      sycl::nd_item<2> const& it,
      TiledMMA const& mma,
      int local_id,
      int expert_id,
      bool const tile_active[1],
      int const tile_token_offset[1],
      int const tile_token_count[1],
      const DTYPE* w2_expert,
      DTYPE* act_slm_ptr,
      int n_idx) const {
    auto wg_tile = mma.tile_mnk();
    const int sg_id = local_id / SG_SIZE;
    const int lane = local_id % SG_SIZE;
    auto thr_mma = mma.get_slice(local_id);

    auto copy_slm = make_block_2d_slm_copy_A(mma);
    auto thr_s2r_A = copy_slm.get_slice(local_id);
    auto cSlm = make_tensor(
        make_smem_ptr(act_slm_ptr),
        make_shape(Int<N_TM * TM>{}, Int<SG_COUNT * TK>{}),
        make_stride(Int<SG_COUNT * TK>{}, _1{}));
    Tensor gSlm = local_tile(cSlm, select<0, 2>(wg_tile), make_coord(0, _));

    constexpr int WG_N_TILE = N_TN * SG_COUNT * TN;
    constexpr int K_STEPS = SG_COUNT / N_TK;

    auto cBw2_ktile = make_identity_tensor(select<1, 2>(wg_tile));
    auto cBw2_full = make_identity_tensor(
        make_shape(Int<WG_N_TILE>{}, Int<SG_COUNT * TK>{}));
    auto gBw2_coord =
        local_tile(cBw2_full, select<1, 2>(wg_tile), make_coord(Int<0>{}, _));

    for (int h_tile_idx = 0; h_tile_idx < this->H; h_tile_idx += WG_N_TILE) {
      Tensor cW2 = make_tensor(
          make_gmem_ptr(w2_expert + h_tile_idx),
          make_layout(
              make_shape(Int<WG_N_TILE>{}, Int<SG_COUNT * TK>{}),
              make_stride(_1{}, (int)this->H)));
      auto copy_w2 = make_block_2d_copy_B(mma, cW2);
      auto thr_copy_w2 = copy_w2.get_slice(local_id);
      auto tCrW2 = thr_mma.partition_sg_fragment_B(cBw2_ktile);
      auto tBrBw2 = thr_copy_w2.partition_sg_fragment_D(cBw2_ktile);
      auto tBgBw2 = thr_copy_w2.partition_S(gBw2_coord);

      const int h_next = h_tile_idx + WG_N_TILE;
      if (h_next < this->H) {
        Tensor cW2_next = make_tensor(
            make_gmem_ptr(w2_expert + h_next),
            make_layout(
                make_shape(Int<WG_N_TILE>{}, Int<SG_COUNT * TK>{}),
                make_stride(_1{}, (int)this->H)));
        auto pref_w2 =
            make_block_2d_prefetch(make_block_2d_copy_B(mma, cW2_next));
        auto pW2gW2 = pref_w2.get_slice(local_id).partition_S(gBw2_coord);
        CUTE_UNROLL
        for (int k_pf = 0; k_pf < K_STEPS; ++k_pf)
          prefetch(pref_w2, pW2gW2(_, _, _, k_pf));
      }

      auto tCrDown = partition_fragment_C(mma, select<0, 1>(wg_tile));
      clear(tCrDown);

      CUTE_UNROLL
      for (int k_step = 0; k_step < K_STEPS; ++k_step) {
        auto gSlm_k = gSlm(_, _, k_step);
        auto tCrSlm = thr_mma.partition_sg_fragment_A(gSlm_k);
        auto tCrSlm_in = thr_s2r_A.retile_D(tCrSlm);
        auto tAsSlm_in = thr_s2r_A.partition_S(gSlm_k);
        copy(copy_slm, tAsSlm_in, tCrSlm_in);
        copy(copy_w2, tBgBw2(_, _, _, k_step), tBrBw2);
        reorder(tBrBw2, tCrW2);
        cute::gemm(mma, tCrSlm, tCrW2, tCrDown);
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
            n_idx,
            tile_active,
            tile_token_offset,
            tile_token_count);
      }
    }

    it.barrier(sycl::access::fence_space::local_space);
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
      bool const tile_active[1],
      int const tile_token_offset[1],
      DTYPE* slm_ptr) const {
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

    const int out_row = tile_token_offset[0] / this->K;
    const DTYPE* row_ptr = this->tokens + (int64_t)out_row * this->H;
    auto cA = make_tensor(
        make_gmem_ptr(row_ptr),
        make_layout(make_shape(_1{}, this->H), make_stride(this->H, _1{})));
    auto cA_full = make_identity_tensor(make_shape(_1{}, this->H));
    auto gA =
        local_tile(cA_full, select<0, 2>(mma.tile_mnk()), make_coord(_, _));
    auto copy_A = make_block_2d_copy_A(mma, cA);
    auto thr_copy_A = copy_A.get_slice(local_id);
    auto tArA = thr_copy_A.partition_sg_fragment_D(gA(_, _, _, 0));
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0, 0));
    auto tAgA = thr_copy_A.partition_S(gA);

    const int k_tile_count = this->H / (N_TK * TK);
    constexpr int DECODE_PREFETCH_COUNT = 4;
    CUTE_UNROLL
    for (int k_pref_tile_idx = 0; k_pref_tile_idx < DECODE_PREFETCH_COUNT;
         ++k_pref_tile_idx) {
      prefetch(
          make_block_2d_prefetch(copy_A),
          make_block_2d_prefetch(copy_A).get_slice(local_id).partition_S(gA)(
              _, _, _, _, k_pref_tile_idx));
      prefetch(prefetch_w1, pBgBw1(_, _, _, k_pref_tile_idx));
      prefetch(prefetch_w3, pBgBw3(_, _, _, k_pref_tile_idx));
    }
    CUTE_UNROLL
    for (int k_tile_idx = 0; k_tile_idx < k_tile_count; ++k_tile_idx) {
      const int k_pref_tile_idx = k_tile_idx + DECODE_PREFETCH_COUNT;
      if (k_pref_tile_idx < k_tile_count) {
        prefetch(
            make_block_2d_prefetch(copy_A),
            make_block_2d_prefetch(copy_A).get_slice(local_id).partition_S(gA)(
                _, _, _, _, k_pref_tile_idx));
        prefetch(prefetch_w1, pBgBw1(_, _, _, k_pref_tile_idx));
        prefetch(prefetch_w3, pBgBw3(_, _, _, k_pref_tile_idx));
      }
      copy(copy_A, tAgA(_, _, _, _, k_tile_idx), tArA);
      reorder(tArA, tCrA);
      copy(copy_w1, tBgBw1(_, _, _, k_tile_idx), tBrBw1);
      reorder(tBrBw1, tCrW1);
      cute::gemm(mma, tCrA, tCrW1, tCrGate);
      copy(copy_w3, tBgBw3(_, _, _, k_tile_idx), tBrBw3);
      reorder(tBrBw3, tCrW3);
      cute::gemm(mma, tCrA, tCrW3, tCrUp);
    }
    {
      auto cC_full =
          make_identity_tensor(make_shape(_1{}, Int<SG_COUNT * TN>{}));
      auto tCcC = thr_mma.partition_C(cC_full);
      CUTE_UNROLL
      for (int i = 0; i < size(tCrGate); ++i) {
        const int n = cute::get<1>(tCcC(i));
        float g = float(tCrGate(i));
        float u = float(tCrUp(i));
        if constexpr (HAS_W13_BIAS) {
          g += this->w13_bias[(int64_t)expert_id * (2 * this->I) + n_idx + n];
          u += this->w13_bias
                   [(int64_t)expert_id * (2 * this->I) + this->I + n_idx + n];
        }
        if constexpr (HAS_CLAMP_LIMIT) {
          g = sycl::fmin(g, this->gemm1_clamp_limit);
          u = sycl::fmax(
              sycl::fmin(u, this->gemm1_clamp_limit), -this->gemm1_clamp_limit);
        }
        slm_ptr[n] = DTYPE(silu_apply(g) * u);
      }
    }
    it.barrier(sycl::access::fence_space::local_space);
  }

  template <typename TAccum, typename TCoord>
  __attribute__((always_inline)) void store_output(
      TAccum const& tCrDown,
      TCoord const& /*tCcC*/,
      int sg_id,
      int lane,
      int h_tile_idx,
      int expert_id,
      int n_idx,
      bool const tile_active[N_TM],
      int const tile_token_offset[N_TM],
      int const tile_token_count[N_TM]) const {
    if (!tile_active[0] || tile_token_count[0] == 0) return;

    const int strips = this->I / (SG_COUNT * TK);
    const int strip_idx = n_idx / (SG_COUNT * TK);
    const int slot_id = tile_token_offset[0];
    const float weight = this->topk_weights[slot_id];
    const int64_t row_base = ((int64_t)slot_id * strips + strip_idx) * this->H;

    auto coord = make_identity_tensor(make_shape(_1{}, Int<TN>{}));

    CUTE_UNROLL
    for (int idx = 0; idx < N_TN; ++idx) {
      float val = float(tCrDown(idx)) * weight;
      if constexpr (HAS_W2_BIAS) {
        if (strip_idx == 0) {
          const int abs_col = h_tile_idx + sg_id * N_TN * TN + idx * TN + lane;
          val += this->w2_bias[(int64_t)expert_id * this->H + abs_col] * weight;
        }
      }
      auto gInter = make_tensor(
          make_gmem_ptr(
              this->intermediate + row_base + h_tile_idx + sg_id * N_TN * TN +
              idx * TN),
          make_layout(make_shape(_1{}, Int<TN>{}), make_stride(this->H, _1{})));
      auto store_copy = make_block_2d_copy(
          XE_STORE_2D<cute::sizeof_bits_v<INTER_DTYPE>, 1, TN>{}, gInter);
      auto thr_store_copy = store_copy.get_slice(lane);
      auto frag = thr_store_copy.partition_fragment_S(coord);
      frag(0) = INTER_DTYPE(val);
      copy(store_copy, frag, thr_store_copy.partition_D(coord));
    }
  }

  __attribute__((always_inline)) void reduce_output(
      sycl::nd_item<2> const& it,
      int local_id,
      int sg_id,
      int lane,
      bool const tile_active[1],
      int const tile_token_offset[1],
      int const tile_token_count[1]) const {
    int32_t* slm_count = reinterpret_cast<int32_t*>(
        this->slm.template get_multi_ptr<sycl::access::decorated::no>().get());
    const int out_row = tile_token_offset[0] / this->K;
    int count = 0;
    if (local_id == 0) {
      sycl::atomic_ref<
          int32_t,
          sycl::memory_order_acq_rel,
          sycl::memory_scope_device,
          sycl::access::address_space::global_space>
          ctr(this->row_counter[out_row]);
      slm_count[0] = ctr.fetch_add(1);
    }
    it.barrier(sycl::access::fence_space::local_space);
    if (slm_count[0] != this->K - 1) return;
    const int k_values = this->K * (this->I / (SG_COUNT * TK));
    this->reduce_one_row(out_row, sg_id, lane, k_values);
  }

  void operator()(sycl::nd_item<2> it) const {
    const int token_id = (int)it.get_group(0);
    const int slot_local_id = (int)it.get_group(1);
    if (slot_local_id >= this->K) return;

    auto sg = it.get_sub_group();
    const int sg_id = (int)sg.get_group_id()[0];
    const int lane = (int)sg.get_local_id()[0];
    const int local_id = sg_id * SG_SIZE + lane;

    bool tile_active[1];
    int tile_token_offset[1], tile_token_count[1];
    int expert_id = compute_tile_info(
        token_id,
        slot_local_id,
        tile_active,
        tile_token_offset,
        tile_token_count);
    if (expert_id < 0) return;

    DTYPE* slm_ptr =
        this->slm.template get_multi_ptr<sycl::access::decorated::no>().get();
    const DTYPE* w13_expert =
        this->w13 + (int64_t)expert_id * this->H * (2 * this->I);
    const DTYPE* w2_expert = this->w2 + (int64_t)expert_id * this->I * this->H;

    auto gate_up_mma = Base::make_gate_up_tiled_mma();
    auto down_mma = Base::make_down_tiled_mma();

    for (int n_idx = 0; n_idx < this->I; n_idx += SG_COUNT * TK) {
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
          slm_ptr);
      compute_gemm2(
          it,
          down_mma,
          local_id,
          expert_id,
          tile_active,
          tile_token_offset,
          tile_token_count,
          w2_expert + (int64_t)n_idx * this->H,
          slm_ptr,
          n_idx);
    }
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
};  // struct FusedMoeDecode
