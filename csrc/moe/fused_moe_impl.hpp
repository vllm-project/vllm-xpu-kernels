#pragma once
#include "fused_moe_common.hpp"
#include "fused_moe_decode.hpp"
#include "fused_moe_prefill.hpp"

// Backward-compatible FusedMoe<..., IS_DECODE, ...> alias.
// Selects FusedMoeDecode (IS_DECODE=true) or FusedMoePrefill (IS_DECODE=false)
// so that fused_moe.cpp can instantiate FusedMoe<...> unchanged.
// The TM/N_TM args are forwarded to FusedMoePrefill only; FusedMoeDecode
// hardcodes them to 1.
template <
    bool HAS_W13_BIAS,
    bool HAS_W2_BIAS,
    bool HAS_CLAMP_LIMIT = false,
    bool IS_DECODE = false,
    typename DTYPE = bf16_t,
    typename INTER_DTYPE = float,
    int TM = (IS_DECODE ? 1 : ::TM),
    int N_TM = (IS_DECODE ? 1 : ::N_TM),
    int N_TK = ::N_TK,
    int N_TN = ::N_TN>
using FusedMoe = std::conditional_t<
    IS_DECODE,
    FusedMoeDecode<
        HAS_W13_BIAS,
        HAS_W2_BIAS,
        HAS_CLAMP_LIMIT,
        DTYPE,
        INTER_DTYPE,
        N_TK,
        N_TN>,
    FusedMoePrefill<
        HAS_W13_BIAS,
        HAS_W2_BIAS,
        HAS_CLAMP_LIMIT,
        DTYPE,
        INTER_DTYPE,
        TM,
        N_TM,
        N_TK,
        N_TN>>;
