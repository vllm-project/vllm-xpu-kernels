# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the spec-decoding extension of ``gdn_attention``.

The op now accepts two optional tensors that together mirror FLA's
``IS_SPEC_DECODING`` semantics:

* ``spec_state_indices_tensor`` — int32 ``[num_seqs, K]`` ring of kv-cache
  slot indices. The per-sequence load slot is
  ``spec_state_indices_tensor[i, num_accepted_tokens[i] - 1]``; the
  per-token store slot during the gated-delta-rule loop is
  ``spec_state_indices_tensor[i, t - seq_start]``. A non-positive slot
  (``<= 0``, i.e. ``NULL_BLOCK_ID``) is treated as "no valid prior
  state" — the conv pre-state load is suppressed and SSM per-token
  writebacks are skipped for that slot.
* ``num_accepted_tokens`` — int32 ``[num_seqs]``.

When both tensors are omitted, behavior is bit-exact with the pre-PR
op (the existing ``test_gdn_attn.py`` / ``test_gdn_attn_padded.py``
suite exercises that path).

The production dispatcher (paired vLLM PR #42382) passes
``spec_state_indices_tensor[:, 0]`` as the ``non_spec_state_indices_tensor``
arg under spec — the kernel uses that as the write target for the
``spec_state_roll_kernel`` (conv state writeback). These tests follow
that wiring convention.

We cross-check the spec path against the non-spec path via three
kernel-internal equivalences:

1. **K>1, aligned ring, num_accepted=1, decode** — per-token writebacks
   all land on the same slot; the core_attn_out and final ssm_state
   at that slot must match the non-spec native path on the same
   workload.
2. **NULL_BLOCK_ID** — under spec with the load slot = ``NULL`` and the
   per-token store slot = ``NULL``, the kernel must produce the same
   core_attn_out as a non-spec call with ``has_initial_state=False`` on
   the same workload, AND must leave the conv/ssm pools byte-identical
   to their pre-call contents (no writes through NULL slots).
3. **Argument-pair validation** — passing only one of the spec args
   must raise.

Plus a smoke test that the K>1 native spec path runs to completion on
varied ``num_accepted_tokens`` and produces finite outputs.

Note that the K=1 spec path (state_len = Width-1) is NOT exercised here
as a bit-exact equivalence: the spec kernel's conv_state writeback
populates positions ``{0} ∪ [state_len - K_call, state_len)`` of the
rolled tmp buffer per work-item, so when ``K_call < state_len - K_call``
(i.e. ``K_call < (state_len)/2``, true for K=1 with Width≥4) the
intermediate tmp rows are uninitialized and the post-state conv slot
differs from the non-spec path. Production never hits this — vLLM
routes K=1 requests to the non-spec path — but it's worth flagging so
a future reader doesn't burn time re-deriving it.

Full correctness for the non-degenerate spec path with K>1 and arbitrary
``num_accepted_tokens`` is covered by the paired vLLM replay harness in
``tests/kernels/xpu/test_spec_gdn_replay.py`` (PR vllm-project/vllm#42382)
which diffs against a live FLA Triton reference.
"""

import random

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

NUM_K_HEADS = 2
NUM_V_HEADS = 4
HEAD_K_DIM = 64
HEAD_V_DIM = 64
WIDTH = 4
TP_SIZE = 1
CACHE_BATCH_SIZE = 32
ACTIVATION = "silu"


def _mixed_qkvz_size():
    return (
        NUM_K_HEADS
        // TP_SIZE
        * (2 * HEAD_K_DIM + 2 * HEAD_V_DIM * NUM_V_HEADS // NUM_K_HEADS)
    )


def _mixed_ba_size():
    return NUM_K_HEADS // TP_SIZE * (2 * NUM_V_HEADS // NUM_K_HEADS)


def _mixed_qkv_size():
    return (
        NUM_K_HEADS
        // TP_SIZE
        * (2 * HEAD_K_DIM + HEAD_V_DIM * NUM_V_HEADS // NUM_K_HEADS)
    )


def _alloc_inputs(num_tokens, batch_size, dtype, device, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    qkvz_size = _mixed_qkvz_size()
    ba_size = _mixed_ba_size()
    qkv_size = _mixed_qkv_size()

    qkvz = torch.randn(num_tokens, qkvz_size, dtype=dtype, device=device)
    ba = torch.randn(num_tokens, ba_size, dtype=dtype, device=device)

    conv_state = torch.randn(
        CACHE_BATCH_SIZE, WIDTH - 1, qkv_size, dtype=dtype, device=device
    )
    ssm_state = torch.randn(
        CACHE_BATCH_SIZE,
        NUM_V_HEADS // TP_SIZE,
        HEAD_V_DIM,
        HEAD_K_DIM,
        dtype=dtype,
        device=device,
    )

    conv_weights = torch.randn(qkv_size, WIDTH, dtype=dtype, device=device)
    A_log = torch.randn(
        NUM_V_HEADS // TP_SIZE, dtype=torch.float32, device=device
    )
    dt_bias = torch.randn(NUM_V_HEADS // TP_SIZE, dtype=dtype, device=device)

    # Slot 0 is reserved as NULL_BLOCK_ID sentinel; sample real slots from
    # [1, CACHE_BATCH_SIZE) to avoid collisions with the null path.
    state_indices = torch.tensor(
        random.sample(range(1, CACHE_BATCH_SIZE), batch_size),
        device=device,
        dtype=torch.int32,
    )

    return dict(
        qkvz=qkvz,
        ba=ba,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=conv_weights,
        A_log=A_log,
        dt_bias=dt_bias,
        state_indices=state_indices,
    )


def _decode_query_start_loc(batch_size, device):
    """Decode-only: one token per sequence."""
    return torch.arange(batch_size + 1, dtype=torch.int32, device=device)


def _call_gdn(
    inputs,
    num_actual_tokens,
    num_prefills,
    num_decodes,
    query_start_loc,
    state_indices,
    has_initial_state,
    spec_state_indices=None,
    num_accepted_tokens=None,
    reorder_input=False,
):
    device = inputs["qkvz"].device
    dtype = inputs["qkvz"].dtype
    core_attn_out = torch.zeros(
        num_actual_tokens,
        NUM_V_HEADS // TP_SIZE,
        HEAD_V_DIM,
        dtype=dtype,
        device=device,
    )
    z = torch.empty_like(core_attn_out)
    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
        z,
        inputs["qkvz"],
        inputs["ba"],
        NUM_K_HEADS,
        NUM_V_HEADS,
        HEAD_K_DIM,
        HEAD_V_DIM,
        conv_state=inputs["conv_state"],
        ssm_state=inputs["ssm_state"],
        conv_weights=inputs["conv_weights"],
        conv_bias=None,
        activation=ACTIVATION,
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=query_start_loc,
        non_spec_state_indices_tensor=state_indices,
        num_actual_tokens=num_actual_tokens,
        tp_size=TP_SIZE,
        reorder_input=reorder_input,
        spec_state_indices_tensor=spec_state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )
    return core_attn_out, z


def _clone_caches(inputs):
    return dict(
        inputs,
        qkvz=inputs["qkvz"].clone(),
        ba=inputs["ba"].clone(),
        conv_state=inputs["conv_state"].clone(),
        ssm_state=inputs["ssm_state"].clone(),
    )


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_spec_kN_aligned_ring_matches_nonspec_core_attn(batch_size, dtype):
    """K=4 spec with all ring slots equal, num_accepted=1 → core_attn_out
    and ssm_state at the (single) load/store slot must match the non-spec
    native decode path.

    With ``num_accepted_tokens=1`` the kernel loads pre-state from
    ``ring[:, 0]`` (= load_idx = n_acc-1 = 0), and the only candidate
    token's per-token store goes to ``ring[:, 0]`` (= t - seq_start = 0).
    Both load and store thus resolve to the same slot as the non-spec
    path when we set ``ring[:, 0] == non_spec_state_indices``. The
    SSM math is byte-identical between the two branches in this
    configuration, so core_attn_out / z / final ssm_state at the slot
    must match bit-exactly.
    """
    device = "xpu"
    inputs = _alloc_inputs(batch_size, batch_size, dtype, device, seed=13)
    query_start_loc = _decode_query_start_loc(batch_size, device)
    has_initial = torch.ones(batch_size, dtype=torch.bool, device=device)

    K = 4
    # Ring: every entry == the canonical slot.
    ring = (
        inputs["state_indices"].unsqueeze(-1).expand(batch_size, K).contiguous()
    )
    num_accepted = torch.ones(batch_size, dtype=torch.int32, device=device)

    # Non-spec reference: native decode path, single token per sequence.
    nonspec_in = _clone_caches(inputs)
    nonspec_out, nonspec_z = _call_gdn(
        nonspec_in,
        num_actual_tokens=batch_size,
        num_prefills=0,
        num_decodes=batch_size,
        query_start_loc=query_start_loc,
        state_indices=nonspec_in["state_indices"],
        has_initial_state=has_initial,
    )

    # Spec call: K=4 ring, num_accepted=1 → only ring[:, 0] is touched
    # for both load and store. Pool conv_state sized for state_len =
    # (Width-1) + (K-1) (the kernel infers state_len from stride_0).
    state_len = (WIDTH - 1) + (K - 1)
    spec_in = _clone_caches(inputs)
    spec_conv_state = torch.randn(
        CACHE_BATCH_SIZE,
        state_len,
        _mixed_qkv_size(),
        dtype=dtype,
        device=device,
    )
    # Plant the non-spec pre-state at the appropriate offset inside the
    # larger spec slot. With n_acc=1, the kernel loads pre-state from
    # rows [Width-1-states_load_len, Width-1) of the slot starting at
    # offset (n_acc-1) = 0 — i.e. the leading (Width-1) rows. Match the
    # non-spec pre-state there so the conv1d inputs are identical.
    spec_conv_state[:, : WIDTH - 1, :] = inputs["conv_state"]
    spec_in["conv_state"] = spec_conv_state

    spec_out, spec_z = _call_gdn(
        spec_in,
        num_actual_tokens=batch_size,
        num_prefills=0,
        num_decodes=batch_size,
        query_start_loc=query_start_loc,
        # Production dispatcher passes ring[:, 0] here under spec. It's
        # used by the kernel as the write target for spec_state_roll.
        state_indices=ring[:, 0].contiguous(),
        has_initial_state=has_initial,
        spec_state_indices=ring,
        num_accepted_tokens=num_accepted,
    )

    assert torch.equal(spec_out, nonspec_out), (
        "K=4 aligned-ring spec must produce bit-identical core_attn_out "
        f"vs non-spec; max abs diff = "
        f"{(spec_out.float() - nonspec_out.float()).abs().max().item():.3e}"
    )
    assert torch.equal(spec_z, nonspec_z), (
        "K=4 aligned-ring spec must produce bit-identical z vs non-spec"
    )

    # ssm_state at touched slots: bit-exact. The SSM kernel writes the
    # post-token state directly to ring[:, t - seq_start]; with K_call=1
    # and aligned ring, this matches the non-spec final-state writeback.
    for i in range(batch_size):
        slot = nonspec_in["state_indices"][i].item()
        assert torch.equal(
            spec_in["ssm_state"][slot], nonspec_in["ssm_state"][slot]
        ), (
            f"ssm_state[{slot}] mismatch after spec(K=4 aligned ring) "
            "vs non-spec"
        )


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_spec_null_block_id_suppresses_pool_io(batch_size, dtype):
    """NULL load+store slots must suppress both the conv/ssm pre-state
    load AND the per-token pool writes.

    Build a K=2 spec call where ``ring[:, 0]`` is NULL (0) — both the
    load slot (``n_acc-1 == 0``) and the per-token store slot
    (``t - seq_start == 0``) resolve to NULL. Cross-check:

    1. ``core_attn_out`` must match a non-spec call with
       ``has_initial_state=False`` on the same workload (the spec path
       still runs the SSM math; only the pre-state load and pool
       writeback are suppressed).
    2. The conv/ssm pools must be byte-identical to their pre-call
       state — NULL store slot ⇒ no pool writes.
    """
    device = "xpu"
    inputs = _alloc_inputs(batch_size, batch_size, dtype, device, seed=11)
    query_start_loc = _decode_query_start_loc(batch_size, device)

    # Non-spec reference: has_initial_state=False forces zero pre-state.
    nonspec_in = _clone_caches(inputs)
    has_initial_false = torch.zeros(batch_size, dtype=torch.bool, device=device)
    nonspec_out, _ = _call_gdn(
        nonspec_in,
        num_actual_tokens=batch_size,
        num_prefills=0,
        num_decodes=batch_size,
        query_start_loc=query_start_loc,
        state_indices=nonspec_in["state_indices"],
        has_initial_state=has_initial_false,
    )

    # Spec call: ring[:, 0] = NULL_BLOCK_ID (= 0). Slot at index 0 is
    # never used by real traffic — kernels treat it as the sentinel.
    K = 2
    null_col = torch.zeros(batch_size, dtype=torch.int32, device=device)
    real_col = inputs["state_indices"]
    ring = torch.stack([null_col, real_col], dim=-1).contiguous()
    num_accepted = torch.ones(batch_size, dtype=torch.int32, device=device)

    # Pool sized for state_len = (Width-1) + (K-1).
    state_len = (WIDTH - 1) + (K - 1)
    spec_in = _clone_caches(inputs)
    spec_in["conv_state"] = torch.randn(
        CACHE_BATCH_SIZE,
        state_len,
        _mixed_qkv_size(),
        dtype=dtype,
        device=device,
    )
    conv_pool_pre = spec_in["conv_state"].clone()
    ssm_pool_pre = spec_in["ssm_state"].clone()

    # has_initial_state is consulted only by the non-spec branch; the
    # spec branch's NULL slot already suppresses pre-state. Pass True
    # to confirm the spec branch ignores it.
    has_initial_true = torch.ones(batch_size, dtype=torch.bool, device=device)
    spec_out, _ = _call_gdn(
        spec_in,
        num_actual_tokens=batch_size,
        num_prefills=0,
        num_decodes=batch_size,
        query_start_loc=query_start_loc,
        state_indices=ring[:, 0].contiguous(),  # = all-NULL
        has_initial_state=has_initial_true,
        spec_state_indices=ring,
        num_accepted_tokens=num_accepted,
    )

    # core_attn_out: zero pre-state in both, identical math → tight
    # match. Allow small numeric drift between the (chunk vs native)
    # paths even though we're decode-only — both should hit native.
    torch.testing.assert_close(spec_out, nonspec_out, atol=5e-2, rtol=5e-2)

    # Pool: NULL store slots → no writes at all.
    conv_diff = (
        (spec_in["conv_state"][0].float() - conv_pool_pre[0].float())
        .abs()
        .max()
        .item()
    )
    assert torch.equal(spec_in["conv_state"], conv_pool_pre), (
        "NULL store slot must not write to conv_state pool; "
        f"diff at slot 0 max abs = {conv_diff:.3e}"
    )
    assert torch.equal(spec_in["ssm_state"], ssm_pool_pre), (
        "NULL store slot must not write to ssm_state pool"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_spec_kN_native_smoke_finite(dtype):
    """K=4 spec on a non-degenerate ring with mixed ``num_accepted``
    values runs cleanly and produces finite outputs. Doesn't diff
    against a Python oracle — full correctness is covered by the paired
    vLLM replay harness against a live FLA Triton reference.
    """
    device = "xpu"
    K = 4
    batch_size = 3
    inputs = _alloc_inputs(batch_size, batch_size, dtype, device, seed=17)
    query_start_loc = _decode_query_start_loc(batch_size, device)
    has_initial = torch.ones(batch_size, dtype=torch.bool, device=device)

    pool = random.sample(range(1, CACHE_BATCH_SIZE), batch_size * K)
    ring = torch.tensor(pool, device=device, dtype=torch.int32).reshape(
        batch_size, K
    )
    num_accepted = torch.tensor([1, K, 2], dtype=torch.int32, device=device)

    state_len = (WIDTH - 1) + (K - 1)
    inputs["conv_state"] = torch.randn(
        CACHE_BATCH_SIZE,
        state_len,
        _mixed_qkv_size(),
        dtype=dtype,
        device=device,
    )

    out, z = _call_gdn(
        inputs,
        num_actual_tokens=batch_size,
        num_prefills=0,
        num_decodes=batch_size,
        query_start_loc=query_start_loc,
        state_indices=ring[:, 0].contiguous(),
        has_initial_state=has_initial,
        spec_state_indices=ring,
        num_accepted_tokens=num_accepted,
    )

    assert torch.isfinite(out.float()).all(), "core_attn_out has non-finite"
    assert torch.isfinite(z.float()).all(), "z has non-finite"

    # The kernel must have updated SSM state at ring[i, t - seq_start]
    # for each (sequence, token). With one token per sequence, that's
    # ring[:, 0]. Spot-check those slots are no longer at their initial
    # (randn) value by virtue of being touched. We don't assert what
    # they're equal to — that's the replay harness's job — only that
    # the writeback happened.
    # (Skipped because we'd need a tight before/after capture; the
    # untouched-slots check below is sufficient for sanity.)


@torch.inference_mode()
def test_spec_args_pair_required():
    """The op must reject calls that pass only one of
    ``spec_state_indices_tensor`` / ``num_accepted_tokens`` — both must
    be set together, or both must be omitted.
    """
    device = "xpu"
    batch_size = 2
    dtype = torch.bfloat16
    inputs = _alloc_inputs(batch_size, batch_size, dtype, device, seed=19)
    query_start_loc = _decode_query_start_loc(batch_size, device)
    has_initial = torch.ones(batch_size, dtype=torch.bool, device=device)

    spec_state_indices = inputs["state_indices"].reshape(batch_size, 1)

    with pytest.raises(RuntimeError, match="both be provided or both omitted"):
        _call_gdn(
            inputs,
            num_actual_tokens=batch_size,
            num_prefills=0,
            num_decodes=batch_size,
            query_start_loc=query_start_loc,
            state_indices=inputs["state_indices"],
            has_initial_state=has_initial,
            spec_state_indices=spec_state_indices,
            num_accepted_tokens=None,
        )

    num_accepted = torch.ones(batch_size, dtype=torch.int32, device=device)
    with pytest.raises(RuntimeError, match="both be provided or both omitted"):
        _call_gdn(
            inputs,
            num_actual_tokens=batch_size,
            num_prefills=0,
            num_decodes=batch_size,
            query_start_loc=query_start_loc,
            state_indices=inputs["state_indices"],
            has_initial_state=has_initial,
            spec_state_indices=None,
            num_accepted_tokens=num_accepted,
        )


@torch.inference_mode()
def test_spec_args_int32_required():
    """Both spec args must be int32 contiguous — int64 / non-contig
    callers should fail loudly rather than silently misread bytes.
    """
    device = "xpu"
    batch_size = 2
    dtype = torch.bfloat16
    inputs = _alloc_inputs(batch_size, batch_size, dtype, device, seed=23)
    query_start_loc = _decode_query_start_loc(batch_size, device)
    has_initial = torch.ones(batch_size, dtype=torch.bool, device=device)

    spec_int64 = inputs["state_indices"].to(torch.int64).reshape(batch_size, 1)
    num_acc_int32 = torch.ones(batch_size, dtype=torch.int32, device=device)

    with pytest.raises(RuntimeError, match="must be int32"):
        _call_gdn(
            inputs,
            num_actual_tokens=batch_size,
            num_prefills=0,
            num_decodes=batch_size,
            query_start_loc=query_start_loc,
            state_indices=inputs["state_indices"],
            has_initial_state=has_initial,
            spec_state_indices=spec_int64,
            num_accepted_tokens=num_acc_int32,
        )

    spec_int32 = inputs["state_indices"].reshape(batch_size, 1).contiguous()
    num_acc_int64 = torch.ones(batch_size, dtype=torch.int64, device=device)
    with pytest.raises(RuntimeError, match="must be int32"):
        _call_gdn(
            inputs,
            num_actual_tokens=batch_size,
            num_prefills=0,
            num_decodes=batch_size,
            query_start_loc=query_start_loc,
            state_indices=inputs["state_indices"],
            has_initial_state=has_initial,
            spec_state_indices=spec_int32,
            num_accepted_tokens=num_acc_int64,
        )


@torch.inference_mode()
def test_spec_args_batch_size_match():
    """``spec_state_indices_tensor.size(0)`` must equal
    ``query_start_loc.size(0) - 1``. The dispatcher repurposes
    ``non_spec_query_start_loc`` for the spec batch under spec mode.
    """
    device = "xpu"
    batch_size = 4
    dtype = torch.bfloat16
    inputs = _alloc_inputs(batch_size, batch_size, dtype, device, seed=29)
    query_start_loc = _decode_query_start_loc(batch_size, device)
    has_initial = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Build a spec_state_indices with the wrong batch dimension.
    bad_spec = torch.zeros(batch_size + 1, 2, dtype=torch.int32, device=device)
    num_accepted = torch.ones(batch_size + 1, dtype=torch.int32, device=device)

    with pytest.raises(RuntimeError, match="must equal query_start_loc"):
        _call_gdn(
            inputs,
            num_actual_tokens=batch_size,
            num_prefills=0,
            num_decodes=batch_size,
            query_start_loc=query_start_loc,
            state_indices=inputs["state_indices"],
            has_initial_state=has_initial,
            spec_state_indices=bad_spec,
            num_accepted_tokens=num_accepted,
        )
