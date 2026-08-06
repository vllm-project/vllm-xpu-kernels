# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-check the KDA recurrent backends against each other.

`VLLM_XPU_KDA_RECURRENT_MODE` is read once and cached inside the extension, so
each backend has to run in its own process. This module doubles as the worker:
the tests re-invoke it with ``python <this file> <mode> <out.pt>``.
"""

import os
import subprocess
import sys

import pytest
import torch

DEVICE = "xpu"
NUM_HEADS = 4
CASES = {
    # name: (batch, seqlen, head_dim, dtype, state_dtype)
    "prefill_b1_s256_d128": (1, 256, 128, "bfloat16", "float32"),
    "prefill_b2_s192_d64": (2, 192, 64, "bfloat16", "float32"),
    "prefill_b3_s333_d128": (3, 333, 128, "bfloat16", "float32"),
    "prefill_b1_s256_d128_fp16": (1, 256, 128, "float16", "float32"),
    "prefill_b1_s256_d32": (1, 256, 32, "bfloat16", "float32"),
    "decode_b8_d128": (8, 1, 128, "bfloat16", "float32"),
    "decode_b8_d128_bf16state": (8, 1, 128, "bfloat16", "bfloat16"),
    "prefill_b1_s256_d128_bf16state": (1, 256, 128, "bfloat16", "bfloat16"),
    # Names ending in `_perm` are built with a non-contiguous
    # `non_spec_token_indx`, so the kernels have to gather their inputs and
    # scatter their outputs row by row instead of walking a strided tile.
    "prefill_b2_s128_d128_perm": (2, 128, 128, "bfloat16", "float32"),
    "prefill_b2_s192_d64_perm": (2, 192, 64, "bfloat16", "float32"),
    # Names ending in `_lb` use the bounded sigmoid gate
    # (`gate_lower_bound=-5.0`, as in Kimi-K3) instead of the unbounded
    # softplus one.
    "prefill_b1_s256_d128_lb": (1, 256, 128, "bfloat16", "float32"),
    "prefill_b3_s333_d128_lb": (3, 333, 128, "bfloat16", "float32"),
    "decode_b8_d128_lb": (8, 1, 128, "bfloat16", "float32"),
}

GATE_LOWER_BOUND = -5.0
# Trained sigmoid-gate models sit deep in the retention regime, so the gate
# logits are strongly negative. Without the shift a zero-mean logit would give
# g ~ lower_bound/2 = -2.5 per token, i.e. a per-chunk cumulative decay of
# ~-160 -- far past the `g_floor` clamp the chunked pipeline folds into its
# GEMM operands, where no backend can agree with the sequential recurrence.
GATE_LOWER_BOUND_LOGIT_SHIFT = -1.9


def _build(case, seed=0, permute=False, lower_bound=None):
    batch, seqlen, head_dim, dtype_str, state_dtype_str = case
    dtype = getattr(torch, dtype_str)
    state_dtype = getattr(torch, state_dtype_str)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    num_tokens = batch * seqlen
    hidden = NUM_HEADS * head_dim

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(DEVICE)

    # Interleaving the sequences' rows keeps every chunk's destination rows
    # non-consecutive, which is what separates a strided tile store from a
    # correct scatter.
    token_indx = None
    if permute:
        token_indx = torch.cat([
            torch.arange(offset, num_tokens, batch) for offset in range(batch)
        ]).to(DEVICE, torch.int32)

    # Keep the gate scale in the range trained models produce: the chunked
    # path needs the per-chunk cumulative log-decay to stay representable.
    return {
        "q": rand(num_tokens, hidden, scale=0.2).to(dtype),
        "k": rand(num_tokens, hidden, scale=0.2).to(dtype),
        "v": rand(num_tokens, hidden, scale=0.2).to(dtype),
        "raw_gate": rand(1, num_tokens, NUM_HEADS, head_dim,
                         scale=0.2).to(dtype),
        "beta": torch.rand(1, num_tokens, NUM_HEADS,
                           generator=gen).to(DEVICE),
        "state": rand(batch, NUM_HEADS, head_dim, head_dim,
                      scale=0.05).to(state_dtype),
        "a_log": rand(1, 1, NUM_HEADS, 1, scale=0.1),
        "dt_bias": (rand(hidden, scale=0.1) + (
            0.0 if lower_bound is None else GATE_LOWER_BOUND_LOGIT_SHIFT)),
        "out": torch.zeros(1, num_tokens, NUM_HEADS, head_dim,
                           device=DEVICE, dtype=dtype),
        "qsl": torch.arange(0, num_tokens + 1, seqlen,
                            device=DEVICE, dtype=torch.int32),
        "idx": torch.arange(batch, device=DEVICE, dtype=torch.int32),
        "tok": token_indx,
        "hi": torch.ones(batch, device=DEVICE, dtype=torch.bool),
        "num_prefills": batch if seqlen > 1 else 0,
        "num_decodes": 0 if seqlen > 1 else batch,
        "num_tokens": num_tokens,
        "lower_bound": lower_bound,
    }


def _run(t):
    torch.ops._xpu_C.kda_gated_delta_rule(
        t["out"], t["q"], t["k"], t["v"], t["raw_gate"], t["beta"],
        t["state"], t["a_log"], t["dt_bias"], t["num_prefills"],
        t["num_decodes"], 0, t["hi"], t["qsl"], t["tok"], t["idx"],
        None, None, None, None, t["num_tokens"], t["lower_bound"])
    return t["out"].float().cpu(), t["state"].float().cpu()


def _permutation_pair(seed=7):
    """Two descriptions of one batch that must produce identical rows.

    ``identity`` permutes the token rows on the host and uses no token map;
    ``permuted`` keeps the rows where they are and reorders them through
    ``non_spec_token_indx``. Both therefore feed the kernel the same sequences
    in the same order, so their results may only differ in *where* the output
    rows land.
    """
    case = CASES["prefill_b2_s128_d128_perm"]
    permuted = _build(case, seed=seed, permute=True)
    perm = permuted["tok"].long()
    identity = _build(case, seed=seed, permute=False)
    for key in ("q", "k", "v"):
        identity[key] = permuted[key][perm].contiguous()
    identity["raw_gate"] = permuted["raw_gate"][:, perm].contiguous()
    identity["beta"] = permuted["beta"][:, perm].contiguous()
    # `_run` updates the recurrent state in place, so both runs need their own
    # copy of the same starting point.
    identity["state"] = permuted["state"].clone()
    return identity, permuted, perm


def _worker(out_path):
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    results = {}
    for i, (name, case) in enumerate(sorted(CASES.items())):
        results[name] = _run(
            _build(
                case,
                seed=i,
                permute=name.endswith("_perm"),
                lower_bound=(GATE_LOWER_BOUND
                             if name.endswith("_lb") else None),
            ))
    torch.save(results, out_path)


def _scatter_worker(out_path):
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    identity, permuted, perm = _permutation_pair()
    torch.save(
        {
            "identity": _run(identity),
            "permuted": _run(permuted),
            "perm": perm.cpu(),
        }, out_path)


def _collect(mode, tmp_path):
    out_path = str(tmp_path / f"{mode}.pt")
    env = dict(os.environ)
    env["VLLM_XPU_KDA_RECURRENT_MODE"] = mode
    # Force the chunked path to engage on the short test sequences too.
    env["VLLM_XPU_KDA_CHUNK_MIN_SEQLEN"] = "1"
    env.setdefault("PYTHONPATH", os.getcwd())
    subprocess.run(
        [sys.executable, os.path.abspath(__file__), out_path],
        check=True, env=env)
    return torch.load(out_path)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")
@pytest.mark.parametrize("mode", ["opt", "chunk"])
def test_kda_recurrent_backends_match_reference(mode, tmp_path):
    reference = _collect("recurrent", tmp_path)
    actual = _collect(mode, tmp_path)
    assert reference.keys() == actual.keys()
    for name in reference:
        ref_out, ref_state = reference[name]
        act_out, act_state = actual[name]
        torch.testing.assert_close(
            act_out, ref_out, atol=6e-2, rtol=6e-2,
            msg=lambda m, n=name: f"{mode}/{n} output mismatch\n{m}")
        torch.testing.assert_close(
            act_state, ref_state, atol=6e-2, rtol=6e-2,
            msg=lambda m, n=name: f"{mode}/{n} state mismatch\n{m}")


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")
def test_chunk_honours_permuted_token_indx(tmp_path):
    """`non_spec_token_indx` has to be applied to every row, not just the first.

    Reordering the rows on the host and reordering them through the token map
    describe the same batch, so the two runs have to agree row for row. The
    comparison is deliberately much tighter than the cross-backend one above:
    these outputs peak near 1e-2, which is *below* the tolerance that
    comparison can afford, so only a tight bound tells a row-by-row scatter
    apart from a strided tile store anchored at the first index.
    """
    out_path = str(tmp_path / "scatter.pt")
    env = dict(os.environ)
    env["VLLM_XPU_KDA_RECURRENT_MODE"] = "chunk"
    env["VLLM_XPU_KDA_CHUNK_MIN_SEQLEN"] = "1"
    env.setdefault("PYTHONPATH", os.getcwd())
    subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--scatter", out_path],
        check=True, env=env)

    saved = torch.load(out_path)
    identity_out, identity_state = saved["identity"]
    permuted_out, permuted_state = saved["permuted"]
    perm = saved["perm"]

    # Guard against the whole comparison passing on empty output.
    assert identity_out.abs().max() > 0
    torch.testing.assert_close(
        permuted_out[:, perm], identity_out, atol=1e-4, rtol=1e-3,
        msg=lambda m: f"chunk output ignores token_indx row order\n{m}")
    torch.testing.assert_close(
        permuted_state, identity_state, atol=1e-4, rtol=1e-3,
        msg=lambda m: f"chunk state depends on token_indx row order\n{m}")


if __name__ == "__main__":
    if sys.argv[1] == "--scatter":
        _scatter_worker(sys.argv[2])
    else:
        _worker(sys.argv[1])
