# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-free TP1/TP2 layer, cache, and fail-stop checks."""

import io
import json
import os
from functools import partial
from multiprocessing.reduction import recv_handle, send_handle
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from vllm_xpu_kernels import minicpm5_mono
from vllm_xpu_kernels.minicpm5_mono import library_info, load, workspace

requires_extension = pytest.mark.skipif(
    not any(Path(minicpm5_mono.__file__).parent.glob("_minicpm5_mono_C*.so")),
    reason="Requires the MiniCPM5 mono extension",
)
pytestmark = requires_extension

H, Q, KV, D, E, TOP = 2048, 32, 2, 128, 160, 16
DTYPE = torch.bfloat16
ROW_CASES = tuple(
    int(row)
    for row in os.environ.get("MINICPM5_TEST_ROWS", "1,2,4,8").split(",")
)
if (
    not ROW_CASES
    or len(set(ROW_CASES)) != len(ROW_CASES)
    or set(ROW_CASES) - set(range(1, 9))
):
    raise ValueError(
        "MINICPM5_TEST_ROWS must select distinct supported row counts"
    )


@pytest.fixture(scope="module", autouse=True, params=["float16", "bfloat16"])
def activation_dtype(request):
    global DTYPE
    DTYPE = getattr(torch, request.param)
    yield DTYPE
    DTYPE = torch.bfloat16


def linear_weight(n, k, device):
    return torch.randn(n, k, device=device).to(torch.float8_e4m3fn).t()


def make_weights(dense, device):
    scale = torch.tensor([0.02], device=device)
    empty = torch.empty(0, device=device)
    prefix = [
        torch.rand(H, device=device).to(DTYPE) + 0.5,
        torch.rand(H, device=device).to(DTYPE) + 0.5,
        linear_weight(2 * (Q + KV) * D, H, device),
        scale,
        linear_weight(H, Q * D, device),
        scale,
    ]
    if dense:
        return prefix + [
            empty,
            empty,
            linear_weight(16384, H, device),
            scale,
            linear_weight(H, 8192, device),
            scale,
            empty,
            empty,
            empty,
            empty,
        ]
    return prefix + [
        (torch.randn(E, H, device=device) * 0.02).to(DTYPE),
        torch.linspace(-0.01, 0.01, E, device=device),
        torch.randn(E, H, 1024, device=device).to(torch.float8_e4m3fn),
        scale.expand(E).clone(),
        torch.randn(E, 512, H, device=device).to(torch.float8_e4m3fn),
        scale.expand(E).clone(),
        linear_weight(1024, H, device),
        scale,
        linear_weight(H, 512, device),
        scale,
    ]


def norm(x, weight):
    x = x.float()
    return (
        x
        * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
    ).to(DTYPE)


def project(x, weight, scale):
    return (F.linear(x.float(), weight.t().float()) * scale).to(DTYPE)


def project_parts(x, weight, scale, tp):
    inputs = x.chunk(tp, -1)
    weights = weight.chunk(tp, 0)
    return [
        project(value, shard, scale) for value, shard in zip(inputs, weights)
    ]


def sum_parts(parts):
    return torch.stack(parts).float().sum(0).to(DTYPE)


def rotary_cache_kwargs(native, length, device):
    if length <= 32768:
        return {}
    assert native.rotary_cache_version() == 1
    inv = 1 / (5000000 ** (torch.arange(0, D, 2, device=device).float() / D))
    angles = torch.arange(length, device=device).float()[:, None] * inv[None, :]
    return {"rope_cache": torch.cat((angles.cos(), angles.sin()), -1).to(DTYPE)}


def reference(
    hidden,
    residual,
    weights,
    cache,
    blocks,
    lengths,
    positions,
    slots,
    dense,
    tp=1,
    trace=None,
):
    summed = (
        hidden
        if residual is None
        else (hidden.float() + residual.float()).to(DTYPE)
    )
    x = norm(summed, weights[0])
    qkv = project(x, weights[2], weights[3])
    qg = qkv[:, : 2 * Q * D].reshape(-1, Q, 2 * D)
    q, gate = qg[..., :D].clone(), qg[..., D:]
    k = qkv[:, 2 * Q * D : 2 * Q * D + KV * D].reshape(-1, KV, D).clone()
    v = qkv[:, 2 * Q * D + KV * D :].reshape(-1, KV, D)
    inv = 1 / (
        5000000 ** (torch.arange(0, D, 2, device=hidden.device).float() / D)
    )
    angles = positions.float()[:, None] * inv[None, :]
    cs, sn = (
        angles.cos().to(DTYPE).float(),
        angles.sin().to(DTYPE).float(),
    )
    for value in (q, k):
        first, second = (
            value[..., : D // 2].float(),
            value[..., D // 2 :].float(),
        )
        value.copy_(
            torch.cat(
                [
                    first * cs[:, None] - second * sn[:, None],
                    second * cs[:, None] + first * sn[:, None],
                ],
                -1,
            )
        )
    outputs = []
    for row, length in enumerate(lengths.tolist()):
        slot = int(slots[row])
        if slot < 0:
            outputs.append(torch.zeros(Q, D, dtype=DTYPE, device=hidden.device))
            continue
        cache[slot // 64, :, slot % 64, :D] = k[row]
        cache[slot // 64, :, slot % 64, D:] = v[row]
        indices = torch.arange(length, device=hidden.device)
        values = cache[blocks[row, indices // 64].long(), :, indices % 64, :]
        keys = (
            values[..., :D]
            .permute(1, 0, 2)
            .repeat_interleave(Q // KV, 0)
            .float()
        )
        vals = (
            values[..., D:]
            .permute(1, 0, 2)
            .repeat_interleave(Q // KV, 0)
            .float()
        )
        scores = torch.einsum("hd,htd->ht", q[row].float(), keys) / D**0.5
        outputs.append(
            torch.einsum("ht,htd->hd", scores.softmax(-1), vals).to(DTYPE)
        )
    attention = (torch.stack(outputs) * gate.sigmoid()).reshape(-1, Q * D)
    res_out = (
        sum_parts(project_parts(attention, weights[4], weights[5], tp)).float()
        + summed.float()
    ).to(DTYPE)
    return reference_mlp(res_out, weights, dense, tp, trace), res_out


def reference_mlp(
    residual,
    weights,
    dense,
    tp=1,
    trace=None,
    *,
    normalized=None,
    router_parts=None,
):
    """Independent MLP math, optionally split at the RMSNorm boundary.

    A BF16 attention difference can cross a discontinuous top-k boundary.
    Validate attention/cache separately, then normalize and route independently
    from the same residual; never inject native expert IDs into this oracle.
    """
    x = norm(residual, weights[1]) if normalized is None else normalized
    if dense:
        gate, up = project(x, weights[8], weights[9]).chunk(2, -1)
        return sum_parts(
            project_parts(F.silu(gate) * up, weights[10], weights[11], tp)
        )
    if normalized is None:
        scores = F.linear(x.float(), weights[6].float()).sigmoid()
    else:
        assert router_parts is not None
        # Check every 64-K partial against independent FP64 math. Products
        # of these FP16/BF16 fixture operands are exact in FP32. The standard
        # gamma64 bound covers any FP32 accumulation order within a partial.
        a = x.cpu().double().reshape(-1, 32, 64)
        b = weights[6].cpu().double().reshape(E, 32, 64)
        expected_parts = torch.einsum("rpk,epk->pre", a, b)
        magnitudes = torch.einsum("rpk,epk->pre", a.abs(), b.abs())
        unit = torch.finfo(torch.float32).eps / 2
        gamma64 = 64 * unit / (1 - 64 * unit)
        bound = gamma64 * magnitudes + 64 * torch.finfo(torch.float32).tiny
        error = (router_parts.double() - expected_parts).abs()
        assert bool((error <= bound).all()), (error.max(), bound.max())
        # Sum WG partials in index order and keep the logits in FP32.
        # Validate sigmoid, top-k and MLP using independent expert IDs.
        logits = torch.zeros_like(router_parts[0])
        for part in router_parts:
            logits.add_(part)
        scores = logits.sigmoid().to(x.device)
    selected = (scores + weights[7]).topk(TOP, dim=-1).indices
    routing = scores.gather(-1, selected)
    routing = routing / routing.sum(-1, keepdim=True) * 3.66
    if trace is not None:
        trace.update(
            normed=x.cpu(),
            router=scores.cpu(),
            experts=selected.cpu(),
            routing=routing.cpu(),
        )
    shared_gate, shared_up = project(x, weights[12], weights[13]).chunk(2, -1)
    shared = project_parts(
        F.silu(shared_gate) * shared_up, weights[14], weights[15], tp
    )
    routed = [torch.zeros_like(value, dtype=torch.float32) for value in shared]
    for row in range(residual.shape[0]):
        for choice in range(TOP):
            expert = int(selected[row, choice])
            gate, up = (
                (
                    x[row].float()
                    @ weights[8][expert].float()
                    * weights[9][expert]
                )
                .to(DTYPE)
                .chunk(2, -1)
            )
            intermediate = F.silu(gate) * up
            for part, (value, down) in enumerate(
                zip(
                    intermediate.chunk(tp, -1), weights[10][expert].chunk(tp, 0)
                )
            ):
                out = (value.float() @ down.float() * weights[11][expert]).to(
                    DTYPE
                )
                routed[part][row] += routing[row, choice] * out.float()
    return sum_parts(
        [value.to(DTYPE) + direct for value, direct in zip(routed, shared)]
    )


def shard_weights(weights, rank, dense, device):
    def column(weight, sizes):
        pieces = [part.chunk(2, 0)[rank] for part in weight.t().split(sizes, 0)]
        return torch.cat(pieces, 0).contiguous().t()

    def row(weight):
        return weight.t().chunk(2, 1)[rank].contiguous().t()

    result = list(weights)
    result[2] = column(weights[2], [2 * Q * D, KV * D, KV * D])
    result[4] = row(weights[4])
    if dense:
        result[8] = column(weights[8], [8192, 8192])
        result[10] = row(weights[10])
    else:
        result[8] = torch.cat(
            [part.chunk(2, -1)[rank] for part in weights[8].chunk(2, -1)],
            -1,
        ).contiguous()
        result[10] = weights[10].chunk(2, 1)[rank].contiguous()
        result[12] = column(weights[12], [512, 512])
        result[14] = row(weights[14])
    # Copy physical NK storage, preserving the ordinary XPU KN view.
    return [
        value.t().contiguous().to(device).t()
        if value.dim() == 2 and value.stride(0) == 1
        else value.contiguous().to(device)
        for value in result
    ]


def _tp2_worker(
    rank,
    connections,
    parent_pid,
    expected_library,
    dtype,
    query_tokens=1,
    use_graph=False,
    chunked_verify=False,
):
    global DTYPE
    DTYPE = dtype
    row_cases = (
        (16,)
        if chunked_verify
        else tuple(r for r in ROW_CASES if r % query_tokens == 0)
    )
    torch.xpu.set_device(rank)
    device = f"xpu:{rank}"
    connection = connections[rank]
    connections[1 - rank].close()
    ops = load()
    actual_library = library_info()
    assert actual_library == expected_library
    print(json.dumps({"rank": rank, "library": actual_library}), flush=True)
    # Match production: load PyTorch weights before mapping native IPC.
    torch.manual_seed(42)
    initial_weights = shard_weights(
        make_weights(True, "cpu"), rank, True, device
    )
    torch.xpu.synchronize()
    print(f"rank={rank} initial weights loaded", flush=True)
    owner = torch.classes.minicpm5_mono.Region(
        4 * len(row_cases) * ops.mailbox_bytes()
    )
    info = owner.export_info()
    connection.send(info)
    send_handle(connection, int(info[2]), parent_pid)
    peer_info = connection.recv()
    print(f"rank={rank} opening peer mapping", flush=True)
    owner.open(peer_info, recv_handle(connection))
    print(f"rank={rank} peer mapping verified", flush=True)
    connection.send("mapped")
    assert connection.recv() == "start"
    del initial_weights
    results = {}
    slot = 0
    for dense in (True, False):
        torch.manual_seed(42)
        full = make_weights(dense, "cpu")
        for rows in row_cases:
            for swap in (False, True):
                shard = rank ^ int(swap)
                weights = shard_weights(full, shard, dense, device)
                torch.manual_seed(123 + rows)
                hidden, cache, blocks, slots, positions, lengths = inputs(
                    rows,
                    16385 if rows >= 8 else 129,
                    "cpu",
                    query_tokens=query_tokens,
                )
                if rows > 1 and query_tokens == 1:
                    slots[-1] = -1
                residual = torch.randn_like(hidden) * 0.1
                local_cache = (
                    cache[:, shard : shard + 1].contiguous().to(device)
                )
                tensors = [
                    value.to(device)
                    for value in (
                        hidden,
                        residual,
                        blocks,
                        slots,
                        positions,
                        lengths,
                    )
                ]
                if os.environ.get("MINICPM5_PAD_TEST_ROWS"):
                    count = 1 << (rows - 1).bit_length()
                    tensors = [
                        torch.cat(
                            [
                                value,
                                torch.full(
                                    (count - rows, *value.shape[1:]),
                                    -1 if i == 3 else 0,
                                    dtype=value.dtype,
                                    device=device,
                                ),
                            ]
                        )
                        for i, value in enumerate(tensors)
                    ]
                x, residual_x, bt, sm, pos, seq = tensors
                out, res = torch.empty_like(x), torch.empty_like(x)
                decode_kwargs = rotary_cache_kwargs(ops, int(seq.max()), device)
                scratch = workspace(2, device, DTYPE)
                control = torch.zeros(
                    ops.control_words(), dtype=torch.int32, device=device
                )
                connection.send((dense, rows, swap))
                assert connection.recv() == "ready"

                calls = []
                for start in range(0, rows, 8):
                    end = min(start + 8, rows)
                    chunk_kwargs = dict(decode_kwargs)
                    table = bt[start:end]
                    if (
                        swap
                        and torch.float16 == DTYPE
                        and query_tokens == 8
                        and end - start == 8
                        and hasattr(ops, "shared_verify_pages_version")
                    ):
                        table = table[:1]
                    if (
                        not dense
                        and end - start == 8
                        and (query_tokens > 1 or chunked_verify)
                    ):
                        chunk_kwargs["ignore_padding"] = True
                    calls.append(
                        partial(
                            ops.decode,
                            x[start:end],
                            residual_x[start:end],
                            pos[start:end],
                            table,
                            seq[start:end],
                            sm[start:end],
                            out[start:end],
                            res[start:end],
                            scratch,
                            control,
                            weights,
                            local_cache,
                            owner,
                            slot,
                            2,
                            dense,
                            **chunk_kwargs,
                        )
                    )

                def launch(_calls=tuple(calls)):
                    for call in _calls:
                        call()

                graph = None
                for step in range(2):
                    print(
                        f"rank={rank} dense={dense} rows={rows} "
                        f"swap={swap} step={step}",
                        flush=True,
                    )
                    if step:
                        x.mul_(-1.25)
                        residual_x.add_(0.0625)
                    if use_graph:
                        if graph is None:
                            launch()
                            torch.xpu.synchronize()
                            graph = torch.xpu.XPUGraph()
                            with torch.xpu.graph(
                                graph, stream=torch.xpu.current_stream()
                            ):
                                launch()
                        graph.replay()
                    else:
                        launch()
                    torch.xpu.synchronize()
                    if int(control[ops.health_offset()]):
                        print(
                            json.dumps(
                                {
                                    "rank": rank,
                                    "control": control.cpu().tolist(),
                                }
                            ),
                            flush=True,
                        )
                    assert int(control[ops.health_offset()]) == 0, (
                        f"rank={rank} dense={dense} rows={rows} swap={swap} "
                        f"step={step} failed the native health check: "
                        f"{control.cpu().tolist()}"
                    )
                entry = {
                    "out": out[:rows].cpu(),
                    "res": res[:rows].cpu(),
                    "cache": local_cache.cpu(),
                }
                if not dense and not chunked_verify:
                    entry["native_trace"] = {
                        "normed": scratch[0][:rows].cpu(),
                        "router": scratch[5][:rows].cpu(),
                        "routing": scratch[6][:rows].cpu(),
                        "experts": scratch[7][:rows].cpu(),
                    }
                if rank == 0:
                    expected_trace = {}
                    expected_cache = cache.to(device)
                    expected, expected_res = reference(
                        x[:rows],
                        residual_x[:rows],
                        [value.to(device) for value in full],
                        expected_cache,
                        bt[:rows],
                        seq[:rows],
                        pos[:rows],
                        sm[:rows],
                        dense,
                        tp=2,
                        trace=expected_trace,
                    )
                    if not dense and not chunked_verify:
                        stage_trace = {}
                        stage_output = reference_mlp(
                            res[:rows],
                            [value.to(device) for value in full],
                            dense,
                            tp=2,
                            trace=stage_trace,
                            normalized=scratch[0][:rows],
                            router_parts=(
                                scratch[3]
                                .reshape(-1)[: 32 * x.shape[0] * E]
                                .reshape(32, x.shape[0], E)[:, :rows]
                                .cpu()
                            ),
                        )
                        stage_trace["residual_norm"] = norm(
                            res[:rows], full[1].to(device)
                        ).cpu()
                        entry.update(
                            stage_output=stage_output.cpu(),
                            stage_trace=stage_trace,
                        )
                    entry.update(
                        expected=expected.cpu(),
                        expected_res=expected_res.cpu(),
                        expected_cache=expected_cache.cpu(),
                        expected_trace=expected_trace,
                    )
                results[(dense, rows, swap)] = entry
                slot += 1
    buffer = io.BytesIO()
    torch.save(results, buffer)
    connection.send_bytes(buffer.getvalue())
    connection.close()


def test_tp2_real_ipc_matches_sharded_math_and_rank_swap_symmetry():
    """Distinct TP shards must exchange peer values and preserve paged KV."""
    _check_tp2_sharded_reference()


@pytest.mark.parametrize("query_tokens", [2, 4, 8])
@pytest.mark.parametrize("use_graph", [False, True])
def test_tp2_verify_causal_shared_cache_matches_sharded_math(
    query_tokens, use_graph
):
    """Shared-page verification keeps causal math on Graph replay."""
    _check_tp2_sharded_reference(query_tokens, use_graph)


@pytest.mark.parametrize("use_graph", [False, True])
def test_tp2_two_request_verify_chunks_match_sharded_math(use_graph):
    """Two 8-row chunks preserve both requests' causal cache and IPC epochs."""
    _check_tp2_sharded_reference(8, use_graph, chunked_verify=True)


def _check_tp2_sharded_reference(
    query_tokens=1, use_graph=False, chunked_verify=False
):
    row_cases = (
        (16,)
        if chunked_verify
        else tuple(r for r in ROW_CASES if r % query_tokens == 0)
    )
    if not row_cases:
        pytest.skip("No selected row count fits this verify geometry")
    require_supported_devices(2)
    parents, children = zip(*(mp.Pipe() for _ in range(2)))
    expected_library = library_info()
    assert expected_library["abi"] == 3
    workers = mp.spawn(
        _tp2_worker,
        args=(
            children,
            os.getpid(),
            expected_library,
            DTYPE,
            query_tokens,
            use_graph,
            chunked_verify,
        ),
        nprocs=2,
        join=False,
    )
    fds = []
    try:
        for child in children:
            child.close()
        infos = []
        for parent in parents:
            infos.append(parent.recv())
            fds.append(recv_handle(parent))
        for rank, parent in enumerate(parents):
            parent.send(infos[1 - rank])
            send_handle(parent, fds[1 - rank], workers.processes[rank].pid)
        assert all(parent.recv() == "mapped" for parent in parents)
        for parent in parents:
            parent.send("start")
        # Never launch a persistent wait while the peer is still preparing
        # its IPC mapping or copying the next case's weights and metadata.
        for dense in (True, False):
            for rows in row_cases:
                for swap in (False, True):
                    assert all(
                        parent.recv() == (dense, rows, swap)
                        for parent in parents
                    )
                    for parent in parents:
                        parent.send("ready")
        try:
            outputs = [
                torch.load(io.BytesIO(parent.recv_bytes()), weights_only=True)
                for parent in parents
            ]
        except EOFError:
            # Surface the worker's assertion instead of a pipe EOF.
            workers.join(timeout=10)
            raise
        while not workers.join():
            pass
        for key, first in outputs[0].items():
            print(f"checking TP2 dense/rows/swap={key}", flush=True)
            second = outputs[1][key]
            torch.testing.assert_close(
                first["out"], second["out"], atol=0, rtol=0
            )
            torch.testing.assert_close(
                first["res"], second["res"], atol=0, rtol=0
            )
            try:
                # Keep all original whole-layer regressions. New shapes also
                # validate each MLP stage with identical boundary inputs.
                if not key[0] and not chunked_verify:
                    actual = first["native_trace"]
                    stage = first["stage_trace"]
                    torch.testing.assert_close(
                        actual["normed"],
                        stage["residual_norm"],
                        atol=0.05,
                        rtol=0.03,
                    )
                    torch.testing.assert_close(
                        actual["router"], stage["router"], atol=1e-6, rtol=1e-6
                    )
                    torch.testing.assert_close(
                        actual["experts"].long(),
                        stage["experts"],
                        atol=0,
                        rtol=0,
                    )
                    torch.testing.assert_close(
                        first["out"],
                        first["stage_output"],
                        atol=0.05,
                        rtol=0.03,
                    )
                if key[0] or key[1] in (1, 2, 4, 8) or chunked_verify:
                    torch.testing.assert_close(
                        first["out"], first["expected"], atol=0.05, rtol=0.03
                    )
            except AssertionError:
                diagnostic = os.environ.get("MINICPM5_MATH_DIAGNOSTIC")
                if diagnostic:
                    torch.save(
                        {"case": key, "dtype": DTYPE, "ranks": [first, second]},
                        diagnostic,
                    )
                if not key[0] and not chunked_verify:
                    actual = first["native_trace"]
                    expected = first["expected_trace"]
                    print(
                        json.dumps(
                            {
                                "case": key,
                                "router_id_mismatches": (
                                    actual["experts"] != expected["experts"]
                                )
                                .sum()
                                .item(),
                                "norm_max_abs": (
                                    actual["normed"].float()
                                    - expected["normed"].float()
                                )
                                .abs()
                                .max()
                                .item(),
                                "router_max_abs": (
                                    actual["router"] - expected["router"]
                                )
                                .abs()
                                .max()
                                .item(),
                                "actual_experts": actual["experts"].tolist(),
                                "expected_experts": expected[
                                    "experts"
                                ].tolist(),
                            }
                        ),
                        flush=True,
                    )
                raise
            torch.testing.assert_close(
                first["res"], first["expected_res"], atol=0.05, rtol=0.03
            )
            for rank, entry in enumerate((first, second)):
                shard = rank ^ int(key[2])
                torch.testing.assert_close(
                    entry["cache"],
                    first["expected_cache"][:, shard : shard + 1],
                    atol=0.02,
                    rtol=0.02,
                )
            if key[2]:
                torch.testing.assert_close(
                    first["out"],
                    outputs[0][(key[0], key[1], False)]["out"],
                    atol=0,
                    rtol=0,
                )
    except EOFError:
        workers.join(timeout=10)
        raise
    finally:
        for parent in parents:
            parent.close()
        for fd in fds:
            os.close(fd)
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def require_supported_devices(count):
    if torch.xpu.device_count() < count:
        pytest.skip(f"Requires {count} qualified Xe2 devices")
    ops = load()
    for index in range(count):
        if hasattr(ops, "supports_device"):
            supported = ops.supports_device(index)
        else:
            with torch.xpu.device(index):
                supported = ops.supported()
        if not supported:
            pytest.skip(f"XPU device {index} does not support MiniCPM5 mono")
    return ops


@pytest.fixture(scope="module")
def native():
    ops = require_supported_devices(1)
    torch.xpu.set_device(0)
    assert library_info()["abi"] == 3
    return ops


def test_shared_health_matches_framework_runtime_contract(native):
    # SpecializedDecode selects shared FailStop with this exact native
    # runtime contract. A descriptive suffix must not silently select the
    # slower per-layer path when Region already owns shared health.
    owner = torch.classes.minicpm5_mono.Region(native.mailbox_bytes())
    shared = (
        native.runtime_variant()
        == "reference-shared-health-parallel-page-validation"
    )
    if shared:
        health = owner.health()
        assert health.dtype == torch.int32 and health.device.type == "xpu"
        assert health.numel() == 8 and torch.count_nonzero(health).item() == 0
    else:
        with pytest.raises(RuntimeError, match="reference mono runtime"):
            owner.health()


def _candidate_worker(rank, connections, parent_pid, expected_library, dtype):
    global DTYPE
    DTYPE = dtype
    torch.xpu.set_device(rank)
    device = f"xpu:{rank}"
    connection = connections[rank]
    connections[1 - rank].close()
    ops = load()
    assert library_info() == expected_library
    assert hasattr(ops, "candidate_topk")
    # Establish the queue before the peer import, as the model's LM head does.
    torch.ones(1, device=device).cpu()
    owner = torch.classes.minicpm5_mono.Region(
        max(ops.candidate_mailbox_bytes(), ops.mailbox_bytes())
    )
    info = owner.export_info()
    connection.send(info)
    send_handle(connection, int(info[2]), parent_pid)
    peer_info = connection.recv()
    owner.open(peer_info, recv_handle(connection))
    connection.send("mapped")
    assert connection.recv() == "start"
    mailbox_control = torch.zeros(
        ops.control_words(), dtype=torch.int32, device=device
    )
    for rows in ROW_CASES:
        for step in range(2):
            connection.send(("mailbox", rows, step))
            assert connection.recv() == "ready"
            data = torch.full((rows, H), rank + step * 0.25, device=device)
            exchanged = torch.empty_like(data)
            ops.mailbox_exchange(data, exchanged, mailbox_control, owner)
            torch.xpu.synchronize()
            assert int(mailbox_control[ops.health_offset()]) == 0
            torch.testing.assert_close(
                exchanged, torch.full_like(data, 1 + step * 0.5), atol=0, rtol=0
            )
    # Independent call counters require independent mailbox regions. Otherwise
    # a later candidate tag can match an old probe packet before its peer sends.
    mailbox_owner = owner
    owner = torch.classes.minicpm5_mono.Region(ops.candidate_mailbox_bytes())
    info = owner.export_info()
    connection.send(info)
    send_handle(connection, int(info[2]), parent_pid)
    owner.open(connection.recv(), recv_handle(connection))
    connection.send("mapped")
    assert connection.recv() == "start"
    partial = torch.empty((8, 32, 64), dtype=torch.int64, device=device)
    control = torch.zeros(ops.control_words(), dtype=torch.int32, device=device)
    cases = 0
    for rows in ROW_CASES:
        torch.manual_seed(93 + rows)
        full = torch.randint(-4, 5, (rows, 130560)).to(DTYPE)
        full[:, 0] = 131072 if torch.bfloat16 == DTYPE else 32768
        full[:, 65280] = full[:, 0]
        full[:, 10] = -0.0
        full[:, 11] = 0.0
        for changed in (False, True):
            if changed:
                full.mul_(-1.25)
            shard = (
                full[:, rank * 65280 : (rank + 1) * 65280]
                .contiguous()
                .to(device)
            )
            for k in (1, 8, 64):
                connection.send((rows, changed, k))
                assert connection.recv() == "ready"
                values = torch.empty(
                    (rows, 128), dtype=torch.float32, device=device
                )
                indices = torch.empty(
                    (rows, 128), dtype=torch.int32, device=device
                )
                ops.candidate_topk(
                    values, indices, shard, k, rank, partial, control, owner
                )
                torch.xpu.synchronize()
                assert int(control[ops.health_offset()]) == 0
                expected_values, expected_indices = [], []
                for peer in (0, 1):
                    source = full[:, peer * 65280 : (peer + 1) * 65280].float()
                    ids = source.argsort(dim=-1, descending=True, stable=True)[
                        :, :k
                    ]
                    ids = ids.sort(-1).values
                    expected_values.append(source.gather(1, ids))
                    expected_indices.append(ids + peer * 65280)
                expected_values = torch.cat(expected_values, -1)
                expected_indices = torch.cat(expected_indices, -1)
                actual_values, actual_indices = (
                    values.cpu(),
                    indices.cpu().long(),
                )
                torch.testing.assert_close(
                    actual_values[:, : 2 * k], expected_values, atol=0, rtol=0
                )
                assert torch.equal(actual_indices[:, : 2 * k], expected_indices)
                assert bool(actual_values[:, 2 * k :].isneginf().all())
                winners = actual_indices.gather(
                    1, actual_values.argmax(-1, keepdim=True)
                )
                assert torch.equal(winners[:, 0], full.float().argmax(-1))
                if k == 1:
                    assert ops.candidate_greedy_version() == 1
                    sampled = torch.empty(
                        (rows, 1), dtype=torch.int64, device=device
                    )
                    counts = torch.empty(rows, dtype=torch.int32, device=device)
                    rejected = torch.empty_like(counts)
                    ops.candidate_greedy(
                        values,
                        indices,
                        shard,
                        rank,
                        partial,
                        control,
                        owner,
                        sampled,
                        counts,
                        rejected,
                    )
                    torch.xpu.synchronize()
                    assert torch.equal(
                        sampled.cpu()[:, 0], full.float().argmax(-1)
                    )
                    assert torch.equal(
                        counts.cpu(), torch.ones(rows, dtype=torch.int32)
                    )
                    assert torch.equal(
                        rejected.cpu(), torch.zeros(rows, dtype=torch.int32)
                    )
                    assert int(control[ops.health_offset()]) == 0
                    tag_before = int(control[0])
                    with pytest.raises(RuntimeError, match="greedy tokens"):
                        ops.candidate_greedy(
                            values,
                            indices,
                            shard,
                            rank,
                            partial,
                            control,
                            owner,
                            sampled.to(torch.int32),
                            counts,
                            rejected,
                        )
                    with pytest.raises(RuntimeError, match="overlap"):
                        ops.candidate_greedy(
                            values,
                            indices,
                            shard,
                            rank,
                            partial,
                            control,
                            owner,
                            sampled,
                            counts,
                            counts,
                        )
                    assert int(control[0]) == tag_before
                    if (
                        torch.float16 == DTYPE
                        and not changed
                        and rows == ROW_CASES[0]
                    ):
                        bias = torch.zeros_like(full)
                        bias[:, 0] = -32768
                        bias[:, 65281] = 32768
                        for use_nan in (False, True):
                            if use_nan:
                                bias[:, 15] = -float("nan")
                                bias[:, 65282] = float("nan")
                            local_bias = (
                                bias[:, rank * 65280 : (rank + 1) * 65280]
                                .contiguous()
                                .to(device)
                            )
                            ops.candidate_greedy_bias(
                                values,
                                indices,
                                shard,
                                rank,
                                partial,
                                control,
                                owner,
                                sampled,
                                counts,
                                rejected,
                                local_bias,
                            )
                            assert torch.equal(
                                sampled.cpu()[:, 0],
                                (full + bias).float().argmax(-1),
                            )
                            assert int(control[ops.health_offset()]) == 0
                cases += 1
    connection.send(("mailbox-fault",))
    assert connection.recv() == "ready"
    data = torch.full((1, H), float(rank), device=device)
    exchanged = torch.full_like(data, 17)
    ops.mailbox_exchange(
        data, exchanged, mailbox_control, mailbox_owner, 1 if rank == 0 else 0
    )
    torch.xpu.synchronize()
    if rank == 1:
        assert int(mailbox_control[ops.health_offset()]) == 1
        assert int(mailbox_control[ops.health_offset() + 1]) in (1, 2)
        exchanged.fill_(17)
        ops.mailbox_exchange(data, exchanged, mailbox_control, mailbox_owner)
        torch.xpu.synchronize()
        assert bool((exchanged == 17).all()), (
            "Sticky peer failure must stop before writes"
        )
    connection.send({"rank": rank, "cases": cases, "library": expected_library})
    connection.close()


def test_tp2_candidate_exchange_preserves_ties_global_ids_and_changed_inputs():
    require_supported_devices(2)
    parents, children = zip(*(mp.Pipe() for _ in range(2)))
    expected_library = library_info()
    workers = mp.spawn(
        _candidate_worker,
        args=(children, os.getpid(), expected_library, DTYPE),
        nprocs=2,
        join=False,
    )
    fds = []
    try:
        for child in children:
            child.close()
        infos = [parent.recv() for parent in parents]
        fds = [recv_handle(parent) for parent in parents]
        for rank, parent in enumerate(parents):
            parent.send(infos[1 - rank])
            send_handle(parent, fds[1 - rank], workers.processes[rank].pid)
        assert all(parent.recv() == "mapped" for parent in parents)
        for parent in parents:
            parent.send("start")
        for rows in ROW_CASES:
            for step in range(2):
                assert all(
                    parent.recv() == ("mailbox", rows, step)
                    for parent in parents
                )
                for parent in parents:
                    parent.send("ready")
        infos = [parent.recv() for parent in parents]
        candidate_fds = [recv_handle(parent) for parent in parents]
        fds.extend(candidate_fds)
        for rank, parent in enumerate(parents):
            parent.send(infos[1 - rank])
            send_handle(
                parent, candidate_fds[1 - rank], workers.processes[rank].pid
            )
        assert all(parent.recv() == "mapped" for parent in parents)
        for parent in parents:
            parent.send("start")
        for rows in ROW_CASES:
            for changed in (False, True):
                for k in (1, 8, 64):
                    assert all(
                        parent.recv() == (rows, changed, k)
                        for parent in parents
                    )
                    for parent in parents:
                        parent.send("ready")
        assert all(parent.recv() == ("mailbox-fault",) for parent in parents)
        for parent in parents:
            parent.send("ready")
        reports = [parent.recv() for parent in parents]
        while not workers.join():
            pass
        assert {row["rank"] for row in reports} == {0, 1}
        assert all(row["cases"] == len(ROW_CASES) * 6 for row in reports)
    except EOFError:
        workers.join(timeout=10)
        raise
    finally:
        for parent in parents:
            parent.close()
        for fd in fds:
            os.close(fd)
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def inputs(rows, length, device, cache_order="BNHD", *, query_tokens=1):
    pages = (length + 63) // 64
    cache = torch.randn(
        rows * pages, 64, KV, 2 * D, dtype=DTYPE, device=device
    ).permute(0, 2, 1, 3)
    if cache_order == "BHND":
        cache = cache.contiguous()
    blocks = torch.arange(
        rows * pages, dtype=torch.int32, device=device
    ).reshape(rows, pages)
    slots = (blocks[:, -1].long() * 64 + (length - 1) % 64).contiguous()
    positions = torch.full(
        (rows,), length - 1, dtype=torch.int64, device=device
    )
    lengths = torch.full((rows,), length, dtype=torch.int32, device=device)
    if query_tokens > 1:
        assert rows % query_tokens == 0 and length >= query_tokens
        for first in range(0, rows, query_tokens):
            for token in range(query_tokens):
                row = first + token
                position = length - query_tokens + token
                blocks[row].copy_(blocks[first])
                positions[row] = position
                lengths[row] = position + 1
                slots[row] = (
                    blocks[row, position // 64].long() * 64 + position % 64
                )
    hidden = torch.randn(rows, H, dtype=DTYPE, device=device)
    return hidden, cache, blocks, slots, positions, lengths


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("rows", ROW_CASES)
@pytest.mark.parametrize("cache_order", ["BNHD", "BHND"])
@pytest.mark.parametrize("length", [1, 65, 129, 32768])
def test_real_shape_layer_matches_independent_math_and_cache(
    native, dense, rows, cache_order, length, mixed_lengths=False
):
    torch.manual_seed(42)
    device = "xpu:0"
    weights = make_weights(dense, device)
    hidden, cache, blocks, slots, positions, lengths = inputs(
        rows, length, device, cache_order
    )
    if mixed_lengths:
        lengths.copy_(
            torch.tensor(
                [32768, 1, 65, 129, 1024, 8192, 16385, 0],
                dtype=torch.int32,
                device=device,
            )
        )
        positions.copy_((lengths.long() - 1).clamp_min(0))
        slots.copy_(
            blocks[torch.arange(rows, device=device), positions // 64].long()
            * 64
            + positions % 64
        )
    if rows > 1:
        slots[-1] = -1
    residual = None if dense else torch.randn_like(hidden) * 0.1
    reference_cache = cache.clone()
    reference_routing = {} if torch.bfloat16 == DTYPE and not dense else None
    expected, expected_res = reference(
        hidden,
        residual,
        weights,
        reference_cache,
        blocks,
        lengths,
        positions,
        slots,
        dense,
        trace=reference_routing,
    )
    out, res = torch.empty_like(hidden), torch.empty_like(hidden)
    control = torch.zeros(
        native.control_words(), dtype=torch.int32, device=device
    )
    owner = torch.classes.minicpm5_mono.Region(native.mailbox_bytes())
    native.decode(
        hidden,
        residual,
        positions,
        blocks,
        lengths,
        slots,
        out,
        res,
        workspace(1, device, DTYPE),
        control,
        weights,
        cache,
        owner,
        0,
        1,
        dense,
        **rotary_cache_kwargs(native, length, device),
    )
    torch.xpu.synchronize()
    assert int(control[native.health_offset()]) == 0
    # FP32 reduction trees differ; BF16 output resolution bounds this initial
    # layer test. Full-model FP8 ordinary/mono fidelity is a separate gate.
    torch.testing.assert_close(cache, reference_cache, atol=0.02, rtol=0.02)
    torch.testing.assert_close(res, expected_res, atol=0.05, rtol=0.03)
    if reference_routing is not None:
        # BF16 rounding in the checked residual can change top-k membership.
        # Recompute those rows from the same residual with independent routing.
        x = norm(res, weights[1])
        scores = F.linear(x.float(), weights[6].float()).sigmoid()
        chosen = (scores + weights[7]).topk(TOP, dim=-1).indices
        original = reference_routing["experts"].to(device)
        changed = (chosen.sort().values != original.sort().values).any(dim=-1)
        if changed.any():
            expected[changed] = reference_mlp(res[changed], weights, dense)
    torch.testing.assert_close(out, expected, atol=0.05, rtol=0.03)


@pytest.mark.parametrize("cache_order", ["BNHD", "BHND"])
@pytest.mark.parametrize(
    ("query_tokens", "long_length"), [(4, 32775), (8, 32775), (8, 32783)]
)
def test_tp1_verify_graph_preserves_causal_prefixes_and_two_kv_heads(
    native, cache_order, query_tokens, long_length
):
    """Graph replay must read current pages, lengths and each KV head."""
    if torch.float16 != DTYPE:
        pytest.skip("The specialized verify path requires FP16")
    torch.manual_seed(42)
    rows, device = 8, "xpu:0"
    weights = make_weights(False, device)
    hidden, cache, blocks, slots, positions, lengths = inputs(
        rows, long_length, device, cache_order, query_tokens=query_tokens
    )
    base = cache.clone()
    residual = torch.randn_like(hidden) * 0.1
    out, res = torch.empty_like(hidden), torch.empty_like(hidden)
    scratch = workspace(1, device, DTYPE)
    control = torch.zeros(
        native.control_words(), dtype=torch.int32, device=device
    )
    owner = torch.classes.minicpm5_mono.Region(native.mailbox_bytes())
    rope_kwargs = rotary_cache_kwargs(native, long_length, device)

    def run():
        table = blocks[:1] if query_tokens == 8 else blocks
        native.decode(
            hidden,
            residual,
            positions,
            table,
            lengths,
            slots,
            out,
            res,
            scratch,
            control,
            weights,
            cache,
            owner,
            0,
            1,
            False,
            ignore_padding=True,
            **rope_kwargs,
        )

    run()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=torch.xpu.current_stream()):
        run()
    for length in (long_length, 65, 129, long_length):
        cache.copy_(base)
        for start in range(0, rows, query_tokens):
            positions[start : start + query_tokens] = torch.arange(
                length - query_tokens, length, device=device
            )
        lengths.copy_(positions.int() + 1)
        slots.copy_(
            blocks[torch.arange(rows, device=device), positions // 64].long()
            * 64
            + positions % 64
        )
        hidden.mul_(0.95)
        expected_cache = base.clone()
        expected, expected_res = reference(
            hidden,
            residual,
            weights,
            expected_cache,
            blocks,
            lengths,
            positions,
            slots,
            False,
        )
        graph.replay()
        torch.xpu.synchronize()
        assert int(control[native.health_offset()]) == 0
        assert owner.health()[0].item() == 0
        torch.testing.assert_close(cache, expected_cache, atol=0.02, rtol=0.02)
        torch.testing.assert_close(res, expected_res, atol=0.05, rtol=0.03)
        torch.testing.assert_close(out, expected, atol=0.05, rtol=0.03)


@pytest.mark.parametrize("cache_order", ["BNHD", "BHND"])
@pytest.mark.parametrize("owners", ["shared", "distinct"])
@pytest.mark.parametrize("length_pattern", ["uniform", "mixed"])
def test_tp1_draft_graph_preserves_noncausal_two_head_cache(
    native, cache_order, owners, length_pattern
):
    """Graph replays preserve both KV heads and per-query lengths."""
    if torch.float16 != DTYPE:
        pytest.skip("Draft mono requires FP16")
    if 1 not in native.draft_tensor_parallel_sizes():
        pytest.skip("Native library predates TP1 draft support")
    torch.manual_seed(12)
    rows, device, count = 7, "xpu:0", 32775

    def rand(*shape, scale=1.0):
        return (
            torch.randn(shape, dtype=DTYPE, device=device) * scale
        ).contiguous()

    weights = [
        rand(H, scale=0.15) + 1,
        rand(H, scale=0.15) + 1,
        rand(D, scale=0.15) + 1,
        rand(D, scale=0.15) + 1,
        rand(4608, H, scale=0.015),
        rand(H, 4096, scale=0.015),
        rand(16384, H, scale=0.015),
        rand(H, 8192, scale=0.011),
    ]
    packed = [
        *weights[:4],
        *(
            w.view(w.shape[0] // 16, 16, w.shape[1] // 16, 8, 2)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            for w in weights[4:]
        ),
    ]
    pages = (count + 63) // 64
    cache_pages = (pages + 4) * (2 if owners == "distinct" else 1)
    base = rand(cache_pages, 64, 2, 256, scale=0.3).permute(0, 2, 1, 3)
    if cache_order == "BHND":
        base = base.contiguous()
    cache = base.clone()
    table = torch.randperm(pages + 4, device=device, dtype=torch.int32)[:pages]
    blocks = table.expand(rows, -1).contiguous()
    if owners == "distinct":
        blocks[3].copy_(
            pages
            + 4
            + torch.randperm(pages + 4, device=device, dtype=torch.int32)[
                :pages
            ]
        )
    positions = torch.arange(count - rows, count, device=device)
    slots = (
        blocks[torch.arange(rows, device=device), positions // 64].long() * 64
        + positions % 64
    )
    lengths = torch.full((rows,), count, dtype=torch.int32, device=device)
    hidden, residual = rand(rows, H), rand(rows, H, scale=0.2)
    out, res = torch.empty_like(hidden), torch.empty_like(hidden)
    scratch = workspace(1, device, DTYPE)
    control = torch.zeros(
        native.control_words(), dtype=torch.int32, device=device
    )
    owner = torch.classes.minicpm5_mono.Region(native.mailbox_bytes())
    angles = torch.arange(count, device=device)[:, None] * 5e6 ** (
        -torch.arange(64, device=device, dtype=torch.float32)[None, :] / 64
    )
    rope = torch.cat((angles.cos(), angles.sin()), -1).to(DTYPE)

    def run():
        native.draft_decode(
            hidden,
            residual,
            positions,
            blocks,
            lengths,
            slots,
            out,
            res,
            scratch,
            control,
            packed,
            cache,
            owner,
            0,
            rope,
        )

    def rotate(x):
        a, b = x.float().chunk(2, -1)
        c, s = rope[positions].float().unsqueeze(1).chunk(2, -1)
        return torch.cat((a * c - b * s, b * c + a * s), -1).to(DTYPE)

    run()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=torch.xpu.current_stream()):
        run()
    for iteration, length in enumerate((32775, 65, 129, 32775)):
        base.copy_(rand(cache_pages, 64, 2, 256, scale=0.3).permute(0, 2, 1, 3))
        for head in range(2):
            base[:, head, :, D:] = (iteration + 1) * (head + 1) * 0.25
        cache.copy_(base)
        lengths.fill_(length)
        positions.copy_(torch.arange(length - rows, length, device=device))
        if length_pattern == "mixed":
            shorter = length - (64 if length > 65 else 16)
            lengths[3] = shorter
            positions[3] = shorter - (rows - 3)
        slots.copy_(
            blocks[torch.arange(rows, device=device), positions // 64].long()
            * 64
            + positions % 64
        )
        assert bool(((positions >= 0) & (positions < lengths)).all())
        assert slots.unique().numel() == rows
        hidden.copy_(rand(rows, H))
        graph.replay()
        torch.xpu.synchronize()
        assert int(control[native.health_offset()]) == 0
        assert owner.health()[0].item() == 0
        wnorm, post, qnorm, knorm, qkv, o, gu, down = weights
        added = hidden + residual
        q, k, v = F.linear(norm(added, wnorm), qkv).split((4096, 256, 256), -1)
        q = rotate(norm(q.reshape(rows, 32, D), qnorm))
        k = rotate(norm(k.reshape(rows, 2, D), knorm))
        expected_cache = base.clone()
        expected_cache[slots // 64, :, slots % 64, :D] = k
        expected_cache[slots // 64, :, slots % 64, D:] = v.reshape(rows, 2, D)
        attention = []
        for row in range(rows):
            row_length = int(lengths[row])
            history = expected_cache[
                blocks[row, : (row_length + 63) // 64].long()
            ]
            history = history.permute(0, 2, 1, 3).reshape(-1, 2, 256)[
                :row_length
            ]
            history = history.repeat_interleave(16, dim=1)
            logits = torch.einsum(
                "hd,khd->hk", q[row].float(), history[:, :, :D].float()
            )
            probs = (logits * D**-0.5).softmax(-1)
            attention.append(
                torch.einsum("hk,khd->hd", probs, history[:, :, D:].float())
            )
        attn = torch.stack(attention)
        expected_res = F.linear(attn.reshape(rows, 4096).to(DTYPE), o) + added
        gate, up = F.linear(norm(expected_res, post), gu).chunk(2, -1)
        expected = F.linear((F.silu(gate.float()) * up.float()).to(DTYPE), down)
        torch.testing.assert_close(
            cache, expected_cache, atol=0.0015, rtol=0.004
        )
        torch.testing.assert_close(res, expected_res, atol=0.003, rtol=0.008)
        torch.testing.assert_close(out, expected, atol=0.003, rtol=0.008)


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("length", [32769, 65536, 131072, 524288])
def test_long_context_matches_independent_math_and_cache(native, dense, length):
    """Long contexts retain the original layer math and KV-write contracts."""
    if (
        not hasattr(native, "max_context_len")
        or native.max_context_len() < length
    ):
        pytest.skip("native library predates long-context support")
    test_real_shape_layer_matches_independent_math_and_cache(
        native, dense, 1, "BNHD", length
    )


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize(
    "rows", [r for r in ROW_CASES if r not in (1, 2, 4, 8)]
)
@pytest.mark.parametrize("length", [129, 32768])
def test_exact_rows_match_padded_bucket_bitwise(native, dense, rows, length):
    """Removing dummy requests must preserve real requests' arithmetic."""
    torch.manual_seed(42)
    device = "xpu:0"
    padded = 1 << (rows - 1).bit_length()
    weights = make_weights(dense, device)
    hidden, initial_cache, blocks, slots, positions, lengths = inputs(
        padded, length, device
    )
    slots[rows:] = -1
    lengths[rows:] = 0
    positions[rows:] = 0
    residual = torch.randn_like(hidden) * 0.1
    results = []
    for count in (padded, rows):
        cache = initial_cache.clone()
        out = torch.empty_like(hidden[:count])
        res = torch.empty_like(out)
        scratch = workspace(1, device, DTYPE)
        control = torch.zeros(
            native.control_words(), dtype=torch.int32, device=device
        )
        owner = torch.classes.minicpm5_mono.Region(native.mailbox_bytes())
        native.decode(
            hidden[:count],
            residual[:count],
            positions[:count],
            blocks[:count],
            lengths[:count],
            slots[:count],
            out,
            res,
            scratch,
            control,
            weights,
            cache,
            owner,
            0,
            1,
            dense,
        )
        torch.xpu.synchronize()
        assert int(control[native.health_offset()]) == 0
        result = [out[:rows].clone(), res[:rows].clone(), cache]
        if not dense:
            result.extend(
                [scratch[6][:rows].clone(), scratch[7][:rows].clone()]
            )
        results.append(result)
    for expected, actual in zip(*results):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("cache_order", ["BNHD", "BHND"])
def test_context_chunks_keep_unequal_requests_and_padding_separate(
    native, dense, cache_order
):
    """Exercise distinct chunk/page boundaries while the full layer uses TP1."""
    test_real_shape_layer_matches_independent_math_and_cache(
        native, dense, 8, cache_order, 32768, mixed_lengths=True
    )


@pytest.mark.parametrize(
    "bad",
    [
        "slot",
        "block",
        "tail_block",
        "length",
        "position",
        "rope_missing",
        "rope_short",
    ],
)
def test_invalid_metadata_stops_before_any_cache_or_output_write(native, bad):
    if bad.startswith("rope_") and not hasattr(native, "rotary_cache_version"):
        pytest.skip("native library predates canonical RoPE cache")
    device = "xpu:0"
    weights = make_weights(True, device)
    length = 32769 if bad.startswith("rope_") else 129
    hidden, cache, blocks, slots, positions, lengths = inputs(1, length, device)
    kwargs = {}
    if bad == "slot":
        slots[0] = cache.shape[0] * 64
    elif bad == "block":
        blocks[0, 0] = -1
    elif bad == "tail_block":
        blocks[0, -1] = cache.shape[0]
    elif bad == "length":
        lengths[0] = (
            native.max_context_len()
            if hasattr(native, "max_context_len")
            else 32768
        ) + 1
    elif bad == "position":
        positions[0] -= 1
    elif bad == "rope_short":
        kwargs = rotary_cache_kwargs(native, length, device)
        kwargs["rope_cache"] = kwargs["rope_cache"][:-1]
    before = cache.clone()
    out, res = torch.full_like(hidden, 17), torch.full_like(hidden, 19)
    control = torch.zeros(
        native.control_words(), dtype=torch.int32, device=device
    )
    owner = torch.classes.minicpm5_mono.Region(native.mailbox_bytes())
    native.decode(
        hidden,
        None,
        positions,
        blocks,
        lengths,
        slots,
        out,
        res,
        workspace(1, device, DTYPE),
        control,
        weights,
        cache,
        owner,
        0,
        1,
        True,
        **kwargs,
    )
    torch.xpu.synchronize()
    assert int(control[native.health_offset()]) == 1
    assert torch.equal(cache, before)
    assert bool((out == 17).all()) and bool((res == 19).all())
    if hasattr(native, "runtime_variant") and native.runtime_variant() == (
        "reference-shared-health-parallel-page-validation"
    ):
        assert int(owner.health()[0]) == 1
        # A different decoder layer has a fresh local control but shares the
        # failed device. It must stop before touching cache or either output.
        if bad == "block":
            blocks[0, 0] = 0
            next_control = torch.zeros_like(control)
            native.decode(
                hidden,
                None,
                positions,
                blocks,
                lengths,
                slots,
                out,
                res,
                workspace(1, device, DTYPE),
                next_control,
                weights,
                cache,
                owner,
                0,
                1,
                True,
            )
            torch.xpu.synchronize()
            assert torch.equal(cache, before)
            assert bool((out == 17).all()) and bool((res == 19).all())
            with pytest.raises(RuntimeError, match="shared health must bind"):
                owner.set_health(torch.zeros_like(owner.health()))


def test_output_alias_is_rejected_before_submission(native):
    device = "xpu:0"
    weights = make_weights(True, device)
    hidden, cache, blocks, slots, positions, lengths = inputs(1, 129, device)
    before = cache.clone()
    control = torch.zeros(
        native.control_words(), dtype=torch.int32, device=device
    )
    owner = torch.classes.minicpm5_mono.Region(native.mailbox_bytes())
    with pytest.raises(RuntimeError, match="overlaps an input"):
        native.decode(
            hidden,
            None,
            positions,
            blocks,
            lengths,
            slots,
            hidden,
            torch.empty_like(hidden),
            workspace(1, device, DTYPE),
            control,
            weights,
            cache,
            owner,
            0,
            1,
            True,
        )
    assert torch.equal(cache, before)


@pytest.mark.parametrize("dense", [True, False])
def test_mixed_row_graphs_share_control_and_preserve_math(native, dense):
    """Changing graph buckets must preserve one layer's phase/epoch state."""
    torch.manual_seed(42)
    device = "xpu:0"
    weights = make_weights(dense, device)
    scratch = workspace(1, device, DTYPE)
    control = torch.zeros(
        native.control_words(), dtype=torch.int32, device=device
    )
    owner = torch.classes.minicpm5_mono.Region(native.mailbox_bytes())
    cases = {}
    for rows in ROW_CASES:
        hidden, cache, blocks, slots, positions, lengths = inputs(
            rows, 129, device
        )
        if rows > 1:
            slots[-1] = -1
        out, res = torch.empty_like(hidden), torch.empty_like(hidden)
        case = (hidden, cache, blocks, slots, positions, lengths, out, res)

        def invoke(values=case):
            h, kv, bt, sl, pos, lens, output, residual_out = values
            native.decode(
                h,
                None,
                pos,
                bt,
                lens,
                sl,
                output,
                residual_out,
                scratch,
                control,
                weights,
                kv,
                owner,
                0,
                1,
                dense,
            )

        invoke()
        torch.xpu.synchronize()
        assert int(control[native.health_offset()]) == 0, (
            rows,
            control.cpu().tolist()[
                native.health_offset() - 1 : native.health_offset() + 5
            ],
        )
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, stream=torch.xpu.current_stream()):
            invoke()
        torch.xpu.synchronize()
        cases[rows] = (case, graph)
    initial_epoch = int(control[native.health_offset() - 1])
    sequence = [*ROW_CASES, *reversed(ROW_CASES), ROW_CASES[0]]
    for index, rows in enumerate(sequence):
        (hidden, cache, blocks, slots, positions, lengths, out, res), graph = (
            cases[rows]
        )
        reference_cache = cache.clone()
        expected, expected_res = reference(
            hidden,
            None,
            weights,
            reference_cache,
            blocks,
            lengths,
            positions,
            slots,
            dense,
        )
        graph.replay()
        torch.xpu.synchronize()
        assert int(control[native.health_offset()]) == 0, control.cpu().tolist()
        assert (
            int(control[native.health_offset() - 1])
            == initial_epoch + index + 1
        )
        torch.testing.assert_close(cache, reference_cache, atol=0.02, rtol=0.02)
        torch.testing.assert_close(res, expected_res, atol=0.05, rtol=0.03)
        torch.testing.assert_close(out, expected, atol=0.05, rtol=0.03)


@pytest.mark.parametrize("length", [129, 65537])
def test_full_graph_replay_reads_new_inputs_and_cache_metadata(native, length):
    if length > 32768 and not hasattr(native, "rotary_cache_version"):
        pytest.skip("native library predates canonical RoPE cache")
    device = "xpu:0"
    weights = make_weights(True, device)
    hidden, cache, blocks, slots, positions, lengths = inputs(1, length, device)
    kwargs = rotary_cache_kwargs(native, length + 1, device)
    out, res = torch.empty_like(hidden), torch.empty_like(hidden)
    scratch = workspace(1, device, DTYPE)
    control = torch.zeros(
        native.control_words(), dtype=torch.int32, device=device
    )
    owner = torch.classes.minicpm5_mono.Region(native.mailbox_bytes())

    def launch():
        native.decode(
            hidden,
            None,
            positions,
            blocks,
            lengths,
            slots,
            out,
            res,
            scratch,
            control,
            weights,
            cache,
            owner,
            0,
            1,
            True,
            **kwargs,
        )

    launch()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=torch.xpu.current_stream()):
        launch()
    hidden.copy_(torch.randn_like(hidden))
    positions.add_(1)
    lengths.add_(1)
    slots.add_(1)
    expected_cache = cache.clone()
    expected, expected_res = reference(
        hidden,
        None,
        weights,
        expected_cache,
        blocks,
        lengths,
        positions,
        slots,
        True,
    )
    graph.replay()
    torch.xpu.synchronize()
    assert int(control[native.health_offset()]) == 0
    torch.testing.assert_close(out, expected, atol=0.05, rtol=0.03)
    torch.testing.assert_close(res, expected_res, atol=0.05, rtol=0.03)
    torch.testing.assert_close(cache, expected_cache, atol=0.02, rtol=0.02)


@pytest.mark.parametrize(
    "bad", ["dtype", "width", "strided", "empty", "overlap"]
)
def test_rope_cache_contract_is_checked_before_submission(native, bad):
    if not hasattr(native, "rotary_cache_version"):
        pytest.skip("native library predates canonical RoPE cache")
    device = "xpu:0"
    weights = make_weights(True, device)
    hidden, cache, blocks, slots, positions, lengths = inputs(1, 129, device)
    out, res = torch.full_like(hidden, 17), torch.full_like(hidden, 19)
    if bad == "overlap":
        rope = out.reshape(16, D)
    elif bad == "dtype":
        rope = torch.zeros(16, D, dtype=torch.float32, device=device)
    elif bad == "width":
        rope = torch.zeros(16, D - 1, dtype=DTYPE, device=device)
    elif bad == "strided":
        rope = torch.zeros(16, 2 * D, dtype=DTYPE, device=device)[:, ::2]
    else:
        rope = torch.empty(0, D, dtype=DTYPE, device=device)
    before = cache.clone()
    control = torch.zeros(
        native.control_words(), dtype=torch.int32, device=device
    )
    owner = torch.classes.minicpm5_mono.Region(native.mailbox_bytes())
    message = "overlaps an input" if bad == "overlap" else "RoPE cache"
    with pytest.raises(RuntimeError, match=message):
        native.decode(
            hidden,
            None,
            positions,
            blocks,
            lengths,
            slots,
            out,
            res,
            workspace(1, device, DTYPE),
            control,
            weights,
            cache,
            owner,
            0,
            1,
            True,
            rope_cache=rope,
        )
    assert torch.equal(cache, before)
    assert bool((out == 17).all()) and bool((res == 19).all())
    assert torch.count_nonzero(control).item() == 0


@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize(
    "pattern", ["all_equal", "one_lane", "tied_lanes", "large_bias"]
)
def test_router_top16_matches_stable_fp32_order(native, rows, pattern):
    """Check tie order and local candidate exhaustion on the real layer."""
    torch.manual_seed(42)
    device = "xpu:0"
    weights = make_weights(False, device)
    weights[6].zero_()
    bias = torch.zeros(E, dtype=torch.float32)
    if pattern == "one_lane":
        bias[:10] = torch.arange(16, 6, -1, dtype=torch.float32)
    elif pattern == "tied_lanes":
        bias[::10] = 1.0
    elif pattern == "large_bias":
        bias = 256 + torch.arange(E, dtype=torch.float32) / 32768
    weights[7].copy_(bias)
    expected = torch.argsort(bias + 0.5, descending=True, stable=True)[:TOP]
    hidden, cache, blocks, slots, positions, lengths = inputs(rows, 1, device)
    out, res = torch.empty_like(hidden), torch.empty_like(hidden)
    scratch = workspace(1, device, DTYPE)
    control = torch.zeros(
        native.control_words(), dtype=torch.int32, device=device
    )
    owner = torch.classes.minicpm5_mono.Region(native.mailbox_bytes())
    native.decode(
        hidden,
        None,
        positions,
        blocks,
        lengths,
        slots,
        out,
        res,
        scratch,
        control,
        weights,
        cache,
        owner,
        0,
        1,
        False,
    )
    torch.xpu.synchronize()
    assert int(control[native.health_offset()]) == 0
    torch.testing.assert_close(
        scratch[7][:rows].cpu().long(),
        expected.expand(rows, -1),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        scratch[6][:rows].cpu(),
        torch.full((rows, TOP), 3.66 / TOP),
        atol=1e-7,
        rtol=0,
    )
    assert torch.isfinite(out).all()
