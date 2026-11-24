# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the direct symmetric-memory DCP A2A LSE reduce kernel.

Run `pytest tests/test_direct_dcp_a2a_lse_reduce.py`. Needs at least 2 XPUs.
"""

import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm_xpu_kernels._C  # noqa: F401

HEADS_PER_RANK = 6
HEAD_DIM = 64
# One 128B cache line per signal slot; the kernel infers it from the size.
# Unpadded slots shared a line across devices and lost back-to-back signals.
SIGNAL_STRIDE = 32
MAX_NUM_TOKENS = 32
WORKER_TIMEOUT_S = 90

pytestmark = [
    pytest.mark.skipif(torch.xpu.device_count() < 2,
                       reason="Need at least 2 XPUs."),
    pytest.mark.skipif(
        not hasattr(torch.ops._C, "direct_dcp_a2a_lse_reduce"),
        reason="direct_dcp_a2a_lse_reduce is not built.",
    ),
]


def _lse_weighted_combine(outputs: torch.Tensor, lses: torch.Tensor,
                          is_lse_base_on_e: bool) -> torch.Tensor:
    """Reference. outputs: [N, T, H, D] (float), lses: [N, T, H]."""
    lses = torch.where(torch.isnan(lses) | torch.isinf(lses),
                       torch.full_like(lses, float("-inf")), lses)
    if is_lse_base_on_e:
        lses = lses * math.log2(math.e)
    lse_max = lses.max(dim=0).values
    lse_max = torch.where(lse_max == float("-inf"),
                          torch.zeros_like(lse_max), lse_max)
    weights = torch.exp2(lses - lse_max)
    total = weights.sum(dim=0)
    weights = weights / torch.where(total > 0, total,
                                    torch.ones_like(total))
    return (outputs * weights.unsqueeze(-1)).sum(dim=0)


class _Workspace:
    """Symmetric-memory buffers for a single ubatch."""

    def __init__(self, device, world_size, rank, dtype):
        import torch.distributed._symmetric_memory as symm_mem

        self.world_size, self.rank = world_size, rank
        group_name = dist.group.WORLD.group_name
        shapes = {
            "output": ((2, world_size, MAX_NUM_TOKENS, HEADS_PER_RANK,
                        HEAD_DIM), dtype),
            "lse": ((2, world_size, MAX_NUM_TOKENS, HEADS_PER_RANK),
                    torch.float32),
            "signal": ((2, world_size, SIGNAL_STRIDE), torch.int32),
        }
        for name, (shape, dt) in shapes.items():
            storage = symm_mem.empty(shape, device=device, dtype=dt)
            storage.zero_()
            torch.accelerator.synchronize()
            handle = symm_mem.rendezvous(storage, group_name)
            handle.barrier()
            ptrs = torch.tensor(
                [
                    handle.get_buffer(peer, list(shape), dt, 0).data_ptr()
                    for peer in range(world_size)
                ],
                dtype=torch.uint64,
                device=device,
            )
            setattr(self, f"received_{name}", storage)
            setattr(self, f"peer_{name}_ptrs", ptrs)
            setattr(self, f"_{name}_handle", handle)
        self.epoch = torch.zeros(1, dtype=torch.int64, device=device)

    def lse_reduce(self, partial_output, partial_lse, is_lse_base_on_e,
                   seq_lens=None, query_start_loc=None):
        output = partial_output.new_empty(
            (partial_output.shape[0], HEADS_PER_RANK, HEAD_DIM))
        torch.ops._C.direct_dcp_a2a_lse_reduce(
            partial_output,
            partial_lse,
            seq_lens,
            query_start_loc,
            self.peer_output_ptrs,
            self.peer_lse_ptrs,
            self.peer_signal_ptrs,
            self.received_output,
            self.received_lse,
            self.received_signal,
            self.epoch,
            output,
            self.world_size,
            self.rank,
            MAX_NUM_TOKENS,
            is_lse_base_on_e,
        )
        return output


def _gather_reference(partial_output, partial_lse, rank, world_size,
                      is_lse_base_on_e, source_empty=None):
    """All-gather every rank's partials and combine this rank's head slice."""
    out_list = [torch.empty_like(partial_output) for _ in range(world_size)]
    lse_list = [
        torch.empty_like(partial_lse, memory_format=torch.contiguous_format)
        for _ in range(world_size)
    ]
    dist.all_gather(out_list, partial_output.contiguous())
    dist.all_gather(lse_list, partial_lse.contiguous())
    heads = slice(rank * HEADS_PER_RANK, (rank + 1) * HEADS_PER_RANK)
    outputs = torch.stack([o[:, heads, :] for o in out_list]).float()
    lses = torch.stack([v[:, heads] for v in lse_list]).float()
    if source_empty is not None:
        for src in range(world_size):
            outputs[src][source_empty[src]] = 0.0
            lses[src][source_empty[src]] = float("-inf")
    return _lse_weighted_combine(outputs, lses, is_lse_base_on_e)


def _make_inputs(device, rank, world_size, num_tokens, lse_dtype, padded,
                 transposed_lse, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)
    total_heads = world_size * HEADS_PER_RANK
    storage_heads = total_heads + 4 if padded else total_heads
    out_storage = torch.randn(num_tokens, storage_heads, HEAD_DIM,
                              device=device, dtype=torch.bfloat16,
                              generator=gen)
    lse_storage = torch.randn(num_tokens, storage_heads, device=device,
                              dtype=lse_dtype, generator=gen)
    partial_output = out_storage[:, :total_heads, :]
    partial_lse = lse_storage[:, :total_heads]
    if transposed_lse:
        # FlashAttention MLA returns [H, T] viewed as [T, H].
        buf = torch.empty(total_heads, num_tokens, device=device,
                          dtype=lse_dtype)
        buf.copy_(partial_lse.transpose(0, 1))
        partial_lse = buf.transpose(0, 1)
    return partial_output, partial_lse


def _worker(rank, world_size, port, case):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank),
                      WORLD_SIZE=str(world_size),
                      LOCAL_WORLD_SIZE=str(world_size),
                      MASTER_ADDR="localhost", MASTER_PORT=port)
    device = torch.device("xpu", rank)
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(backend="xccl")
    try:
        _run_case(device, rank, world_size, case)
    finally:
        dist.destroy_process_group()


def _run_case(device, rank, world_size, case):
    lse_dtype = getattr(torch, case["lse_dtype"])
    base_e = case["base_e"]
    ws = _Workspace(device, world_size, rank, torch.bfloat16)
    tol = dict(rtol=3e-2, atol=3e-2)

    if case["kind"] == "eager":
        # Repeating covers both epoch parities and slot reuse.
        for it, num_tokens in enumerate((1, 17, MAX_NUM_TOKENS, 5)):
            partial_output, partial_lse = _make_inputs(
                device, rank, world_size, num_tokens, lse_dtype,
                case["padded"], case["transposed_lse"], 100 * it)
            actual = ws.lse_reduce(partial_output, partial_lse, base_e)
            torch.accelerator.synchronize()
            expected = _gather_reference(partial_output, partial_lse, rank,
                                         world_size, base_e)
            torch.testing.assert_close(actual.float(), expected, **tol)
            # Layout alone must not change a single output bit.
            packed = ws.lse_reduce(partial_output, partial_lse.contiguous(),
                                   base_e)
            torch.accelerator.synchronize()
            torch.testing.assert_close(actual, packed, rtol=0, atol=0)

    elif case["kind"] == "empty_seqs":
        # Sequence 0 is empty on every rank, sequence 1 + r on rank r.
        num_seqs = world_size + 2
        query_lens = [1, 3, 2] + [1] * (world_size - 1)
        num_tokens = sum(query_lens)
        partial_output, partial_lse = _make_inputs(
            device, rank, world_size, num_tokens, lse_dtype, False, False, 7)
        partial_output = partial_output.contiguous()
        partial_lse = partial_lse.contiguous()

        def is_empty(seq_idx, src):
            return seq_idx == 0 or seq_idx == 1 + src

        q_lens = torch.tensor(query_lens, dtype=torch.int32, device=device)
        query_start_loc = torch.cat(
            (q_lens.new_zeros(1), q_lens.cumsum(0, dtype=torch.int32)))
        seq_lens = torch.tensor(
            [0 if is_empty(i, rank) else i + 3 for i in range(num_seqs)],
            dtype=torch.int32, device=device)
        empty_rows = torch.repeat_interleave(seq_lens == 0, q_lens)
        # Attention outputs for empty rows are undefined; model them as NaN.
        partial_output[empty_rows] = float("nan")
        partial_lse[empty_rows] = float("nan")

        actual = ws.lse_reduce(partial_output, partial_lse, base_e,
                               seq_lens=seq_lens,
                               query_start_loc=query_start_loc)
        torch.accelerator.synchronize()
        source_empty = [
            torch.tensor([is_empty(i, src) for i in range(num_seqs)],
                         device=device).repeat_interleave(q_lens)
            for src in range(world_size)
        ]
        expected = _gather_reference(partial_output, partial_lse, rank,
                                     world_size, base_e, source_empty)
        all_empty = torch.tensor(
            [all(is_empty(i, r) for r in range(world_size))
             for i in range(num_seqs)],
            device=device).repeat_interleave(q_lens)
        assert not torch.isnan(actual.float()).any()
        assert torch.equal(actual[all_empty],
                           torch.zeros_like(actual[all_empty]))
        torch.testing.assert_close(actual.float(), expected, **tol)

    elif case["kind"] == "graph":
        partial_output, partial_lse = _make_inputs(
            device, rank, world_size, MAX_NUM_TOKENS, lse_dtype, True, False,
            11)
        torch.accelerator.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = ws.lse_reduce(partial_output, partial_lse, base_e)
        for _ in range(3):
            graph.replay()
            if case["sync_between_replays"]:
                torch.accelerator.synchronize()
        torch.accelerator.synchronize()
        expected = _gather_reference(partial_output, partial_lse, rank,
                                     world_size, base_e)
        torch.testing.assert_close(actual.float(), expected, **tol)
    else:
        raise AssertionError(case["kind"])


def _run_distributed(world_size, case):
    port = str(_free_port())
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(r, world_size, port, case))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=WORKER_TIMEOUT_S)
    hung = [p for p in procs if p.is_alive()]
    for p in hung:
        p.kill()
        p.join()
    assert not hung, "worker hung (likely a signal/epoch deadlock)"
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


WORLD_SIZES = [
    pytest.param(2),
    pytest.param(
        4,
        marks=pytest.mark.skipif(torch.xpu.device_count() < 4,
                                 reason="Need at least 4 XPUs."),
    ),
]


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize(
    "lse_dtype,transposed_lse,base_e,padded",
    [
        ("bfloat16", False, False, False),
        ("float32", True, True, True),
    ],
)
def test_direct_dcp_a2a_lse_reduce_matches_reference(
        world_size, lse_dtype, transposed_lse, base_e, padded):
    _run_distributed(
        world_size,
        dict(kind="eager", lse_dtype=lse_dtype, transposed_lse=transposed_lse,
             base_e=base_e, padded=padded),
    )


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_direct_dcp_a2a_lse_reduce_empty_sequences(world_size):
    _run_distributed(
        world_size,
        dict(kind="empty_seqs", lse_dtype="float32", base_e=True),
    )


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("sync_between_replays", [True, False])
def test_direct_dcp_a2a_lse_reduce_graph_replay(world_size,
                                                sync_between_replays):
    _run_distributed(
        world_size,
        dict(kind="graph", lse_dtype="float32", base_e=False,
             sync_between_replays=sync_between_replays),
    )


@pytest.mark.parametrize(
    "signal_stride,max_num_tokens",
    [(1, MAX_NUM_TOKENS), (SIGNAL_STRIDE, 1)],
    ids=["unpadded_signal", "unaligned_partition"],
)
def test_direct_dcp_a2a_lse_reduce_rejects_line_sharing_layout(
        signal_stride, max_num_tokens):
    """Layouts where peers would write a shared cache line must be refused."""
    world_size, device = 2, torch.device("xpu", 0)
    total_heads = world_size * HEADS_PER_RANK
    partial_output = torch.zeros(1, total_heads, HEAD_DIM,
                                 dtype=torch.bfloat16, device=device)
    partial_lse = torch.zeros(1, total_heads, device=device)
    ptrs = torch.zeros(world_size, dtype=torch.uint64, device=device)
    received_output = torch.zeros(2, world_size, max_num_tokens,
                                  HEADS_PER_RANK, HEAD_DIM,
                                  dtype=torch.bfloat16, device=device)
    received_lse = torch.zeros(2, world_size, max_num_tokens,
                               HEADS_PER_RANK, device=device)
    received_signal = torch.zeros(2, world_size, signal_stride,
                                  dtype=torch.int32, device=device)
    with pytest.raises(RuntimeError, match="bytes"):
        torch.ops._C.direct_dcp_a2a_lse_reduce(
            partial_output, partial_lse, None, None, ptrs, ptrs, ptrs,
            received_output, received_lse, received_signal,
            torch.zeros(1, dtype=torch.int64, device=device),
            partial_output.new_empty(1, HEADS_PER_RANK, HEAD_DIM),
            world_size, 0, max_num_tokens, False)


@pytest.mark.parametrize(
    "bad", ["short_ptr_table", "small_staging", "unaligned_output",
            "cpu_epoch"])
def test_direct_dcp_a2a_lse_reduce_rejects_unsafe_arguments(bad):
    """Arguments that would make the kernels access memory out of bounds,
    misaligned, or on the wrong device must be refused."""
    world_size, device = 2, torch.device("xpu", 0)
    total_heads = world_size * HEADS_PER_RANK
    args = dict(
        partial_output=torch.zeros(1, total_heads, HEAD_DIM,
                                   dtype=torch.bfloat16, device=device),
        partial_lse=torch.zeros(1, total_heads, device=device),
        ptrs=torch.zeros(world_size, dtype=torch.uint64, device=device),
        received_output=torch.zeros(2, world_size, MAX_NUM_TOKENS,
                                    HEADS_PER_RANK, HEAD_DIM,
                                    dtype=torch.bfloat16, device=device),
        received_lse=torch.zeros(2, world_size, MAX_NUM_TOKENS,
                                 HEADS_PER_RANK, device=device),
        received_signal=torch.zeros(2, world_size, SIGNAL_STRIDE,
                                    dtype=torch.int32, device=device),
        epoch=torch.zeros(1, dtype=torch.int64, device=device),
        combined_output=torch.empty(1, HEADS_PER_RANK, HEAD_DIM,
                                    dtype=torch.bfloat16, device=device),
    )
    if bad == "short_ptr_table":
        args["ptrs"] = args["ptrs"][:1]
    elif bad == "small_staging":
        args["received_output"] = args["received_output"][:1]
    elif bad == "unaligned_output":
        storage = torch.empty(1 + HEADS_PER_RANK * HEAD_DIM,
                              dtype=torch.bfloat16, device=device)
        args["combined_output"] = storage[1:].view(1, HEADS_PER_RANK,
                                                   HEAD_DIM)
    else:
        args["epoch"] = args["epoch"].cpu()
    ptrs = args["ptrs"]
    with pytest.raises(RuntimeError):
        torch.ops._C.direct_dcp_a2a_lse_reduce(
            args["partial_output"], args["partial_lse"], None, None, ptrs,
            ptrs, ptrs, args["received_output"], args["received_lse"],
            args["received_signal"], args["epoch"], args["combined_output"],
            world_size, 0, MAX_NUM_TOKENS, False)
