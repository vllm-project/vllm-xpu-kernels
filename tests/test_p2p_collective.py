# SPDX-License-Identifier: Apache-2.0
"""Two-rank tests for the Level Zero IPC p2p all-reduce.

The collective is a peer handshake, so it needs two XPUs and two processes.
The suite runs once in a spawn and each pytest case asserts on one named
result from it. Both ranks build both ranks' inputs from the same CPU seed,
so the reference is an exact local add.
"""

import contextlib
import multiprocessing as mp
import os
import socket
import struct
import tempfile
import traceback
import uuid

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401
from vllm_xpu_kernels import p2p

# The staging slot vLLM provisions.
SLOT_BYTES = 64 * 1024
# Workgroup 0's counter in the region layout documented in p2p_collective.cpp:
# the slots rounded up to a 4 KiB page, then the flag page.
COUNTER_OFFSET = (2 * SLOT_BYTES + 4095) // 4096 * 4096 + 4096

AR_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
AR_CASES = [f"all_reduce/{d}".replace("torch.", "") for d in AR_DTYPES]
OTHER_CASES = [
    "all_reduce/repeated",
    "all_reduce/at_slot_limit",
    "all_reduce/over_slot_raises",
    "all_reduce/empty_is_noop",
    "errors/out_on_other_device",
    "errors/misaligned_region_raises",
    "counter/advances_once_per_launch",
]
ALL_CASES = AR_CASES + OTHER_CASES

# Around the 8-wide vector tail, around one workgroup (2048 elements), and up
# to a full float32 slot; at_slot_limit reaches the workgroup cap.
AR_SIZES = [1, 7, 8, 9, 63, 64, 1023, 2047, 2048, 2049, 5120, 16384]

_TIMEOUT_S = 300.0


def _pair(n, dtype, dev, seed):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(n, generator=gen).to(dtype)
    b = torch.randn(n, generator=gen).to(dtype)
    return a.to(dev), b.to(dev)


def _ref(a, b):
    # What the kernel does: widen to fp32, add once, narrow back.
    return (a.float() + b.float()).to(a.dtype).cpu()


def _exchange(rank, sock_path, barrier, payload, fd):
    """Swap payloads, carrying the process-local dma-buf fd over SCM_RIGHTS."""
    srv = None
    try:
        if rank == 0:
            srv = socket.socket(socket.AF_UNIX)
            srv.settimeout(_TIMEOUT_S)
            srv.bind(sock_path)
            srv.listen(1)
            # Bound before rank 1 is released to connect.
            barrier.wait(timeout=_TIMEOUT_S)
            conn, _ = srv.accept()
        else:
            barrier.wait(timeout=_TIMEOUT_S)
            conn = socket.socket(socket.AF_UNIX)
            conn.settimeout(_TIMEOUT_S)
            conn.connect(sock_path)
        with conn:
            conn.settimeout(_TIMEOUT_S)
            socket.send_fds(conn, [payload], [fd])
            data, fds, _, _ = socket.recv_fds(conn, 1024, 1)
    finally:
        if srv is not None:
            srv.close()
            with contextlib.suppress(OSError):
                os.unlink(sock_path)
    return data, fds[0]


class _Region:
    """This rank's zeroed region, exported, and the peer's, mapped."""

    def __init__(self, rank, dev, sock_path, barrier):
        self.stage = p2p.alloc_region(SLOT_BYTES)
        handle, fd, offset = p2p.export_handle(self.stage.data_ptr())
        # alloc_region gives the region an allocation of its own.
        assert offset == 0, f"region shares an allocation (offset {offset})"
        payload = struct.pack("<Q", offset) + bytes(handle.tolist())
        data, self.peer_fd = _exchange(rank, sock_path, barrier, payload, fd)
        peer_off = struct.unpack_from("<Q", data)[0]
        peer = p2p.open_handle(
            torch.frombuffer(bytearray(data[8:]), dtype=torch.uint8),
            self.peer_fd,
            peer_off,
        )
        self.peer_base = peer - peer_off
        self.fptrs = (p2p.as_fptr(self.stage.data_ptr()), p2p.as_fptr(peer))
        # Release only after both ranks have opened: the peer's open resolves
        # against the export reference.
        barrier.wait(timeout=_TIMEOUT_S)
        p2p.release_handle(handle)

    def all_reduce(self, x, out=None):
        out = torch.empty_like(x) if out is None else out
        torch.ops._xpu_C.xpu_p2p_all_reduce(out, x, *self.fptrs, SLOT_BYTES)
        return out

    def close(self):
        p2p.close_handle(self.peer_base)
        os.close(self.peer_fd)


def _run_rank(rank, sock_path, barrier):
    torch.xpu.set_device(rank)
    dev = torch.device(f"xpu:{rank}")
    reg = _Region(rank, dev, sock_path, barrier)
    res = {}

    def reduce_matches(n, dtype, seed):
        a, b = _pair(n, dtype, dev, seed)
        out = reg.all_reduce((a, b)[rank])
        torch.xpu.synchronize()
        return torch.equal(out.cpu(), _ref(a, b))

    def expect_raise(name, match, fn):
        # Both ranks raise before launching, so later cases stay in step.
        try:
            fn()
            res[name] = (False, "no exception")
        except RuntimeError as e:
            res[name] = (match in str(e), str(e)[:200])

    try:
        for dtype in AR_DTYPES:
            bad = [
                n
                for i, n in enumerate(AR_SIZES)
                if not reduce_matches(n, dtype, 1000 + i)
            ]
            name = f"all_reduce/{dtype}".replace("torch.", "")
            res[name] = (not bad, f"mismatched sizes: {bad}")

        # Many launches walk the counter through both staging slots.
        bad = sum(
            not reduce_matches(
                1 + (i * 131) % (SLOT_BYTES // 2), torch.bfloat16, 3000 + i
            )
            for i in range(300)
        )
        res["all_reduce/repeated"] = (bad == 0, f"{bad}/300 mismatched")

        res["all_reduce/at_slot_limit"] = (
            reduce_matches(SLOT_BYTES // 2, torch.bfloat16, 5000),
            "",
        )

        big = torch.zeros(SLOT_BYTES // 2 + 1, dtype=torch.bfloat16, device=dev)
        expect_raise(
            "all_reduce/over_slot_raises",
            "staging slot",
            lambda: reg.all_reduce(big),
        )

        x = torch.zeros(256, dtype=torch.bfloat16, device=dev)
        wrong = torch.empty_like(x, device=f"xpu:{1 - rank}")
        expect_raise(
            "errors/out_on_other_device",
            "same XPU",
            lambda: reg.all_reduce(x, wrong),
        )

        my_region, peer_region = reg.fptrs
        expect_raise(
            "errors/misaligned_region_raises",
            "xpu_p2p_alloc_region",
            lambda: torch.ops._xpu_C.xpu_p2p_all_reduce(
                torch.empty_like(x), x, my_region + 1, peer_region, SLOT_BYTES
            ),
        )

        empty = torch.empty(0, dtype=torch.bfloat16, device=dev)
        ok = reg.all_reduce(empty).numel() == 0
        res["all_reduce/empty_is_noop"] = (ok, "")

        # The device-side counter is what keeps a replayed launch making
        # progress, so it must advance exactly once per launch.
        counter = reg.stage[COUNTER_OFFSET : COUNTER_OFFSET + 4]
        counter = counter.view(torch.int32)
        torch.xpu.synchronize()
        before = int(counter.item())
        for i in range(10):
            reduce_matches(5120, torch.bfloat16, 9000 + i)
        advanced = int(counter.item()) - before
        res["counter/advances_once_per_launch"] = (
            advanced == 10,
            f"advanced by {advanced}, expected 10",
        )
    finally:
        torch.xpu.synchronize()
        reg.close()

    return res


def _worker(rank, sock_path, barrier, q):
    try:
        q.put((rank, None, _run_rank(rank, sock_path, barrier)))
    except BaseException:
        q.put((rank, traceback.format_exc(), {}))


def _unavailable_reason(devices):
    """Why the path cannot run here, or None if it can.

    Only the documented multi-tile refusal counts as unavailable; any other
    export failure is a regression and raises.
    """
    count = torch.xpu.device_count()
    if count < devices:
        return f"needs {devices} XPUs, found {count}"
    probe = torch.empty(4096, dtype=torch.uint8, device="xpu:0")
    try:
        handle, _fd, _offset = p2p.export_handle(probe.data_ptr())
    except RuntimeError as exc:
        if "single-tile" not in str(exc):
            raise
        return f"Level Zero IPC export is unavailable here: {exc}"
    p2p.release_handle(handle)
    return None


@pytest.fixture(scope="module")
def p2p_results():
    reason = _unavailable_reason(2)
    if reason:
        pytest.skip(reason)

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    q = ctx.Queue()
    sock_path = os.path.join(
        tempfile.gettempdir(), f"vllm_xpu_p2p_test_{uuid.uuid4().hex[:8]}.sock"
    )
    procs = [
        ctx.Process(target=_worker, args=(r, sock_path, barrier, q))
        for r in range(2)
    ]
    for p in procs:
        p.start()

    out = {}
    try:
        for _ in range(2):
            rank, err, res = q.get(timeout=_TIMEOUT_S)
            if err is not None:
                pytest.fail(f"rank {rank} raised:\n{err}")
            out[rank] = res
        for p in procs:
            p.join(timeout=60)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        if os.path.exists(sock_path):
            os.unlink(sock_path)

    for p in procs:
        assert p.exitcode == 0, f"rank exited with {p.exitcode}"

    # A case passes only if it passed on both ranks.
    merged = {}
    for name in out[0]:
        failed = [r for r in (0, 1) if not out[r][name][0]]
        merged[name] = (
            (False, f"rank {failed[0]}: {out[failed[0]][name][1]}")
            if failed
            else (True, "")
        )
    return merged


@pytest.mark.parametrize("case", ALL_CASES)
def test_p2p_collective(p2p_results, case):
    assert case in p2p_results, f"rank never reached {case}"
    ok, detail = p2p_results[case]
    assert ok, detail


def test_export_release_does_not_leak_fds():
    """Each export creates a dma-buf fd; releasing must reclaim it.

    This relies on the driver closing the fd inside zeMemPutIpcHandle, which
    the Level Zero spec words as *may*. If it starts failing, the exporter
    should close the fd itself.
    """
    reason = _unavailable_reason(1)
    if reason:
        pytest.skip(reason)

    buf = torch.empty(1 << 20, dtype=torch.uint8, device="xpu:0")
    # The first exports can set up driver state that is not per handle.
    for _ in range(2):
        p2p.release_handle(p2p.export_handle(buf.data_ptr())[0])

    cycles = 32
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(cycles):
        p2p.release_handle(p2p.export_handle(buf.data_ptr())[0])
    grew = len(os.listdir("/proc/self/fd")) - before

    # A leak is one fd per cycle; the slack absorbs unrelated one-off fds.
    assert grew < cycles // 4, f"{grew} fds leaked over {cycles} cycles"
