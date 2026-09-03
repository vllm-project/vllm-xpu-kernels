# SPDX-License-Identifier: Apache-2.0
"""Two-rank tests for the Level Zero IPC p2p collectives.

These need two XPUs and two processes: the collectives are a peer handshake,
so a single process cannot exercise them. The whole two-rank suite runs in
one spawn (importing torch and the extension twice is the expensive part)
and each pytest case then asserts on one named result from it.

The suite skips, rather than fails, where the path is genuinely unavailable:
fewer than two XPUs, or a device whose allocations cannot be exported over
Level Zero IPC at all (a multi-tile part under
ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE). Both are probed, not assumed.

Neither rank uses a real collective library for its reference. Both ranks
build *both* ranks' inputs from the same fixed CPU seed, feed their own to
the kernel, and reduce locally -- so the reference is exact, deterministic
and dependency-free.
"""

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

# Matches the capacities the vLLM communicator provisions: the largest
# all-reduce a captured decode batch issues (512 x 8192 x 2 B) and the
# all-gather shard at the same token count (half of it).
AR_SLOT_BYTES = 8 << 20
AG_SLOT_BYTES = 4 << 20

AR_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
AG_DTYPES = [
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.int64,
    torch.int32,
    torch.uint8,
    torch.bool,
]

# Case names, in the order the worker produces them. Kept at module scope so
# pytest can parametrize on them at collection time.
AR_CASES = [f"all_reduce/{d}".replace("torch.", "") for d in AR_DTYPES]
AG_CASES = [f"all_gather/{d}".replace("torch.", "") for d in AG_DTYPES]
OTHER_CASES = [
    "all_reduce/repeated",
    "all_gather/repeated",
    "all_reduce/at_slot_limit",
    "all_gather/at_slot_limit",
    "all_reduce/over_slot_raises",
    "all_gather/over_slot_raises",
    "all_reduce/empty_is_noop",
    "errors/out_on_other_device",
    "interleaved/reduce_and_gather",
    "counters/advance_once_per_launch",
]
ALL_CASES = AR_CASES + AG_CASES + OTHER_CASES

# Element counts covering the eager decode sizes, the sizes that straddle the
# 8-wide vector tail, and the multi-workgroup regime.
AR_SIZES = [1, 7, 8, 9, 63, 64, 1023, 2048, 5120, 100_000, 1 << 20]
AG_SIZES = [1, 3, 15, 16, 17, 255, 4096, 65_536, 1 << 19]

_TIMEOUT_S = 300.0


# ---------------------------------------------------------------------------
# rank-side helpers
# ---------------------------------------------------------------------------


def _rand(shape, dtype, gen):
    """Deterministic CPU tensor; both ranks generate both ranks' inputs."""
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=torch.float32, generator=gen).to(dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=gen).bool()
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=dtype, generator=gen)
    return torch.randint(-1000, 1000, shape, dtype=dtype, generator=gen)


def _pair(shape, dtype, dev, seed):
    gen = torch.Generator().manual_seed(seed)
    a = _rand(shape, dtype, gen)
    b = _rand(shape, dtype, gen)
    return a.to(dev), b.to(dev)


def _ar_ref(a, b):
    # A single two-operand add rounds once, which is what the kernel does:
    # widen to fp32, add, narrow back.
    return (a.float() + b.float()).to(a.dtype)


def _exchange_handle(rank, sock_path, barrier, stage_ptr):
    """Trade Level Zero IPC handles, passing the dma-buf fd over SCM_RIGHTS.

    The fd inside a ze_ipc_mem_handle_t is process-local, so it has to travel
    as a real file descriptor, not as bytes.
    """
    handle, fd, offset = p2p.export_handle(stage_ptr)
    payload = struct.pack("<Q", offset) + bytes(handle.tolist())

    srv = None
    try:
        if rank == 0:
            srv = socket.socket(socket.AF_UNIX)
            srv.settimeout(_TIMEOUT_S)
            srv.bind(sock_path)
            srv.listen(1)
            # Bind before releasing rank 1, so its connect cannot race it.
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
            os.unlink(sock_path)

    peer_off = struct.unpack_from("<Q", data)[0]
    peer_handle = torch.frombuffer(bytearray(data[8:]), dtype=torch.uint8)
    peer_ptr = p2p.open_handle(peer_handle, fds[0], peer_off)
    # `handle` comes back so the caller can release it once BOTH ranks have
    # opened; releasing it here would drop the export reference the peer's
    # open still has to resolve against.
    return peer_ptr, peer_ptr - peer_off, fds[0], handle


class _Region:
    """The shared staging region, laid out identically on both ranks.

    Same offsets on both sides means the peer's slots and flags sit at the
    same offsets of the mapped peer buffer.
    """

    def __init__(self, rank, dev, sock_path, barrier):
        page = p2p.signal_page_bytes()
        self.page = page
        total = 2 * AR_SLOT_BYTES + 2 * AG_SLOT_BYTES + 4 * page
        self.stage = torch.empty(total, dtype=torch.uint8, device=dev)
        # Flags and counters are monotonic sequence numbers that are never
        # reset, so they must start below any seq the kernels will use.
        self.stage.zero_()
        torch.xpu.synchronize()

        (
            self.peer_ptr,
            self.peer_base,
            self.peer_fd,
            local_handle,
        ) = _exchange_handle(rank, sock_path, barrier, self.stage.data_ptr())

        mine = self.stage.data_ptr()
        peer = self.peer_ptr
        ag = 2 * AR_SLOT_BYTES
        f_ar = ag + 2 * AG_SLOT_BYTES
        c_ar = f_ar + page
        f_ag = c_ar + page
        c_ag = f_ag + page

        # Level Zero device addresses sit above 2**63, so they reach the
        # ops as two's-complement int64. Converted once here, not per call.
        self.ar_counter_off = c_ar
        self.ar = tuple(
            p2p.as_fptr(x)
            for x in (mine, peer, mine + f_ar, peer + f_ar, mine + c_ar)
        )
        self.ag = tuple(
            p2p.as_fptr(x)
            for x in (
                mine + ag,
                peer + ag,
                mine + f_ag,
                peer + f_ag,
                mine + c_ag,
            )
        )

        # Both ranks must have zeroed their flags before either launches a
        # kernel that signals into them.
        barrier.wait(timeout=_TIMEOUT_S)

        # Past that barrier both ranks have opened, so the export reference
        # has done its job and can go. Not releasing it would keep the
        # dma-buf fd and the driver's export bookkeeping for the life of the
        # process -- bounded in production, but this test builds and tears
        # down regions repeatedly. The fd is not closed separately: the
        # driver may close it here.
        p2p.release_handle(local_handle)

    def all_reduce(self, x):
        out = torch.empty_like(x)
        my_stage, peer_stage, lf, pf, ctr = self.ar
        torch.ops._xpu_C.xpu_p2p_all_reduce(
            out, x, my_stage, peer_stage, lf, pf, ctr, AR_SLOT_BYTES
        )
        return out

    def all_gather(self, x, rank):
        out = torch.empty(
            (2 * x.shape[0],) + tuple(x.shape[1:]),
            dtype=x.dtype,
            device=x.device,
        )
        my_stage, peer_stage, lf, pf, ctr = self.ag
        torch.ops._xpu_C.xpu_p2p_all_gather(
            out, x, my_stage, peer_stage, lf, pf, ctr, AG_SLOT_BYTES, rank
        )
        return out

    def close(self):
        p2p.close_handle(self.peer_base)
        os.close(self.peer_fd)


# ---------------------------------------------------------------------------
# the two-rank body
# ---------------------------------------------------------------------------


def _run_rank(rank, sock_path, barrier):
    torch.xpu.set_device(rank)
    dev = torch.device(f"xpu:{rank}")
    reg = _Region(rank, dev, sock_path, barrier)
    res = {}

    def record(name, ok, detail=""):
        res[name] = (bool(ok), detail)

    try:
        # --- all-reduce, every dtype, across the size regimes -------------
        for dtype in AR_DTYPES:
            bad = []
            for i, n in enumerate(AR_SIZES):
                a, b = _pair((n,), dtype, dev, 1000 + i)
                out = reg.all_reduce((a, b)[rank])
                torch.xpu.synchronize()
                if not torch.equal(out.cpu(), _ar_ref(a, b).cpu()):
                    bad.append(n)
            name = f"all_reduce/{dtype}".replace("torch.", "")
            record(name, not bad, f"mismatched sizes: {bad}")

        # --- all-gather, every dtype --------------------------------------
        for dtype in AG_DTYPES:
            bad = []
            for i, n in enumerate(AG_SIZES):
                a, b = _pair((n,), dtype, dev, 2000 + i)
                out = reg.all_gather((a, b)[rank], rank)
                torch.xpu.synchronize()
                if not torch.equal(out.cpu(), torch.cat([a, b]).cpu()):
                    bad.append(n)
            name = f"all_gather/{dtype}".replace("torch.", "")
            record(name, not bad, f"mismatched sizes: {bad}")

        # --- repeated back-to-back calls ----------------------------------
        # Exercises the per-workgroup counter and the double-buffer parity:
        # mixed sizes mean different workgroup counts advance at different
        # rates, which both ranks must agree on.
        bad = 0
        for i in range(300):
            n = 1 + (i * 131) % 65536
            a, b = _pair((n,), torch.bfloat16, dev, 3000 + i)
            out = reg.all_reduce((a, b)[rank])
            if not torch.equal(out.cpu(), _ar_ref(a, b).cpu()):
                bad += 1
        record("all_reduce/repeated", bad == 0, f"{bad}/300 mismatched")

        bad = 0
        for i in range(300):
            n = 1 + (i * 977) % 100_000
            a, b = _pair((n,), torch.bfloat16, dev, 4000 + i)
            out = reg.all_gather((a, b)[rank], rank)
            if not torch.equal(out.cpu(), torch.cat([a, b]).cpu()):
                bad += 1
        record("all_gather/repeated", bad == 0, f"{bad}/300 mismatched")

        # --- exactly at the staging slot ----------------------------------
        n = AR_SLOT_BYTES // 2  # bf16
        a, b = _pair((n,), torch.bfloat16, dev, 5000)
        out = reg.all_reduce((a, b)[rank])
        torch.xpu.synchronize()
        record(
            "all_reduce/at_slot_limit",
            torch.equal(out.cpu(), _ar_ref(a, b).cpu()),
        )
        del a, b, out

        n = AG_SLOT_BYTES // 2  # bf16
        a, b = _pair((n,), torch.bfloat16, dev, 6000)
        out = reg.all_gather((a, b)[rank], rank)
        torch.xpu.synchronize()
        record(
            "all_gather/at_slot_limit",
            torch.equal(out.cpu(), torch.cat([a, b]).cpu()),
        )
        del a, b, out

        # --- over the slot must raise, not corrupt ------------------------
        # Both ranks raise, so neither launches and the handshake stays in
        # step: the checks after this must still pass.
        big = torch.zeros(
            AR_SLOT_BYTES // 2 + 8, dtype=torch.bfloat16, device=dev
        )
        try:
            reg.all_reduce(big)
            record("all_reduce/over_slot_raises", False, "no exception")
        except RuntimeError as e:
            record(
                "all_reduce/over_slot_raises",
                "staging slot" in str(e),
                str(e)[:200],
            )
        del big

        big = torch.zeros(
            AG_SLOT_BYTES // 2 + 8, dtype=torch.bfloat16, device=dev
        )
        try:
            reg.all_gather(big, rank)
            record("all_gather/over_slot_raises", False, "no exception")
        except RuntimeError as e:
            record(
                "all_gather/over_slot_raises",
                "staging slot" in str(e),
                str(e)[:200],
            )
        del big

        # --- out on a different XPU must be rejected ----------------------
        # The staging and signal pointers belong to one device and the
        # kernel runs on input's queue, so an out elsewhere would be written
        # from the wrong device. Both ranks raise before launching, so the
        # handshake stays in step and the checks below still pass.
        a, b = _pair((256,), torch.bfloat16, dev, 6500)
        wrong = torch.empty(
            256, dtype=torch.bfloat16, device=torch.device(f"xpu:{1 - rank}")
        )
        my_stage, peer_stage, lf, pf, ctr = reg.ar
        try:
            torch.ops._xpu_C.xpu_p2p_all_reduce(
                wrong,
                (a, b)[rank],
                my_stage,
                peer_stage,
                lf,
                pf,
                ctr,
                AR_SLOT_BYTES,
            )
            record("errors/out_on_other_device", False, "no exception")
        except RuntimeError as e:
            record(
                "errors/out_on_other_device", "same XPU" in str(e), str(e)[:200]
            )
        del wrong

        # --- empty input is a no-op on both ranks -------------------------
        empty = torch.empty(0, dtype=torch.bfloat16, device=dev)
        out = reg.all_reduce(empty)
        torch.xpu.synchronize()
        record("all_reduce/empty_is_noop", out.numel() == 0)

        # --- interleaving the two collectives -----------------------------
        # They share nothing (own slots, own flags, own counter), so their
        # interleaving needs no reasoning -- this pins that down.
        bad = 0
        for i in range(50):
            a, b = _pair((1024,), torch.bfloat16, dev, 7000 + i)
            c, d = _pair((768,), torch.float32, dev, 8000 + i)
            r1 = reg.all_reduce((a, b)[rank])
            g1 = reg.all_gather((c, d)[rank], rank)
            r2 = reg.all_reduce((c, d)[rank])
            g2 = reg.all_gather((a, b)[rank], rank)
            if not (
                torch.equal(r1.cpu(), _ar_ref(a, b).cpu())
                and torch.equal(r2.cpu(), _ar_ref(c, d).cpu())
                and torch.equal(g1.cpu(), torch.cat([c, d]).cpu())
                and torch.equal(g2.cpu(), torch.cat([a, b]).cpu())
            ):
                bad += 1
        record("interleaved/reduce_and_gather", bad == 0, f"{bad}/50 bad")
        # --- the device-side sequence counter -----------------------------
        # This is what makes the kernel graph-capture safe: work-item 0
        # advances a per-workgroup counter in device memory and derives the
        # handshake seq and slot parity from it, so nothing is frozen at
        # capture time. Watch it directly -- workgroup 0 takes part in every
        # launch, so its counter must advance exactly once per call.
        page = reg.page
        off = reg.ar_counter_off
        counters = reg.stage[off : off + page].view(torch.int32)
        torch.xpu.synchronize()
        before = int(counters[0].item())
        n_launches = 10
        for i in range(n_launches):
            a, b = _pair((5120,), torch.bfloat16, dev, 9000 + i)
            out = reg.all_reduce((a, b)[rank])
        torch.xpu.synchronize()
        advanced = int(counters[0].item()) - before
        record(
            "counters/advance_once_per_launch",
            advanced == n_launches,
            f"advanced by {advanced}, expected {n_launches}",
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


# ---------------------------------------------------------------------------
# pytest
# ---------------------------------------------------------------------------


# A visible device is not the same as a usable one.  On a multi-tile XPU
# under ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE the device count is fine and
# every export still refuses, by design, because one Level Zero IPC handle
# cannot cover a multi-tile allocation.  That is an unavailable
# configuration, not a failure, so probe what the path actually needs.
def _export_unavailable_reason():
    """Why IPC export cannot work here, or None if it can.

    Only the documented multi-tile refusal counts as unavailable; any other
    export failure is re-raised, because that would be a real regression and
    skipping past it would hide exactly the bug worth catching.
    """
    if torch.xpu.device_count() < 1:
        return "needs an XPU"
    probe = torch.empty(4096, dtype=torch.uint8, device="xpu:0")
    try:
        handle, _fd, _offset = p2p.export_handle(probe.data_ptr())
    except RuntimeError as exc:
        # Matches the wording of reject_untested_multi_tile_allocation() in
        # csrc/xpu/p2p/p2p_ipc.cpp.
        if "single-tile" not in str(exc):
            raise
        return (
            "Level Zero IPC export is unavailable on this device, so the "
            "p2p collectives cannot run here. This is a deliberate limit, "
            f"not a broken build: {exc}"
        )
    p2p.release_handle(handle)
    return None


def _p2p_unavailable_reason():
    """Why the two-rank collectives cannot run here, or None if they can."""
    count = torch.xpu.device_count()
    if count < 2:
        return f"p2p collectives need two XPUs, found {count}"
    return _export_unavailable_reason()


@pytest.fixture(scope="module")
def p2p_results():
    reason = _p2p_unavailable_reason()
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

    # Both ranks check the same things against the same references; a case
    # passes only if it passed on both.
    merged = {}
    for name in out[0]:
        for rank in (0, 1):
            ok, detail = out[rank][name]
            if not ok:
                merged[name] = (False, f"rank {rank}: {detail}")
                break
        else:
            merged[name] = (True, "")
    return merged


@pytest.mark.parametrize("case", ALL_CASES)
def test_p2p_collective(p2p_results, case):
    assert case in p2p_results, f"rank never reached {case}"
    ok, detail = p2p_results[case]
    assert ok, detail


def test_export_release_does_not_leak_fds():
    """Every export creates a dma-buf fd; releasing must reclaim it.

    Needs no peer, so it runs on a single XPU: exporting and releasing is a
    purely local pair. In production the leak is bounded and harmless -- a
    communicator is built a handful of times per process -- but a test
    process that builds and tears down repeatedly is exactly where an
    unreleased export shows up first, which is why this guards it.

    Note this asserts the driver reclaims the fd inside zeMemPutIpcHandle,
    which the Level Zero spec words as *may*. If it ever fails with a
    per-cycle growth, the fix is for the exporter to close the fd itself,
    not to drop this test.
    """
    reason = _export_unavailable_reason()
    if reason:
        pytest.skip(reason)

    buf = torch.empty(1 << 20, dtype=torch.uint8, device="xpu:0")

    def fd_count():
        return len(os.listdir("/proc/self/fd"))

    cycles = 32
    # Warm up first: the first export can set up driver state that is not
    # per-handle, which would otherwise read as a leak.
    for _ in range(2):
        p2p.release_handle(p2p.export_handle(buf.data_ptr())[0])

    before = fd_count()
    for _ in range(cycles):
        handle, _fd, _offset = p2p.export_handle(buf.data_ptr())
        p2p.release_handle(handle)
    grew = fd_count() - before

    # A leak is one fd per cycle. The slack absorbs unrelated one-off fds
    # (a lazily opened driver node, say) without letting a real leak pass.
    assert grew < cycles // 4, (
        f"{grew} fds leaked over {cycles} export/release cycles"
    )


def test_signal_page_bytes_covers_the_grid():
    """The caller sizes its flag and counter pages from this number."""
    page = p2p.signal_page_bytes()
    assert page > 0
    assert page % 4 == 0
    # One uint32 per workgroup on its own 64-byte line.
    assert page % 64 == 0
