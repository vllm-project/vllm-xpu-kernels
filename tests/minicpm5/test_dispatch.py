# SPDX-License-Identifier: Apache-2.0
"""Availability and split decode contracts without model checkpoints."""

from pathlib import Path

import pytest
import torch

from vllm_xpu_kernels import minicpm5_mono

HAS_EXTENSION = any(
    Path(minicpm5_mono.__file__).parent.glob("_minicpm5_mono_C*.so")
)


@pytest.fixture(scope="module")
def native():
    if not HAS_EXTENSION:
        pytest.skip("The optional MiniCPM5 extension is absent")
    return minicpm5_mono.load()


def test_registration_matches_installed_extension():
    if HAS_EXTENSION:
        assert hasattr(minicpm5_mono.load(), "supports_device")
    else:
        with pytest.raises(ImportError, match="MINICPM5_MONO_KERNELS_ENABLED"):
            minicpm5_mono.load()
        assert not hasattr(torch.ops.minicpm5_mono, "supported")


def test_device_probe(native):
    count = torch.xpu.device_count()
    assert not native.supports_device(-1)
    assert not native.supports_device(count)
    # The Python device count can use Sysman when SYCL exposes no devices.
    try:
        original = torch.xpu.current_device()
    except RuntimeError as error:
        assert "No XPU devices are available" in str(error)
        assert not native.supported()
        assert all(not native.supports_device(index) for index in range(count))
        return

    supports = [native.supports_device(index) for index in range(count)]
    assert torch.xpu.current_device() == original
    for index, expected in enumerate(supports):
        with torch.xpu.device(index):
            assert native.supported() == expected
            if not expected:
                with pytest.raises(RuntimeError, match="require Xe2"):
                    torch.classes.minicpm5_mono.Region(native.mailbox_bytes())


@pytest.mark.parametrize(
    ("op_name", "wrong_rows", "match"),
    [
        ("decode_tail_rows4", 8, "post-attention activation"),
        ("decode_tail", 4, "post-attention activation"),
        ("decode_pre_o", 4, "gated attention"),
        ("decode_pre_o_rows4", 8, "gated attention"),
        ("decode_post_o", 4, "O projection partial"),
    ],
)
def test_split_rejects_wrong_physical_rows(native, op_name, wrong_rows, match):
    if not native.supports_device(0) or not hasattr(native, op_name):
        pytest.skip("Requires the qualified Xe2 split decoder")
    with torch.xpu.device(0):
        activation = torch.empty(
            (wrong_rows, 2048), dtype=torch.float16, device="xpu"
        )
        slots = torch.empty((wrong_rows,), dtype=torch.int64, device="xpu")
        control = torch.zeros(
            native.control_words(), dtype=torch.int32, device="xpu"
        )
        owner = torch.classes.minicpm5_mono.Region(native.mailbox_bytes())
        if op_name in {"decode_pre_o", "decode_pre_o_rows4", "decode_post_o"}:
            args = (activation, activation, slots, activation, activation)
        else:
            args = (activation, slots, activation)
        with pytest.raises(RuntimeError, match=match):
            getattr(native, op_name)(*args, [], control, [], owner, 0)
        assert torch.count_nonzero(control).item() == 0


@pytest.mark.parametrize("case", ["wrong_device", "misaligned_cache"])
def test_draft_rejects_invalid_input_before_submission(native, case):
    if not native.supports_device(0) or not hasattr(native, "draft_decode"):
        pytest.skip("Requires the qualified Xe2 draft decoder")
    wrong_device = case == "wrong_device"
    if wrong_device and torch.xpu.device_count() < 2:
        pytest.skip("Requires a second XPU device")
    device = torch.device("xpu", 1 if wrong_device else 0)
    # Both checks run before the draft weight/workspace checks.
    hidden = torch.empty((7, 2048), dtype=torch.float16, device=device)
    output = torch.full_like(hidden, 17)
    residual = torch.full_like(hidden, 19)
    control = torch.zeros(
        native.control_words(), dtype=torch.int32, device=device
    )
    shape = (1, 2, 64, 256)
    offset = 0 if wrong_device else 1
    cache = torch.empty(
        2 * 64 * 256 + offset, dtype=torch.float16, device=device
    )[offset:].view(shape)
    positions = torch.zeros(7, dtype=torch.int64, device=device)
    lengths = torch.ones(7, dtype=torch.int32, device=device)
    blocks = torch.zeros((7, 1), dtype=torch.int32, device=device)
    rope = torch.empty((1, 128), dtype=torch.float16, device=device)
    match = "draft hidden" if wrong_device else "KV cache.*64-byte"
    with torch.xpu.device(0):
        owner = torch.classes.minicpm5_mono.Region(native.mailbox_bytes())
        with pytest.raises(RuntimeError, match=match):
            native.draft_decode(
                hidden,
                None,
                positions,
                blocks,
                lengths,
                positions,
                output,
                residual,
                [],
                control,
                [],
                cache,
                owner,
                0,
                rope,
            )
    assert torch.count_nonzero(control).item() == 0
    assert bool((output == 17).all()) and bool((residual == 19).all())
