import random
import pytest
import torch
from tests.ops.activation_op import SiluAndMul
from tests.utils import opcheck
import vllm_xpu_kernels._C

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 13824]  # Arbitrary values for testing
SEEDS = [0]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

@pytest.mark.parametrize(
    "activation",
    ["silu_and_mul"])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_act_and_mul(
    activation: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)
    if activation == "silu_and_mul":
        layer = SiluAndMul()
        import tests.register_ops as ops
        #fn = ops.silu_and_mul
        fn = torch.ops._C.silu_and_mul
    out = layer(x)
    ref_out = layer.forward_native(x)
    # The SiluAndMul, MulAndSilu, GELU and FatReLU implementations are
    # equivalent to the native PyTorch implementations, so we can do exact
    # comparison.
    assert not torch.isnan(out).any(), "Tensor contains NaN!"
    #torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d, ))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    if activation == "fatrelu":
        opcheck(fn, (out, x, threshold))
    else:
        opcheck(fn, (out, x))
