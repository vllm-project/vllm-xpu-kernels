# SPDX-License-Identifier: Apache-2.0
import random
import unittest
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import torch
from torch._prims_common import TensorLikeType

ALL_OPCHECK_TEST_UTILS: tuple[str, ...] = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


# Copied/modified from torch._refs.__init__.py
def fp8_allclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """
    Reference implementation of torch.allclose
    """
    torch._refs._check_close_args(name="torch.allclose",
                                  a=a,
                                  b=b,
                                  rtol=rtol,
                                  atol=atol)

    return bool(
        torch.all(
            torch.isclose(a.double(),
                          b.double(),
                          rtol=rtol,
                          atol=atol,
                          equal_nan=equal_nan)).item())


# A special version of op check that has a restricted default set of test_utils
# and a patched version of allclose that supports fp8 types.
def opcheck(op: Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket,
                      torch._library.custom_ops.CustomOpDef],
            args: tuple[Any, ...],
            kwargs: Optional[dict[str, Any]] = None,
            *,
            test_utils: Union[str, Sequence[str]] = ALL_OPCHECK_TEST_UTILS,
            raise_exception: bool = True,
            cond: bool = True) -> dict[str, str]:
    with unittest.mock.patch('torch.allclose', new=fp8_allclose):
        return torch.library.opcheck(
            op,
            args,
            kwargs,
            test_utils=test_utils,
            raise_exception=raise_exception) if cond else {}


STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
    "int8": torch.int8,
    "fp8_inc": torch.float8_e4m3fn,
}


def seed_everything(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
