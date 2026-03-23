# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.ops.custom_ops import CustomOp


def fused_add_rms_norm(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        variance_epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    import tests.register_ops as ops
    ops.fused_add_rms_norm(
        x,
        residual,
        weight,
        variance_epsilon,
    )
    return x, residual


def rms_norm(x: torch.Tensor, weight: torch.Tensor,
             variance_epsilon: float) -> torch.Tensor:
    import tests.register_ops as ops
    out = torch.empty_like(x)
    ops.rms_norm(
        out,
        x,
        weight,
        variance_epsilon,
    )
    return out


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor,
                   variance_epsilon: float) -> torch.Tensor:
    import tests.register_ops as ops
    out = torch.empty_like(x)
    ops.gemma_rms_norm(
        out,
        x,
        weight,
        variance_epsilon,
    )
    return out


def fused_add_gemma_rms_norm(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        variance_epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    import tests.register_ops as ops
    ops.fused_add_gemma_rms_norm(
        x,
        residual,
        weight,
        variance_epsilon,
    )
    return x, residual


def rms_norm_gated(
        x: torch.Tensor,
        weight: torch.Tensor,
        z: Optional[torch.Tensor],
        variance_epsilon: float,
        norm_before_gate: bool = True,
        activation: str = "swish") -> torch.Tensor:
    import tests.register_ops as ops

    out = torch.empty_like(x)
    ops.rms_norm_gated(out, x, weight, z, variance_epsilon,
                       norm_before_gate, activation)
    return out


def dispatch_cuda_rmsnorm_func(add_residual: bool):
    if add_residual:
        return fused_add_rms_norm
    return rms_norm


class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (None if var_hidden_size == hidden_size
                                       else var_hidden_size)
        self.has_weight = has_weight
        if dtype is not None:
            self.weight = torch.ones(hidden_size, dtype=dtype)
        else:
            self.weight = torch.ones(hidden_size)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError("Expected hidden_size to be "
                             f"{self.hidden_size}, but found: {hidden_size}")

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}")

            x_var = x[:, :, :self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype)
        if self.has_weight:
            x = x * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        add_residual = residual is not None
        norm_func = dispatch_cuda_rmsnorm_func(add_residual)

        if add_residual:
            return norm_func(x, residual, self.weight.data,
                             self.variance_epsilon)
        else:
            return norm_func(x, self.weight.data, self.variance_epsilon)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_cuda(x, residual)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


class GemmaRMSNorm(CustomOp):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    @staticmethod
    def _forward_static_no_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        x = x.to(orig_dtype)
        return x

    @staticmethod
    def _forward_static_with_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = (x.float() + residual.float() if orig_dtype == torch.float16 else
             x + residual)
        residual = x.to(orig_dtype)

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        x = x.to(orig_dtype)
        return x, residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if residual is None:
            return self._forward_static_no_residual(
                self.weight.data, self.variance_epsilon, x)
        return self._forward_static_with_residual(
            self.weight.data, self.variance_epsilon, x, residual)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if residual is None:
            return gemma_rms_norm(x, self.weight.data, self.variance_epsilon)
        return fused_add_gemma_rms_norm(x, residual, self.weight.data,
                                        self.variance_epsilon)


class RMSNormGated(CustomOp):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        norm_before_gate: bool = False,
        dtype: Optional[torch.dtype] = None,
        activation: str = "swish",
    ) -> None:
        super().__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        if dtype is not None:
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        else:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)

    def forward_native(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z is not None and not self.norm_before_gate:
            x = x * F.silu(z)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(variance + self.eps)
        out = out * self.weight

        if z is not None and self.norm_before_gate:
            out = out * F.silu(z)

        return out.to(x.dtype)

    def forward_xpu(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return rms_norm_gated(
            x,
            self.weight,
            z,
            self.eps,
            self.norm_before_gate,
            self.activation,
        )
