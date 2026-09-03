# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch
import torch.nn as nn

from tests.ops.custom_ops import CustomOp


def fused_add_rms_norm(
        x: torch.Tensor, residual: torch.Tensor,
        weight: Optional[torch.Tensor],
        variance_epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    import tests.register_ops as ops
    ops.fused_add_rms_norm(
        x,
        residual,
        weight,
        variance_epsilon,
    )
    return x, residual


def rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor],
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


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor,
                   weight: Optional[torch.Tensor], variance_epsilon: float,
                   activation: str) -> torch.Tensor:
    import tests.register_ops as ops
    out = torch.empty_like(x)
    ops.fused_rms_norm_gated(
        out,
        x,
        gate,
        weight,
        variance_epsilon,
        activation,
    )
    return out


def dispatch_cuda_rmsnorm_func(add_residual: bool):
    if add_residual:
        return fused_add_rms_norm
    return rms_norm


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


def dispatch_cuda_gemma_rmsnorm_func(add_residual: bool):
    if add_residual:
        return fused_add_gemma_rms_norm
    return gemma_rms_norm


class GemmaRMSNorm(CustomOp):
    """RMS normalization for Gemma.

    Two differences from RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * (1 + w)) is computed in fp32 then downcast.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        if dtype is not None:
            self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        else:
            self.weight = nn.Parameter(torch.zeros(hidden_size))

    @staticmethod
    def forward_static(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        x = x * (1.0 + weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_static(self.weight.data, self.variance_epsilon, x,
                                   residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        add_residual = residual is not None
        norm_func = dispatch_cuda_gemma_rmsnorm_func(add_residual)
        weight = self.weight.data
        if add_residual:
            return norm_func(x, residual, weight, self.variance_epsilon)
        return norm_func(x, weight, self.variance_epsilon)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_cuda(x, residual)


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
        weight = self.weight.data if self.has_weight else None

        if add_residual:
            return norm_func(x, residual, weight, self.variance_epsilon)
        else:
            return norm_func(x, weight, self.variance_epsilon)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_cuda(x, residual)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.hidden_size}"
        s += f", eps={self.variance_epsilon}"
        return s


class RMSNormGated(CustomOp):
    """Gated root mean square normalization.

    Computes ``out = w * x / sqrt(E[x^2] + eps) * act(g)``, the output norm
    used by linear-attention layers. Mirrors ``fla.modules.FusedRMSNormGated``
    (and vLLM's ``RMSNormGated``): KDA uses ``sigmoid``, the Gated DeltaNet
    family uses ``swish``.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        activation: str = "sigmoid",
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if activation not in ("sigmoid", "swish", "silu"):
            raise ValueError(f"Unsupported activation: {activation}")

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.activation = activation
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
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x_f = x.to(torch.float32)
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_f = x_f * torch.rsqrt(variance + self.variance_epsilon)
        if self.has_weight:
            x_f = x_f * self.weight.to(torch.float32)
        g = gate.to(torch.float32)
        if self.activation == "sigmoid":
            x_f = x_f * torch.sigmoid(g)
        else:
            x_f = x_f * g * torch.sigmoid(g)
        return x_f.to(orig_dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.weight.data if self.has_weight else None
        return rms_norm_gated(x, gate, weight, self.variance_epsilon,
                              self.activation)

    def forward_xpu(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_cuda(x, gate)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.hidden_size}"
        s += f", eps={self.variance_epsilon}"
        s += f", activation={self.activation}"
        return s
