# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import torch.nn as nn

logger = logging.getLogger("vllm_xpu_kernel")


class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    def __new__(cls, *args, **kwargs):
        try:
            op_name = cls.__name__
        except AttributeError:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@CustomOp.register, or it's the CustomOp base class itself."
            ) from None
        logger.debug("Instantiating custom op: %s", op_name)
        op_cls_to_instantiate = cls
        return super().__new__(op_cls_to_instantiate)

    def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.
        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_xpu(self, *args, **kwargs):
        # By default, we assume that XPU ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        # By default, we assume that CPU ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    def dispatch_forward(self):
        return self.forward_xpu
