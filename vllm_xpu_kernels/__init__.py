# SPDX-License-Identifier: Apache-2.0

# Import xpumem_allocator if available
from contextlib import suppress

from .flash_attn_interface import flash_attn_varlen_func  # noqa: F401

with suppress(ImportError):
    from . import xpumem_allocator  # noqa: F401
