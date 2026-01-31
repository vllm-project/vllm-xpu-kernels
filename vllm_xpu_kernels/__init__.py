# SPDX-License-Identifier: Apache-2.0

from .flash_attn_interface import flash_attn_varlen_func  # noqa: F401

# Import xpumem_allocator if available
try:
    from . import xpumem_allocator  # noqa: F401
except ImportError:
    # Module not available (not built or not installed)
    pass
