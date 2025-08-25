#include "core/registration.h"
#include "xpu/ops.h"

#include <torch/library.h>
#include <torch/version.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, xpu_ops) {
  at::Tag stride_tag = at::Tag::needs_fixed_stride_order;
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
