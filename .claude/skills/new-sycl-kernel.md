# New SYCL Kernel

Implement a SYCL kernel in vllm-xpu-kernels as a counterpart to a vLLM CUDA kernel.

## Prerequisites

- Identify the CUDA kernel to port from https://github.com/vllm-project/vllm/tree/main/csrc
- Understand the kernel's inputs, outputs, and algorithm
- Determine which extension module it belongs to (`_C`, `_vllm_fa2_C`, `_moe_C`, or `_xpu_C`)

## Kernel Implementation Steps

### 1. Add Function Declaration to `csrc/ops.h`

Add the C++ function signature that will be called from Python:

```cpp
void your_kernel_name(
    torch::Tensor& out,
    torch::Tensor& input,
    // ... other parameters
);
```

### 2. Implement the SYCL Kernel

Create or modify the appropriate `.cpp` file in `csrc/`:

```cpp
#include <sycl/sycl.hpp>
#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {

// SYCL kernel class (functor)
template <typename scalar_t>
class your_kernel_class {
 public:
  your_kernel_class(
      scalar_t* out_,
      const scalar_t* input_,
      // ... other params
      )
      : out(out_), input(input_) /* ... */ {}

  // Main kernel entry point
  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    // Get indices
    int token_idx = item_ct1.get_group(2);
    int local_idx = item_ct1.get_local_id(2);

    // Kernel implementation using SYCL
    // Use sycl::reduce_over_group for reductions
    // Use sycl::local_accessor for shared memory
  }

 private:
  scalar_t* __restrict__ out;
  const scalar_t* __restrict__ input;
  // ... other members
};

// Launcher function template
template <typename scalar_t>
void call_your_kernel(
    torch::Tensor& out,
    torch::Tensor& input,
    // ... other params
) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  // Get pointers
  auto out_ptr = out.data_ptr<scalar_t>();
  auto input_ptr = input.data_ptr<scalar_t>();

  // Define grid/block dimensions
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, 1024));

  // Get SYCL queue
  auto& queue = vllm::xpu::vllmGetQueue();

  // Submit kernel
  queue.submit([&](sycl::handler& cgh) {
    // Declare local memory if needed
    sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(size), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block),
        your_kernel_class<sycl_t>(
            (sycl_t*)out_ptr,
            (const sycl_t*)input_ptr,
            // ... other args
            shared_mem));
  });
}

} // namespace vllm
```

### 3. Add Dispatcher/Launcher Function

At the end of the kernel file, add the C++ function that PyTorch will call:

```cpp
void your_kernel_name(
    torch::Tensor& out,
    torch::Tensor& input,
    // ... other params
) {
  // Optional: device checks
  CHECK_DEVICE(out);
  CHECK_DEVICE(input);

  // Dispatch to the correct template specialization based on dtype
  DISPATCH_FOR_2_DTYPE(out.scalar_type(), input.scalar_type(), [&] {
    call_your_kernel<scalar_t>(out, input, /* ... */);
  });
}
```

### 4. Register the Op in `torch_bindings.cpp`

Add to the appropriate `TORCH_LIBRARY_EXPAND` block:

```cpp
// In csrc/torch_bindings.cpp (for _C module)
// or csrc/xpu/torch_bindings.cpp (for _xpu_C module)

ops.def("your_kernel_name(Tensor! out, Tensor input, ...) -> ()");
ops.impl("your_kernel_name", torch::kXPU, &your_kernel_name);
```

### 5. Write Tests in `tests/test_your_kernel.py`

```python
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.utils import opcheck

DTYPES = [torch.half, torch.bfloat16]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

# Optional: mini test params for faster iteration
MINI_PYTEST_PARAMS = {
    "default": {
        "param_name": [value],
    },
}

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", XPU_DEVICES)
@torch.inference_mode()
def test_your_kernel(dtype: torch.dtype, device: str) -> None:
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    # Create test inputs
    input = torch.randn(..., dtype=dtype)
    out = torch.empty_like(input)

    # Call your kernel via torch.ops
    torch.ops._C.your_kernel_name(out, input, ...)

    # Verify results against reference implementation
    expected = reference_implementation(input)
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    # Op check for correctness validation
    opcheck(torch.ops._C.your_kernel_name, (out, input, ...))
```

## Key Patterns and Utilities

### SYCL Type Mapping

Use `SyclTypeTrait` to map PyTorch types to SYCL types:

```cpp
using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
// Maps: c10::Half -> sycl::half, c10::BFloat16 -> sycl::ext::oneapi::bfloat16
```

### SYCL Queue Access

```cpp
auto& queue = vllm::xpu::vllmGetQueue();
```

### Device Architecture Checks

```cpp
#include "utils.h"

if (vllm::xpu::is_pvc()) { /* PVC-specific code */ }
if (vllm::xpu::is_bmg()) { /* BMG-specific code */ }
if (vllm::xpu::is_xe2_arch()) { /* XE2 arch code */ }
```

### Vectorized Memory Access

```cpp
// Use aligned_vec for vectorized loads/stores
template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vec {
  scalar_t val[vec_size];
};

// Or manual vectorization via reinterpret_cast
using vec4_t = vllm::aligned_vec<scalar_t, 4>;
auto* vec_in = reinterpret_cast<const vec4_t*>(input_ptr);
```

### Accumulation Types

```cpp
using acc_t = vllm::xpu::acc_type<scalar_t>;
// Promotes half/bfloat16 to float for accumulation
```

### Dispatch Macros

Available in `csrc/dispatch_utils.h`:

```cpp
// Dispatch by output dtype
DISPATCH_FOR_2_DTYPE(out.scalar_type(), input.scalar_type(), [&] { ... });

// Rank dispatch for handling 2D/3D/4D tensors
VLLM_DISPATCH_RANK234(num_dims, [&]() { ... });
```

## Build and Test

```bash
# Incremental rebuild
python -m build --wheel --no-isolation

# Run your test
.venv/bin/python -m pytest tests/test_your_kernel.py -v

# Run with mini params (if MINI_PYTEST_PARAMS defined)
XPU_KERNEL_PYTEST_PROFILER=MINI .venv/bin/python -m pytest tests/test_your_kernel.py -v
```

## CUDA-to-SYCL Translation Reference

| CUDA | SYCL |
|------|------|
| `blockIdx.x` | `item_ct1.get_group(2)` |
| `threadIdx.x` | `item_ct1.get_local_id(2)` |
| `blockDim.x` | `item_ct1.get_local_range(2)` |
| `gridDim.x` | `item_ct1.get_group_range(2)` |
| `__shared__` | `sycl::local_accessor<T, 1>` |
| `__syncthreads()` | `item_ct1.barrier(sycl::access::fence_space::local_space)` |
| `warpReduceSum` | `sycl::reduce_over_group` |
| `__half` | `sycl::half` |
| `__nv_bfloat16` | `sycl::ext::oneapi::bfloat16` |
| `cudaStream_t` | `sycl::queue&` |

## Tips

1. **Always use `[[sycl::reqd_sub_group_size(32)]]`** for warp-sized operations
2. **Check alignment** before vectorized loads - use scalar fallback for unaligned data
3. **Use barriers carefully** - SYCL barriers are work-group scoped
4. **Prefer `sycl::reduce_over_group`** over manual shuffle reductions
5. **Test on multiple devices** if possible (PVC, BMG)
