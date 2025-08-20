#include <sycl/sycl.hpp>

#include "utils.h"
#include "dispatch_utils.h"

#define VLLM_LDG(arg) *(arg)

namespace vllm {
template <typename T>
[[intel::device_indirectly_callable]] inline __attribute__((always_inline)) T
swigluoai_and_mul(const T& gate, const T& up, float alpha, float limit) {
  // clamp gate: min=None, max=limit
  const float gate_f = (float)gate;
  const float clamped_gate = gate_f > limit ? limit : gate_f;

  // clamp up: min=-limit, max=limit
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  // glu = gate * sigmoid(gate * alpha)
  const float sigmoid_val = 1.0f / (1.0f + sycl::exp(-clamped_gate * alpha));
  const float glu = clamped_gate * sigmoid_val;

  // (up + 1) * glu
  return (T)((clamped_up + 1.0f) * glu);
}

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float,
                             const float)>
class swigluoai_and_mul_kernel {
 public:
  swigluoai_and_mul_kernel(scalar_t* __restrict__ out,          // [..., d]
                           const scalar_t* __restrict__ input,  // [..., 2, d]
                           const int d, const float alpha, const float limit)
      : out(out), input(input), d(d), alpha(alpha), limit(limit) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_idx = item.get_group(0);
    for (int64_t idx = item.get_local_id(0); idx < d;
         idx += item.get_local_range(0)) {
      // gate = x[..., ::2]  (even indices)
      const scalar_t gate = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx]);
      // up = x[..., 1::2]   (odd indices)
      const scalar_t up = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx + 1]);

      out[token_idx * d + idx] = ACT_FN(gate, up, alpha, limit);
    }
  }

 private:
  scalar_t* out;          // [..., d]
  const scalar_t* input;  // [..., topk, d]
  const int d;
  const float alpha;
  const float limit;
};

#define LAUNCH_SWIGLUOAI_AND_MUL(KERNEL, ALPHA, LIMIT)                     \
  int d = input.size(-1) / 2;                                              \
  int64_t num_tokens = input.numel() / input.size(-1);                     \
  sycl::range<1> grid(num_tokens);                                         \
  sycl::range<1> block(std::min(d, 1024));                                 \
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());  \
  const at::DeviceGuard device_guard(curDevice);                           \
  auto& queue = vllm::xpu::vllmGetQueue();                                 \
  VLLM_DISPATCH_FLOATING_TYPES(                                            \
      input.scalar_type(), "clamp_swiglu_kernel_with_params", [&] {        \
        queue.submit([&](sycl::handler& cgh) {                             \
          cgh.parallel_for(                                                \
              sycl::nd_range<1>(grid * block, block),                      \
              vllm::swigluoai_and_mul_kernel<scalar_t, KERNEL<scalar_t>>(  \
                  out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d, \
                  ALPHA, LIMIT));                                          \
        });                                                                \
      });
}  // namespace vllm

void swigluoai_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input,  // [..., 2 * d]
                       double alpha, double limit) {
  LAUNCH_SWIGLUOAI_AND_MUL(vllm::swigluoai_and_mul, alpha, limit);
}
