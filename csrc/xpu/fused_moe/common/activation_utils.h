#pragma once

namespace FusedMOE {

enum class ActivationType {
  NONE = -1,
  SILU = 0,
  GELU = 1,
  GELU_TANH = 2,
  SWIGLUOAI = 3,
  RELU2_NO_MUL = 4,
  SWIGLUSTEP = 5,
};

template <typename T>
inline T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + sycl::exp((float)-x)));
}

template <typename T>
inline T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + sycl::erf(f * ALPHA)));
}

template <typename T>
inline T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + sycl::tanh(inner)));
}

template <typename T>
inline T relu2_no_mul_kernel(const T& x) {
  // square(relu(x))
  const float f = (float)x;
  const float r = f > 0.0f ? f : 0.0f;
  return (T)(r * r);
}

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

template <typename T>
[[intel::device_indirectly_callable]] inline __attribute__((always_inline)) T
swiglustep_and_mul(const T& gate, const T& up, float limit) {
  // gate = silu(gate).clamp(max=limit)
  const float gate_f = (float)gate;
  const float silu_gate = gate_f / (1.0f + sycl::exp(-gate_f));
  const float clamped_gate = silu_gate > limit ? limit : silu_gate;

  // up = up.clamp(min=-limit, max=limit)
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  return (T)(clamped_gate * clamped_up);
}
}  // namespace FusedMOE
