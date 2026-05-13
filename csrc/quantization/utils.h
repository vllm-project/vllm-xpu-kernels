#pragma once

// Vectorization containers
template <typename scalar_t, size_t vec_size>
struct alignas(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};

template <typename scalar_t>
using vec4_t = vec_n_t<scalar_t, 4>;

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
struct DefaultVecOp {
  ScaOp scalar_op;

  inline void operator()(
      vec_n_t<OutT, VEC_SIZE>& dst, const vec_n_t<InT, VEC_SIZE>& src) const {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      scalar_op(dst.val[i], src.val[i]);
    }
  }
};

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
void vectorize_with_alignment(
    const InT* in,
    OutT* out,
    int num_elems,
    int local_idx,
    int local_range,
    ScaOp&& scalar_op) {
  static_assert(
      VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
      "VEC_SIZE must be a positive power-of-two");

  DefaultVecOp<VEC_SIZE, InT, OutT, std::decay_t<ScaOp>> vec_op{scalar_op};

  constexpr int WIDTH = VEC_SIZE * sizeof(InT);
  uintptr_t addr = reinterpret_cast<uintptr_t>(in);

  bool can_vec =
      ((addr & (WIDTH - 1)) == 0) && ((num_elems & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    int64_t num_vec = num_elems / VEC_SIZE;

    using vin_t = vec_n_t<InT, VEC_SIZE>;
    using vout_t = vec_n_t<OutT, VEC_SIZE>;
    auto const* v_in = reinterpret_cast<const vin_t*>(in);
    auto* v_out = reinterpret_cast<vout_t*>(out);

    for (int64_t i = local_idx; i < num_vec; i += local_range) {
      vout_t tmp;
      // Make a local copy of the entire pack
      vin_t src = v_in[i];  // <- encourages a single vector ld
      vec_op(tmp, src);
      v_out[i] = tmp;  // <- encourages a single vector st
    }
    return;
  }

  int misalignment_offset = addr & (WIDTH - 1);
  int alignment_bytes = WIDTH - misalignment_offset;
  int prefix_elems = alignment_bytes & (WIDTH - 1);
  prefix_elems /= sizeof(InT);
  prefix_elems = sycl::min(prefix_elems, num_elems);

  // 1. prefill elements when it is unsafe to vectorize
  for (int i = local_idx; i < prefix_elems; i += local_range) {
    scalar_op(out[i], in[i]);
  }

  in += prefix_elems;
  out += prefix_elems;
  num_elems -= prefix_elems;

  int num_vec = num_elems / VEC_SIZE;
  using vin_t = vec_n_t<InT, VEC_SIZE>;
  using vout_t = vec_n_t<OutT, VEC_SIZE>;
  auto const* v_in = reinterpret_cast<const vin_t*>(in);
  auto* v_out = reinterpret_cast<vout_t*>(out);

  // 2. vectorize the main part
  for (int i = local_idx; i < num_vec; i += local_range) {
    vout_t tmp;
    // Make a local copy of the entire pack
    vin_t src = v_in[i];  // <- encourages a single vector ld
    vec_op(tmp, src);
    v_out[i] = tmp;  // <- encourages a single vector st
  }

  // 3. handle the tail
  int tail_start = num_vec * VEC_SIZE;
  for (int i = local_idx + tail_start; i < num_elems; i += local_range) {
    scalar_op(out[i], in[i]);
  }
}