#include <sycl/sycl.hpp>
#include "utils.h"
#include "dispatch_utils.h"
#include <cmath>
#include <c10/macros/Macros.h>

namespace vllm {

template <typename T, int64_t rotary_dim, bool is_neox>
class deepseek_scaling_rope_kernel {
 public:
  static constexpr int sg_size = 16;
  deepseek_scaling_rope_kernel(
      const int64_t* positions,
      const T* query,
      const T* key,
      const int64_t* offsets,
      const T* cos_sin_cache,
      T* query_out,
      T* key_out,
      const int64_t batch,
      const int64_t q_num_head,
      const int64_t k_num_head,
      const int64_t head_size,
      const int64_t q_num_head_d,
      const int64_t q_batch_d,
      const int64_t k_num_head_d,
      const int64_t k_batch_d)
      : positions(positions),
        query(query),
        key(key),
        offsets(offsets),
        cos_sin_cache(cos_sin_cache),
        query_out(query_out),
        key_out(key_out),
        batch(batch),
        q_num_head(q_num_head),
        k_num_head(k_num_head),
        head_size(head_size),
        q_num_head_d(q_num_head_d),
        q_batch_d(q_batch_d),
        k_num_head_d(k_num_head_d),
        k_batch_d(k_batch_d) {}

  void rotary_embedding_kernel(
      const int64_t position,
      const T* pe,
      const T* cos_sin_cache,
      T* res) const {
    constexpr int64_t half_rotary_dim = rotary_dim / 2;
    constexpr int64_t vec_2_len = 2;
    using v2_type = sycl::vec<T, vec_2_len>;
    const int64_t cache_idx = position * rotary_dim;
    const T* cos_cache_offset = &cos_sin_cache[cache_idx];
    const T* sin_cache_offset = cos_cache_offset + half_rotary_dim;
    if constexpr (is_neox) {
      // repeat & rotate mul add
      for (int64_t i = 0; i < half_rotary_dim; ++i) {
        int64_t j = i + half_rotary_dim;
        T cv = cos_cache_offset[i];
        T sv = sin_cache_offset[i];
        res[i] = pe[i] * cv - pe[j] * sv;
        res[j] = pe[j] * cv + pe[i] * sv;
      }
    } else {
      // interleave & rotate mul add, unfortunately no prefetch in sycl
      const v2_type* pe_2 = reinterpret_cast<const v2_type*>(pe);
      v2_type* res_2 = reinterpret_cast<v2_type*>(res);
      for (int64_t h = 0; h < half_rotary_dim; ++h) {
        T c = cos_cache_offset[h];
        T s = sin_cache_offset[h];
        v2_type c2 = {c, c};
        v2_type s2 = {s, s};
        v2_type t = pe_2[h];
        v2_type* dst = &res_2[h];
        v2_type tr = {-t[1], t[0]};
        *dst = t * c2 + tr * s2;
      }
    }
  }

  [[sycl::reqd_sub_group_size(sg_size)]] void
  operator()(sycl::nd_item<3> idx) const {
    int64_t batch_idx = idx.get_global_id(0);
    int64_t sg_idx = idx.get_local_id(1);
    int64_t local_id = idx.get_global_id(2);
    int64_t head_idx = sg_idx * sg_size + local_id;
    int64_t qo_idx = batch_idx * q_num_head * head_size + head_idx * head_size;
    int64_t ko_idx = batch_idx * k_num_head * head_size +
                     (head_idx - q_num_head) * head_size;
    int64_t qi_idx = batch_idx * q_batch_d + head_idx * q_num_head_d;
    int64_t ki_idx =
        batch_idx * k_batch_d + (head_idx - q_num_head) * k_num_head_d;
    if (head_idx < q_num_head) {
      rotary_embedding_kernel(
          positions[batch_idx],
          &query[qi_idx],
          cos_sin_cache,
          &query_out[qo_idx]);
    } else if (head_idx < q_num_head + k_num_head) {
      rotary_embedding_kernel(
          positions[batch_idx], &key[ki_idx], cos_sin_cache, &key_out[ko_idx]);
    }
  }

 private:
  const int64_t* positions;
  const T* query;
  const T* key;
  const int64_t* offsets;
  const T* cos_sin_cache;
  T* query_out;
  T* key_out;
  const int64_t batch;
  const int64_t q_num_head;
  const int64_t k_num_head;
  const int64_t head_size;
  const int64_t q_num_head_d;
  const int64_t q_batch_d;
  const int64_t k_num_head_d;
  const int64_t k_batch_d;
};

}  // namespace vllm

template <typename T>
void call_deepseek_scaling_rope(
    const int64_t* positions,
    const T* query,
    const T* key,
    const int64_t* offsets,
    const T* cos_sin_cache,
    T* query_out,
    T* key_out,
    int64_t batch,
    int64_t q_num_head,
    int64_t k_num_head,
    int64_t head_size,
    int64_t rotary_dim,
    bool is_neox,
    int64_t q_num_head_d,
    int64_t q_batch_d,
    int64_t k_num_head_d,
    int64_t k_batch_d) {
  static constexpr std::array<int, 5> allowed_dims = {32, 64, 96, 128, 256};
  auto it = std::find(allowed_dims.begin(), allowed_dims.end(), rotary_dim);

  TORCH_CHECK(
      it != allowed_dims.end(),
      "Invalid rotary_dim (",
      rotary_dim,
      "). Supported: 32,64,96,128,256");
  TORCH_CHECK(
      rotary_dim == head_size,
      "rotary_dim (",
      rotary_dim,
      ") must equal head_size (",
      head_size,
      ")");

  const int rot_idx = std::distance(allowed_dims.begin(), it);
  const int neox_idx = is_neox ? 1 : 0;
  const int func_idx = neox_idx * allowed_dims.size() + rot_idx;

  using LaunchFn = void (*)(
      sycl::queue&,
      const int64_t*,
      const T*,
      const T*,
      const int64_t*,
      const T*,
      T*,
      T*,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t,
      int64_t);

// Table builder macro
#define REGISTER_CASE(dim, neox)                                         \
  [](sycl::queue& q,                                                     \
     const int64_t* pos,                                                 \
     const T* q_in,                                                      \
     const T* k_in,                                                      \
     const int64_t* off,                                                 \
     const T* cache,                                                     \
     T* q_out,                                                           \
     T* k_out,                                                           \
     int64_t b,                                                          \
     int64_t qh,                                                         \
     int64_t kh,                                                         \
     int64_t hs,                                                         \
     int64_t qhd,                                                        \
     int64_t qbd,                                                        \
     int64_t khd,                                                        \
     int64_t kbd) {                                                      \
    constexpr int64_t sg_size = 16;                                      \
    int64_t sg_per_heads = (qh + kh + sg_size - 1) / sg_size;            \
    sycl::range<3> local(1, sg_per_heads, sg_size);                      \
    sycl::range<3> global(b, sg_per_heads, sg_size);                     \
    at::DeviceGuard dg(at::Device(at::kXPU, at::xpu::current_device())); \
    q.submit([&](sycl::handler& cgh) {                                   \
      cgh.parallel_for(                                                  \
          sycl::nd_range<3>(global, local),                              \
          vllm::deepseek_scaling_rope_kernel<T, dim, neox>{              \
              pos,                                                       \
              q_in,                                                      \
              k_in,                                                      \
              off,                                                       \
              cache,                                                     \
              q_out,                                                     \
              k_out,                                                     \
              b,                                                         \
              qh,                                                        \
              kh,                                                        \
              hs,                                                        \
              qhd,                                                       \
              qbd,                                                       \
              khd,                                                       \
              kbd});                                                     \
    });                                                                  \
  }

  static constexpr std::array<LaunchFn, allowed_dims.size() * 2> table = {
      REGISTER_CASE(32, false),
      REGISTER_CASE(64, false),
      REGISTER_CASE(96, false),
      REGISTER_CASE(128, false),
      REGISTER_CASE(256, false),
      REGISTER_CASE(32, true),
      REGISTER_CASE(64, true),
      REGISTER_CASE(96, true),
      REGISTER_CASE(128, true),
      REGISTER_CASE(256, true),
  };

  auto& queue = vllm::xpu::vllmGetQueue();
  table[func_idx](
      queue,
      positions,
      query,
      key,
      offsets,
      cos_sin_cache,
      query_out,
      key_out,
      batch,
      q_num_head,
      k_num_head,
      head_size,
      q_num_head_d,
      q_batch_d,
      k_num_head_d,
      k_batch_d);

#undef REGISTER_CASE
}

/**
 * @brief Perform deepseek rotary embedding with q&k.
 * @param positions index of embedding [batch]
 * @param query query to be processed [batch, num_head, head_dim]
 * @param key key to be processed [batch, num_head, head_dim]
 * @param offsets optional tensor for offset with position
 * @param cos_sin_cache shared cache with cos/sin
 * @param is_neox choose interleave or half.
 * @return A tuple of tensors (query_out, key_out).
 */
std::tuple<torch::Tensor, torch::Tensor> deepseek_scaling_rope(
    const torch::Tensor& positions,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const c10::optional<torch::Tensor>& offsets_opt,
    const torch::Tensor& cos_sin_cache,
    int64_t rotary_dim,
    bool is_neox) {
  auto query_out = at::empty_like(query);
  auto key_out = at::empty_like(key);

  auto q_shape = query.sizes();
  auto q_stride = query.strides();
  int64_t head_size = q_shape[2];
  int64_t q_num_head = q_shape[1];
  int64_t batch = q_shape[0];
  int64_t q_num_head_d = q_stride[1];
  int64_t q_batch_d = q_stride[0];
  auto k_shape = key.sizes();
  auto k_stride = key.strides();
  int64_t k_num_head = k_shape[1];
  int64_t k_num_head_d = k_stride[1];
  int64_t k_batch_d = k_stride[0];
  if (is_neox) {
    query_out = query_out.reshape({1, batch, q_num_head, head_size});
    key_out = key_out.reshape({1, batch, k_num_head, head_size});
  }
  TORCH_CHECK(
      cos_sin_cache.sizes()[1] == head_size,
      "Rotary dim doesn't match query head_size");
  TORCH_CHECK(
      cos_sin_cache.sizes()[1] == k_shape[2],
      "Rotary dim doesn't match key head_size");
  const c10::MaybeOwned<torch::Tensor> offsets_maybe_owned =
      at::borrow_from_optional_tensor(offsets_opt);
  const torch::Tensor& offsets = *offsets_maybe_owned;
  auto offsets_ptr = offsets.defined() ? offsets.data_ptr() : nullptr;
  switch (query.scalar_type()) {
    case torch::kFloat:
      call_deepseek_scaling_rope<float>(
          reinterpret_cast<int64_t*>(positions.data_ptr()),
          reinterpret_cast<float*>(query.data_ptr()),
          reinterpret_cast<float*>(key.data_ptr()),
          reinterpret_cast<int64_t*>(offsets_ptr),
          reinterpret_cast<float*>(cos_sin_cache.data_ptr()),
          reinterpret_cast<float*>(query_out.data_ptr()),
          reinterpret_cast<float*>(key_out.data_ptr()),
          batch,
          q_num_head,
          k_num_head,
          head_size,
          rotary_dim,
          is_neox,
          q_num_head_d,
          q_batch_d,
          k_num_head_d,
          k_batch_d);
      break;
    case torch::kFloat16:
      call_deepseek_scaling_rope<sycl::half>(
          reinterpret_cast<int64_t*>(positions.data_ptr()),
          reinterpret_cast<sycl::half*>(query.data_ptr()),
          reinterpret_cast<sycl::half*>(key.data_ptr()),
          reinterpret_cast<int64_t*>(offsets_ptr),
          reinterpret_cast<sycl::half*>(cos_sin_cache.data_ptr()),
          reinterpret_cast<sycl::half*>(query_out.data_ptr()),
          reinterpret_cast<sycl::half*>(key_out.data_ptr()),
          batch,
          q_num_head,
          k_num_head,
          head_size,
          rotary_dim,
          is_neox,
          q_num_head_d,
          q_batch_d,
          k_num_head_d,
          k_batch_d);
      break;
    case torch::kBFloat16:
      call_deepseek_scaling_rope<sycl::ext::oneapi::bfloat16>(
          reinterpret_cast<int64_t*>(positions.data_ptr()),
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(query.data_ptr()),
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(key.data_ptr()),
          reinterpret_cast<int64_t*>(offsets_ptr),
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(
              cos_sin_cache.data_ptr()),
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(query_out.data_ptr()),
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(key_out.data_ptr()),
          batch,
          q_num_head,
          k_num_head,
          head_size,
          rotary_dim,
          is_neox,
          q_num_head_d,
          q_batch_d,
          k_num_head_d,
          k_batch_d);
      break;
    default:
      throw std::invalid_argument(
          "Invalid dtype, only supports float32, float16, and bfloat16");
      break;
  }
  return {query_out, key_out};
}
