#pragma once
#include <ATen/ATen.h>
#include <ATen/native/mkldnn/xpu/detail/DnnlExt.h>
#include <ATen/native/mkldnn/xpu/detail/LRUCache.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
#include <dnnl.hpp>

inline constexpr std::string_view USES_FP64_MATH("uses-fp64-math");
inline constexpr std::string_view ASPECT_FP64_IS_NOT_SUPPORTED(
    "aspect fp64 is not supported");
inline constexpr std::string_view FP64_ERROR_FROM_MKL(
    "double type is not supported");
inline constexpr std::string_view OUT_OF_RESOURCES("PI_ERROR_OUT_OF_RESOURCES");

#define DPCPP_EXCEP_TRY try {
#define DPCPP_EXCEP_CATCH                                                  \
  }                                                                        \
  catch (const sycl::exception& ep) {                                      \
    const std::string_view err_msg(ep.what());                             \
    if (err_msg.find(USES_FP64_MATH) != std::string::npos ||               \
        err_msg.find(ASPECT_FP64_IS_NOT_SUPPORTED) != std::string::npos || \
        err_msg.find(FP64_ERROR_FROM_MKL) != std::string::npos) {          \
      throw std::runtime_error(                                            \
          "FP64 data type is unsupported on current platform.");           \
    } else if (err_msg.find(OUT_OF_RESOURCES) != std::string::npos) {      \
      throw std::runtime_error(                                            \
          "Allocation is out of device memory on current platform.");      \
    } else {                                                               \
      throw ep;                                                            \
    }                                                                      \
  }

#define DPCPP_EXT_SUBMIT(q, str, ker_submit) \
  {                                          \
    DPCPP_EXCEP_TRY                          \
    auto e = (ker_submit);                   \
    (q).throw_asynchronous();                \
    DPCPP_EXCEP_CATCH                        \
  }

#define DPCPP_ONEDNN_EXEC_WITH_ARGHANDLES(prim_ext, stream, engine,           \
                                          arg_handles, ...)                   \
  {                                                                           \
    auto q = dnnl::sycl_interop::get_queue((stream));                         \
    DPCPP_EXT_SUBMIT((q), "onednn_ext_kernel",                                \
                     prim_ext.execute(stream, engine, std::move(arg_handles), \
                                      ##__VA_ARGS__));                        \
  }

namespace oneDNN {

static inline dnnl::memory::data_type get_onednn_dtype(
    const at::Tensor& tensor, bool allow_undef = false) {
  switch (tensor.scalar_type()) {
    case at::ScalarType::Byte:
      return dnnl::memory::data_type::u8;
    case at::ScalarType::Char:
      return dnnl::memory::data_type::s8;
    case at::ScalarType::QInt8:
      return dnnl::memory::data_type::s8;
    case at::ScalarType::QUInt8:
      return dnnl::memory::data_type::u8;
    case at::ScalarType::Int:
      return dnnl::memory::data_type::s32;
    case at::ScalarType::Half:
      return dnnl::memory::data_type::f16;
    case at::ScalarType::Float:
      return dnnl::memory::data_type::f32;
    case at::ScalarType::BFloat16:
      return dnnl::memory::data_type::bf16;
    case at::ScalarType::Float8_e4m3fn:
      return dnnl::memory::data_type::f8_e4m3;
    case at::ScalarType::Float8_e5m2:
      return dnnl::memory::data_type::f8_e5m2;
    default:
      if (!allow_undef) {
        TORCH_CHECK(false, c10::toString(tensor.scalar_type()),
                    " is not supported in oneDNN!");
      }
      return dnnl::memory::data_type::undef;
  };
}

static inline dnnl::memory::data_type get_onednn_dtype_include_double(
    const at::Tensor& tensor, bool allow_undef = false) {
  c10::DeviceIndex curDevID = at::xpu::current_device();
  c10::xpu::DeviceProp dev_prop;
  c10::xpu::get_device_properties(&dev_prop, curDevID);
  bool fp64_valid = dev_prop.has_fp64;

  if (tensor.scalar_type() == at::ScalarType::Double) {
    if (fp64_valid)
      return dnnl::memory::data_type::f64;
    else if (allow_undef)
      return dnnl::memory::data_type::undef;
    else
      TORCH_CHECK(false, "Double is not supported on this device!");
  }

  return get_onednn_dtype(tensor, allow_undef);
}

static inline memory::dims get_onednn_dims(const at::Tensor& tensor) {
  memory::dims dims;
  for (int i = 0; i < tensor.sizes().size(); i++)
    dims.push_back(tensor.size(i));
  return dims;
}

static inline memory::dims get_onednn_strides(const at::Tensor& tensor) {
  memory::dims strides;
  for (int i = 0; i < tensor.strides().size(); i++)
    strides.push_back(tensor.stride(i));
  return strides;
}

static inline dnnl::memory::desc get_onednn_md(const at::Tensor& tensor) {
  return {get_onednn_dims(tensor), get_onednn_dtype_include_double(tensor),
          get_onednn_strides(tensor)};
}

enum class joint_dtypes_t {
  f32 = 0,
  f16,
  bf16,
  int8,
  f16_int4,
  bf16_int4,
  s8_int4,
  u8_int4,
  f16_f8_e5m2,
  bf16_f8_e5m2,
  f16_f8_e4m3,
  bf16_f8_e4m3,
};

template <joint_dtypes_t Ts>
struct onednn_types_mapper;

template <>
struct onednn_types_mapper<joint_dtypes_t::f16_int4> {
  static inline std::tuple<dnnl::memory::data_type, dnnl::memory::data_type>
  get() {
    return std::make_tuple(dnnl::memory::data_type::f16,
                           dnnl::memory::data_type::u4);
  }
};

template <>
struct onednn_types_mapper<joint_dtypes_t::bf16_int4> {
  static inline std::tuple<dnnl::memory::data_type, dnnl::memory::data_type>
  get() {
    return std::make_tuple(dnnl::memory::data_type::bf16,
                           dnnl::memory::data_type::u4);
  }
};

template <>
struct onednn_types_mapper<joint_dtypes_t::s8_int4> {
  static inline std::tuple<dnnl::memory::data_type, dnnl::memory::data_type>
  get() {
    return std::make_tuple(dnnl::memory::data_type::s8,
                           dnnl::memory::data_type::u4);
  }
};

template <>
struct onednn_types_mapper<joint_dtypes_t::u8_int4> {
  static inline std::tuple<dnnl::memory::data_type, dnnl::memory::data_type>
  get() {
    return std::make_tuple(dnnl::memory::data_type::u8,
                           dnnl::memory::data_type::u4);
  }
};

template <>
struct onednn_types_mapper<joint_dtypes_t::f16_f8_e5m2> {
  static inline std::tuple<dnnl::memory::data_type, dnnl::memory::data_type>
  get() {
    return std::make_tuple(dnnl::memory::data_type::f16,
                           dnnl::memory::data_type::f8_e5m2);
  }
};

template <>
struct onednn_types_mapper<joint_dtypes_t::bf16_f8_e5m2> {
  static inline std::tuple<dnnl::memory::data_type, dnnl::memory::data_type>
  get() {
    return std::make_tuple(dnnl::memory::data_type::bf16,
                           dnnl::memory::data_type::f8_e5m2);
  }
};

template <>
struct onednn_types_mapper<joint_dtypes_t::f16_f8_e4m3> {
  static inline std::tuple<dnnl::memory::data_type, dnnl::memory::data_type>
  get() {
    return std::make_tuple(dnnl::memory::data_type::f16,
                           dnnl::memory::data_type::f8_e4m3);
  }
};

template <>
struct onednn_types_mapper<joint_dtypes_t::bf16_f8_e4m3> {
  static inline std::tuple<dnnl::memory::data_type, dnnl::memory::data_type>
  get() {
    return std::make_tuple(dnnl::memory::data_type::bf16,
                           dnnl::memory::data_type::f8_e4m3);
  }
};
using bias_type_t = at::native::onednn::bias_type_t;
using trans_type_t = at::native::onednn::trans_type_t;
using GpuEngineManager = at::native::onednn::GpuEngineManager;
using primitive_ext = at::native::onednn::primitive_ext;
using primitive_cache =
    at::native::onednn::lru_cache<dnnl::memory::dims, primitive_ext>;

template <trans_type_t Tt>
static inline void get_strides(memory::dims& src_strides,
                               memory::dims& wei_strides,
                               memory::dims& dst_strides, const int64_t lda,
                               const int64_t ldb, const int64_t ldc) {}

template <>
static inline void get_strides<trans_type_t::nt>(memory::dims& src_strides,
                                                 memory::dims& wei_strides,
                                                 memory::dims& dst_strides,
                                                 const int64_t lda,
                                                 const int64_t ldb,
                                                 const int64_t ldc) {
  src_strides = {lda, 1};
  wei_strides = {1, ldb};
  dst_strides = {ldc, 1};
}

template <>
static inline void get_strides<trans_type_t::nn>(memory::dims& src_strides,
                                                 memory::dims& wei_strides,
                                                 memory::dims& dst_strides,
                                                 const int64_t lda,
                                                 const int64_t ldb,
                                                 const int64_t ldc) {
  src_strides = {lda, 1};
  wei_strides = {ldb, 1};
  dst_strides = {ldc, 1};
}

template <trans_type_t Tt, joint_dtypes_t Ts, typename F>
struct matmul_primitive_cache_t {
  static inline primitive_ext& get(
      const int m, const int n, const int k, const int64_t lda,
      const int64_t ldb, const int64_t ldc,
      const bias_type_t
          b_dims,  // for shapeless bias, not put it into template parameter
      const int device_id, F f_attr, const int scale_group_size,
      const int zp_group_size) {
    auto& cached = get_cache(device_id);
    dnnl::memory::dims src_strides, wei_strides, dst_strides;
    get_strides<Tt>(src_strides, wei_strides, dst_strides, lda, ldb, ldc);
    auto pri_key = at::native::onednn::concat(src_strides, wei_strides, m, n, k,
                                              int(b_dims), scale_group_size,
                                              zp_group_size);
    auto iter = cached.find(pri_key);
    if (iter == cached.end()) {
      auto [src_dt, wei_dt] = onednn_types_mapper<Ts>::get();

      auto src_md = dnnl::memory::desc({m, k}, src_dt, src_strides);
      auto wei_md = dnnl::memory::desc({k, n}, wei_dt, wei_strides);
      // TODO: should decide dst dt in a better way?
      auto dst_dt = (((src_dt == dnnl::memory::data_type::s8 ||
                       src_dt == dnnl::memory::data_type::u8) &&
                      (wei_dt == dnnl::memory::data_type::u4))
                         ? dnnl::memory::data_type::f16
                         : src_dt);
      auto dst_md = dnnl::memory::desc({m, n}, dst_dt, dst_strides);
      auto bias_format = b_dims == bias_type_t::none
                             ? dnnl::memory::format_tag::undef
                             : dnnl::memory::format_tag::ab;
      auto bias_md = dnnl::memory::desc(
          at::native::onednn::get_bias_type(b_dims, m, n), dst_dt,
          bias_format);  // {m, n} or {1, n}

      primitive_attr pattr;
      f_attr(pattr);

      dnnl::matmul::primitive_desc matmul_pd;
      at::Device curDevice = at::Device(at::kXPU, device_id);
      auto aengine = GpuEngineManager::Instance().get_engine(curDevice);
      if (b_dims == bias_type_t::none) {
        matmul_pd = dnnl::matmul::primitive_desc(aengine, src_md, wei_md,
                                                 dst_md, pattr);
      } else {
        matmul_pd = dnnl::matmul::primitive_desc(aengine, src_md, wei_md,
                                                 bias_md, dst_md, pattr);
      }

      return cached.insert({pri_key, primitive_ext(dnnl::matmul(matmul_pd))})
          .first->second;
    } else {
      return iter->second;
    }
  }

 private:
  static constexpr int max_cache_capacity = 512;
  // if default constructor of primitive cache could read the environment
  // variable then it'll save a lot of trouble
  static inline thread_local std::array<primitive_cache, 16> mappings;

  // this won't be needed if primitive_cache have good default constructor
  static inline primitive_cache& get_cache(const int device_id) {
    auto& mapping = mappings[device_id];
    if (mapping.max_size() == 0) {
      mapping.resize(max_cache_capacity);
    }
    return mapping;
  }
};

template <joint_dtypes_t Ts, typename F>
static inline primitive_ext& matmul_primitive_create_and_cache(
    const trans_type_t Tt, const bias_type_t b_dims, const int m, const int n,
    const int k, const int64_t lda, const int64_t ldb, const int64_t ldc,
    const int device_id, F attr, const int scale_group_size,
    const int zp_group_size) {
  switch (Tt) {
    case trans_type_t::nt:
      return matmul_primitive_cache_t<trans_type_t::nt, Ts, F>::get(
          m, n, k, lda, ldb, ldc, b_dims, device_id, attr, scale_group_size,
          zp_group_size);
    case trans_type_t::nn:
      return matmul_primitive_cache_t<trans_type_t::nn, Ts, F>::get(
          m, n, k, lda, ldb, ldc, b_dims, device_id, attr, scale_group_size,
          zp_group_size);
    default:
      throw std::runtime_error("unsupported trans type ...");
  }
}

template <typename F>
static inline primitive_ext& matmul_primitive_create_and_cache(
    const joint_dtypes_t Ts, const trans_type_t Tt, const bias_type_t b_dims,
    const int m, const int n, const int k, const int64_t lda,
    const int64_t ldb,  // is weight ldb necessary?
    const int64_t ldc, const int device_id, F attr,
    const int scale_group_size = 1, const int zp_group_size = 1) {
  switch (Ts) {
    case joint_dtypes_t::f16_int4:
      return matmul_primitive_create_and_cache<joint_dtypes_t::f16_int4, F>(
          Tt, b_dims, m, n, k, lda, ldb, ldc, device_id, attr, scale_group_size,
          zp_group_size);
    case joint_dtypes_t::bf16_int4:
      return matmul_primitive_create_and_cache<joint_dtypes_t::bf16_int4, F>(
          Tt, b_dims, m, n, k, lda, ldb, ldc, device_id, attr, scale_group_size,
          zp_group_size);
    case joint_dtypes_t::s8_int4:
      return matmul_primitive_create_and_cache<joint_dtypes_t::s8_int4, F>(
          Tt, b_dims, m, n, k, lda, ldb, ldc, device_id, attr, scale_group_size,
          zp_group_size);
    case joint_dtypes_t::u8_int4:
      return matmul_primitive_create_and_cache<joint_dtypes_t::u8_int4, F>(
          Tt, b_dims, m, n, k, lda, ldb, ldc, device_id, attr, scale_group_size,
          zp_group_size);
    case joint_dtypes_t::f16_f8_e5m2:
      return matmul_primitive_create_and_cache<joint_dtypes_t::f16_f8_e5m2, F>(
          Tt, b_dims, m, n, k, lda, ldb, ldc, device_id, attr, scale_group_size,
          zp_group_size);
    case joint_dtypes_t::bf16_f8_e5m2:
      return matmul_primitive_create_and_cache<joint_dtypes_t::bf16_f8_e5m2, F>(
          Tt, b_dims, m, n, k, lda, ldb, ldc, device_id, attr, scale_group_size,
          zp_group_size);
    case joint_dtypes_t::f16_f8_e4m3:
      return matmul_primitive_create_and_cache<joint_dtypes_t::f16_f8_e4m3, F>(
          Tt, b_dims, m, n, k, lda, ldb, ldc, device_id, attr, scale_group_size,
          zp_group_size);
    case joint_dtypes_t::bf16_f8_e4m3:
      return matmul_primitive_create_and_cache<joint_dtypes_t::bf16_f8_e4m3, F>(
          Tt, b_dims, m, n, k, lda, ldb, ldc, device_id, attr, scale_group_size,
          zp_group_size);
    default:
      throw std::runtime_error("Only support int4 and fp8 gemm ...");
  }
}

}  // namespace oneDNN