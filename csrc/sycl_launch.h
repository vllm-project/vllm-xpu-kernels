#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <utility>

// Central SYCL kernel submission layer.
//
// Every kernel launch in this project should go through
// vllm::xpu::sycl_kernel_submit() instead of calling queue::submit(),
// handler::parallel_for() or the enqueue-functions entry points directly.
// Funnelling all launches through one header means the submission mechanism
// can be changed project-wide in a single place: today that is
// sycl_ext_oneapi_enqueue_functions (handler-free nd_launch), tomorrow it may
// be free function kernels, a profiling wrapper, or a graph recorder.
//
// Backend selection:
//   VLLM_XPU_USE_ENQUEUE_FUNCTIONS=1  handler-free nd_launch / parallel_for
//   VLLM_XPU_USE_ENQUEUE_FUNCTIONS=0  classic queue::submit + parallel_for
// It defaults to the extension when the toolchain advertises it, and can be
// forced either way from the build system for A/B comparison.

#if !defined(VLLM_XPU_USE_ENQUEUE_FUNCTIONS)
  #if defined(SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS)
    #define VLLM_XPU_USE_ENQUEUE_FUNCTIONS 1
  #else
    #define VLLM_XPU_USE_ENQUEUE_FUNCTIONS 0
  #endif
#endif

namespace vllm {
namespace xpu {

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

// Tag for kernels that need command-group-scoped setup, the usual case being
// a sycl::local_accessor for shared local memory. Such kernels cannot be
// launched handler-free, so derive the functor from this tag and give it
//
//   void sycl_ker_config_convention(sycl::handler& cgh);
//
// sycl_kernel_submit() then routes it through queue::submit() while leaving
// every other kernel on the cheaper handler-free path. Keeping these kernels
// on the same entry point means they can be moved to
// work_group_scratch_memory later without touching the call sites.
struct SyclKernelConfigConvention {};

namespace detail {

template <typename T>
inline constexpr bool needs_handler_v =
    std::is_base_of_v<SyclKernelConfigConvention, std::decay_t<T>>;

template <typename T>
struct is_nd_range : std::false_type {};

template <int Dims>
struct is_nd_range<sycl::nd_range<Dims>> : std::true_type {};

// The one function that decides how a kernel actually reaches the device.
// Every public overload below funnels into it.
template <typename KernelName, typename RangeT, typename KernelT>
inline void
launch_impl(sycl::queue queue, const RangeT& range, KernelT kernel) {
  constexpr bool named = !std::is_void_v<KernelName>;

#if VLLM_XPU_USE_ENQUEUE_FUNCTIONS
  if constexpr (!needs_handler_v<KernelT>) {
    if constexpr (is_nd_range<RangeT>::value) {
      if constexpr (named) {
        syclexp::nd_launch<KernelName>(queue, range, kernel);
      } else {
        syclexp::nd_launch(queue, range, kernel);
      }
    } else {
      if constexpr (named) {
        syclexp::parallel_for<KernelName>(queue, range, kernel);
      } else {
        syclexp::parallel_for(queue, range, kernel);
      }
    }
    return;
  }
#endif

  queue.submit([&](sycl::handler& cgh) {
    if constexpr (needs_handler_v<KernelT>) {
      kernel.sycl_ker_config_convention(cgh);
    }
    if constexpr (named) {
      cgh.parallel_for<KernelName>(range, kernel);
    } else {
      cgh.parallel_for(range, kernel);
    }
  });
}

}  // namespace detail

// Launch over a sycl::nd_range or a plain sycl::range.
//
// KernelName is optional and only needed when a kernel must be nameable, for
// instance to query it with sycl::info::kernel_device_specific. Lambdas and
// functors are otherwise named implicitly.
template <typename KernelName = void, typename RangeT, typename KernelT>
inline void sycl_kernel_submit(
    sycl::queue queue, const RangeT& range, const KernelT& kernel) {
  detail::launch_impl<KernelName>(queue, range, kernel);
}

// Launch over an nd_range built from separate global and local ranges.
template <typename KernelName = void, int Dims, typename KernelT>
inline void sycl_kernel_submit(
    sycl::queue queue,
    sycl::range<Dims> global,
    sycl::range<Dims> local,
    const KernelT& kernel) {
  detail::launch_impl<KernelName>(
      queue, sycl::nd_range<Dims>(global, local), kernel);
}

// 1-D convenience overload. global is the total number of work items, not the
// number of work groups.
template <typename KernelName = void, typename KernelT>
inline void sycl_kernel_submit(
    sycl::queue queue, int64_t global, int64_t local, const KernelT& kernel) {
  detail::launch_impl<KernelName>(
      queue,
      sycl::nd_range<1>(
          sycl::range<1>(static_cast<size_t>(global)),
          sycl::range<1>(static_cast<size_t>(local))),
      kernel);
}

#if VLLM_XPU_USE_ENQUEUE_FUNCTIONS

// Launch a free function kernel, that is a plain function annotated with
//
//   SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<Dims>))
//
// which reads its index space through syclext::this_work_item::get_nd_item.
// Free function kernels take their arguments by value rather than through a
// captured functor, and request shared local memory with slm_bytes instead of
// a local_accessor. See
// sycl_ext_oneapi_free_function_kernels in the DPC++ extension docs.
template <auto* KPtr, int Dims, typename... KArgs>
inline void sycl_kernel_submit_fn(
    sycl::queue queue,
    sycl::range<Dims> global,
    sycl::range<Dims> local,
    size_t slm_bytes,
    KArgs&&... args) {
  const sycl::nd_range<Dims> range(global, local);
  if (slm_bytes != 0) {
    syclexp::launch_config config{
        range,
        syclexp::properties{syclexp::work_group_scratch_size(slm_bytes)}};
    syclexp::nd_launch(
        queue,
        config,
        syclexp::kernel_function<KPtr>,
        std::forward<KArgs>(args)...);
  } else {
    syclexp::nd_launch(
        queue,
        range,
        syclexp::kernel_function<KPtr>,
        std::forward<KArgs>(args)...);
  }
}

// 1-D convenience overload for free function kernels.
template <auto* KPtr, typename... KArgs>
inline void sycl_kernel_submit_fn(
    sycl::queue queue,
    int64_t global,
    int64_t local,
    size_t slm_bytes,
    KArgs&&... args) {
  sycl_kernel_submit_fn<KPtr, 1>(
      queue,
      sycl::range<1>(static_cast<size_t>(global)),
      sycl::range<1>(static_cast<size_t>(local)),
      slm_bytes,
      std::forward<KArgs>(args)...);
}

#endif  // VLLM_XPU_USE_ENQUEUE_FUNCTIONS

}  // namespace xpu
}  // namespace vllm
