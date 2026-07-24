#include <torch/all.h>
#include "csrc/utils.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe2/fused_moe_xe2_policy.h"
  #include "xe2/xe2_utils.h"
#endif
#include "fused_moe_kernel.hpp"
#include "fused_moe_up.h"

// #pragma clang diagnostic ignored "-Wpass-failed"
// #pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace FusedMOE {
using namespace cute;

// type tag to define a unique sycl kernel name
template <
    bool,
    ActivationType,
    typename,
    typename,
    typename,
    typename,
    char,
    char,
    class>
class FusedMOEUpName;

template <
    bool has_clamping,
    ActivationType activation_type,
    char layoutA,
    char layoutB,
    class policy,
    typename ElementA,
    typename ElementB,
    typename ElementS,
    typename ElementBI,
    typename ElementD>
void FusedMOEUpLauncher(
    sycl::queue& queue,
    const ElementA* activations,
    const ElementB* weights,
    const ElementS* scales,
    const ElementBI* bias,
    ElementD* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* rows_per_expert,
    const int num_experts,
    const int group_size,
    int32_t* atomic_buffer,
    const double gemm1_clamp_limit) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<systolic_m, float, ElementA_non_CV>{};

  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;
  auto mma = MMA{};

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(mma);

  TORCH_CHECK(
      MaxThreadsPerSM % MaxThreadsPerWorkgroup == 0,
      "MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup");

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(
      1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{
      syclex::sub_group_size<sub_group_size>, intelex::grf_size<grf_size>};

  using GmemTiledCopyA = typename policy::GmemTiledCopyA;
  using GmemTiledCopyB = typename policy::GmemTiledCopyB;
  using GmemTiledCopyD = typename policy::GmemTiledCopyD;

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<FusedMOEUpName<
        has_clamping,
        activation_type,
        ElementA,
        ElementB,
        ElementS,
        ElementD,
        layoutA,
        layoutB,
        policy>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          FusedMOEUp<
              has_clamping,
              activation_type,
              GmemTiledCopyA,
              GmemTiledCopyB,
              GmemTiledCopyD,
              layoutA,
              layoutB,
              'R'>(
              activations,
              weights,
              scales,
              bias,
              outputs,
              mma,
              rows_per_expert,
              num_experts,
              group_size,
              gemm_n,
              gemm_k,
              atomic_buffer,
              local_mem,
              gemm1_clamp_limit);
        });
  });
}
}  // namespace FusedMOE

using namespace FusedMOE;

torch::Tensor fused_moe_up(
    torch::Tensor& ptr_A,
    const c10::optional<at::Tensor>& ptr_A_scale,
    torch::Tensor& ptr_B,
    const c10::optional<at::Tensor>& ptr_B_scale,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor& ptr_D,
    torch::Tensor& rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    std::string activation,
    double gemm1_clamp_limit) {
  auto& dpcpp_queue =
      at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();
  auto A_dtype = ptr_A.dtype();
  auto B_dtype = ptr_B.dtype();
  bool is_weight_fp8 =
      ((B_dtype == at::kFloat8_e4m3fn) || (B_dtype == at::kFloat8_e5m2));
  bool is_B_int4 = (B_dtype == at::kChar) && ptr_B_scale.has_value();
  bool is_B_mxfp4 =
      (B_dtype == at::kFloat4_e2m1fn_x2) && ptr_B_scale.has_value();

  TORCH_CHECK(ptr_A.dim() == 2, "ptr_A must be 2D [Total_M, K]");
  TORCH_CHECK(ptr_B.dim() == 3, "ptr_B must be 3D [num_experts, K, 2 * N]");
  TORCH_CHECK(ptr_D.dim() == 2, "ptr_D must be 2D [Total_M, N]");

  if (ptr_bias.has_value()) {
    TORCH_CHECK(
        ptr_bias->dim() == 2, "ptr_bias must be 2D [num_experts, 2 * N]");
  }

  TORCH_CHECK(ptr_A.is_contiguous(), "ptr_A must be contiguous");
  TORCH_CHECK(ptr_B.is_contiguous(), "ptr_B must be contiguous");
  TORCH_CHECK(ptr_D.is_contiguous(), "ptr_D must be contiguous");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(ptr_bias->is_contiguous(), "ptr_bias must be contiguous");
  }

  int A_total_M = ptr_A.size(0);
  int A_K = ptr_A.size(1);

  int B_E = ptr_B.size(0);
  int B_K = ptr_B.size(1);
  int B_N = ptr_B.size(2);
  if (is_B_int4 || is_B_mxfp4) {
    B_K = ptr_B.size(2) * 2;
    B_N = ptr_B.size(1);
  }

  int D_total_M = ptr_D.size(0);
  int D_N = ptr_D.size(1);
  int group_size = -1;
  int A_avg_M = A_total_M / num_experts;

  TORCH_CHECK(B_E == num_experts, "ptr_B.size(0) must match num_experts");
  TORCH_CHECK(A_total_M == D_total_M, "ptr_A.size(0) must match ptr_D.size(0)");
  TORCH_CHECK(A_K == B_K && B_K == K, "ptr_A.size(1) must match ptr_B.size(1)");
  TORCH_CHECK(
      B_N == 2 * D_N && D_N == N,
      "ptr_B.size(2) must match double ptr_D.size(1)");

  if (ptr_bias.has_value()) {
    TORCH_CHECK(
        ptr_bias->size(0) == num_experts,
        "ptr_bias.size(0) must match num_experts");
    TORCH_CHECK(
        ptr_bias->size(1) == 2 * N, "ptr_bias.size(1) must match double N");
  }

  at::Tensor atomic_buffer =
      at::empty({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));

  // Clamping is only applied when a positive limit is provided.
  const bool has_clamping = gemm1_clamp_limit > 0;

  // Validate scales up-front (once) based on the weight dtype.
  if (is_B_int4 || is_B_mxfp4) {
    TORCH_CHECK(ptr_B_scale.has_value(), "w4a16 grouped gemm must have scales");
    TORCH_CHECK(ptr_B_scale->is_contiguous(), "ptr_B_scale must be contiguous");
    TORCH_CHECK(
        ptr_B_scale->dim() == 3,
        "ptr_B_scale of int4 must be 3D [num_experts, 2 * N, group_num]");
    TORCH_CHECK(
        ptr_B_scale->size(0) == num_experts,
        "ptr_B_scale.size(0) of int4 must match num_experts");
    TORCH_CHECK(
        K % ptr_B_scale->size(2) == 0,
        "K must be divisible by ptr_B_scale.size(2) (group_num) of int4");
    TORCH_CHECK(
        ptr_B_scale->size(1) == 2 * N,
        "ptr_B_scale.size(1) of int4 must match 2 * N");
    int group_num = ptr_B_scale->size(2);
    group_size = K / group_num;

    TORCH_CHECK(
        group_size == 32 || group_size == 64 || group_size == 128 ||
            group_size == 256,
        "group_size must be 32, 64, 128 or 256");
  } else if (is_weight_fp8) {
    TORCH_CHECK(ptr_B_scale.has_value(), "w8a16 grouped gemm must have scales");
    TORCH_CHECK(ptr_B_scale->is_contiguous(), "ptr_B_scale must be contiguous");
    TORCH_CHECK(
        ptr_B_scale->dim() == 1, "ptr_B_scale of fp8 must be 1D [num_experts]");
    TORCH_CHECK(
        ptr_B_scale->size(0) == num_experts,
        "ptr_B_scale.size(0) of fp8 must match num_experts");
    TORCH_CHECK(
        ptr_B_scale->dtype() == at::kFloat, "ptr_B_scale must be float");
  } else {
    TORCH_CHECK(
        !ptr_B_scale.has_value(), "w16a16 grouped gemm must not have scales");
  }

#define FusedMOEUpLauncherCallER(                                              \
    HasClamping,                                                               \
    ActType,                                                                   \
    LayoutA,                                                                   \
    LayoutB,                                                                   \
    Policy,                                                                    \
    ElementA,                                                                  \
    ElementB,                                                                  \
    ElementS)                                                                  \
  FusedMOEUpLauncher<HasClamping, ActType, LayoutA, LayoutB, Policy>(          \
      dpcpp_queue,                                                             \
      reinterpret_cast<ElementA*>(ptr_A.data_ptr()),                           \
      reinterpret_cast<ElementB*>(ptr_B.data_ptr()),                           \
      ptr_B_scale.has_value()                                                  \
          ? reinterpret_cast<ElementS*>(ptr_B_scale->data_ptr())               \
          : static_cast<ElementS*>(nullptr),                                   \
      ptr_bias.has_value() ? reinterpret_cast<ElementA*>(ptr_bias->data_ptr()) \
                           : static_cast<ElementA*>(nullptr),                  \
      reinterpret_cast<ElementA*>(ptr_D.data_ptr()),                           \
      N,                                                                       \
      K,                                                                       \
      reinterpret_cast<int*>(rows_per_expert.data_ptr()),                      \
      num_experts,                                                             \
      group_size,                                                              \
      static_cast<int*>(atomic_buffer.data_ptr()),                             \
      gemm1_clamp_limit);

#define W4A16LauncherCallER(HasClamping, ActType, policy)                      \
  if (is_B_int4) {                                                             \
    if (A_dtype == at::kBFloat16) {                                            \
      using scalar_t = bfloat16_t;                                             \
      FusedMOEUpLauncherCallER(                                                \
          HasClamping,                                                         \
          ActType,                                                             \
          'R',                                                                 \
          'C',                                                                 \
          policy,                                                              \
          scalar_t,                                                            \
          uint8_t,                                                             \
          scalar_t);                                                           \
    } else if (A_dtype == at::kHalf) {                                         \
      using scalar_t = half_t;                                                 \
      FusedMOEUpLauncherCallER(                                                \
          HasClamping,                                                         \
          ActType,                                                             \
          'R',                                                                 \
          'C',                                                                 \
          policy,                                                              \
          scalar_t,                                                            \
          uint8_t,                                                             \
          scalar_t);                                                           \
    }                                                                          \
  } else if (is_B_mxfp4) {                                                     \
    if (A_dtype == at::kBFloat16) {                                            \
      using scalar_t = bfloat16_t;                                             \
      FusedMOEUpLauncherCallER(                                                \
          HasClamping, ActType, 'R', 'C', policy, scalar_t, uint8_t, uint8_t); \
    } else if (A_dtype == at::kHalf) {                                         \
      using scalar_t = half_t;                                                 \
      FusedMOEUpLauncherCallER(                                                \
          HasClamping, ActType, 'R', 'C', policy, scalar_t, uint8_t, uint8_t); \
    }                                                                          \
  }

#define W8A16LauncherCallER(HasClamping, ActType, policy)                 \
  if (B_dtype == at::kFloat8_e4m3fn && A_dtype == at::kHalf) {            \
    using scalar_t = half_t;                                              \
    FusedMOEUpLauncherCallER(                                             \
        HasClamping,                                                      \
        ActType,                                                          \
        'R',                                                              \
        'R',                                                              \
        policy,                                                           \
        scalar_t,                                                         \
        float_e4m3_t,                                                     \
        float);                                                           \
  } else if (B_dtype == at::kFloat8_e5m2 && A_dtype == at::kHalf) {       \
    using scalar_t = half_t;                                              \
    FusedMOEUpLauncherCallER(                                             \
        HasClamping,                                                      \
        ActType,                                                          \
        'R',                                                              \
        'R',                                                              \
        policy,                                                           \
        scalar_t,                                                         \
        float_e5m2_t,                                                     \
        float);                                                           \
  } else if (B_dtype == at::kFloat8_e4m3fn && A_dtype == at::kBFloat16) { \
    using scalar_t = bfloat16_t;                                          \
    FusedMOEUpLauncherCallER(                                             \
        HasClamping,                                                      \
        ActType,                                                          \
        'R',                                                              \
        'R',                                                              \
        policy,                                                           \
        scalar_t,                                                         \
        float_e4m3_t,                                                     \
        float);                                                           \
  } else if (B_dtype == at::kFloat8_e5m2 && A_dtype == at::kBFloat16) {   \
    using scalar_t = bfloat16_t;                                          \
    FusedMOEUpLauncherCallER(                                             \
        HasClamping,                                                      \
        ActType,                                                          \
        'R',                                                              \
        'R',                                                              \
        policy,                                                           \
        scalar_t,                                                         \
        float_e5m2_t,                                                     \
        float);                                                           \
  }

#define W16A16LauncherCallER(HasClamping, ActType, policy)                     \
  if (A_dtype == at::kBFloat16) {                                              \
    using scalar_t = bfloat16_t;                                               \
    FusedMOEUpLauncherCallER(                                                  \
        HasClamping, ActType, 'R', 'R', policy, scalar_t, scalar_t, scalar_t); \
  } else if (A_dtype == at::kHalf) {                                           \
    using scalar_t = half_t;                                                   \
    FusedMOEUpLauncherCallER(                                                  \
        HasClamping, ActType, 'R', 'R', policy, scalar_t, scalar_t, scalar_t); \
  }

#define DISPATCH_MOE_UP(HasClamping, ActType)               \
  if (is_B_int4 || is_B_mxfp4) {                            \
    if (A_avg_M <= 4) {                                     \
      using policy = w4a16_policy_m_8;                      \
      W4A16LauncherCallER(HasClamping, ActType, policy);    \
    } else if (A_avg_M <= 8) {                              \
      using policy = w4a16_policy_m_16;                     \
      W4A16LauncherCallER(HasClamping, ActType, policy);    \
    } else if (A_avg_M <= 128) {                            \
      using policy = w4a16_policy_m_32;                     \
      W4A16LauncherCallER(HasClamping, ActType, policy);    \
    } else {                                                \
      using policy = w4a16_policy;                          \
      W4A16LauncherCallER(HasClamping, ActType, policy);    \
    }                                                       \
  } else if (is_weight_fp8) {                               \
    if (A_avg_M <= 8) {                                     \
      using policy = w8a16_policy_m_16;                     \
      W8A16LauncherCallER(HasClamping, ActType, policy);    \
    } else if (A_avg_M <= 32) {                             \
      using policy = w8a16_policy_m_32;                     \
      W8A16LauncherCallER(HasClamping, ActType, policy);    \
    } else {                                                \
      using policy = w8a16_policy;                          \
      W8A16LauncherCallER(HasClamping, ActType, policy);    \
    }                                                       \
  } else {                                                  \
    if (A_avg_M <= 8) {                                     \
      using policy = w16a16_policy_m_16;                    \
      W16A16LauncherCallER(HasClamping, ActType, policy);   \
    } else if (A_avg_M <= 16) {                             \
      using policy = w16a16_policy_m_32;                    \
      W16A16LauncherCallER(HasClamping, ActType, policy);   \
    } else {                                                \
      if (B_N <= 64) {                                      \
        using policy = w16a16_policy_n_64;                  \
        W16A16LauncherCallER(HasClamping, ActType, policy); \
      } else if (B_N <= 512) {                              \
        using policy = w16a16_policy_n_128;                 \
        W16A16LauncherCallER(HasClamping, ActType, policy); \
      } else {                                              \
        using policy = w16a16_policy;                       \
        W16A16LauncherCallER(HasClamping, ActType, policy); \
      }                                                     \
    }                                                       \
  }

#define DISPATCH_MOE_UP_ACT(ActType) \
  if (has_clamping) {                \
    DISPATCH_MOE_UP(true, ActType);  \
  } else {                           \
    DISPATCH_MOE_UP(false, ActType); \
  }

  if (activation == "silu") {
    DISPATCH_MOE_UP_ACT(ActivationType::SILU);
  } else if (activation == "gelu") {
    DISPATCH_MOE_UP_ACT(ActivationType::GELU);
  } else if (activation == "gelu_tanh") {
    DISPATCH_MOE_UP_ACT(ActivationType::GELU_TANH);
  } else if (activation == "swigluoai") {
    DISPATCH_MOE_UP_ACT(ActivationType::SWIGLUOAI);
  } else if (activation == "relu2_no_mul") {
    DISPATCH_MOE_UP_ACT(ActivationType::RELU2_NO_MUL);
  } else if (activation == "swiglustep") {
    DISPATCH_MOE_UP_ACT(ActivationType::SWIGLUSTEP);
  } else {
    TORCH_CHECK(false, "Unsupported activation: ", activation);
  }

#undef DISPATCH_MOE_UP_ACT
#undef DISPATCH_MOE_UP
#undef W16A16LauncherCallER
#undef W8A16LauncherCallER
#undef W4A16LauncherCallER
#undef FusedMOEUpLauncherCallER

  return ptr_D;
}