#pragma once

#include <torch/all.h>

#include "xe2/fused_moe_xe2_policy.h"
#include "xe2/xe2_utils.h"
#include "fused_moe_kernel.hpp"
#include "fused_moe_up_dispatch.h"

namespace FusedMOE {
using namespace cute;

// Type tag to define a unique SYCL kernel name.
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

#define FUSED_MOE_UP_LAUNCH(                                          \
    HasClamping,                                                      \
    ActType,                                                          \
    LayoutA,                                                          \
    LayoutB,                                                          \
    Policy,                                                           \
    ElementA,                                                         \
    ElementB,                                                         \
    ElementS)                                                         \
  FusedMOEUpLauncher<HasClamping, ActType, LayoutA, LayoutB, Policy>( \
      params.queue,                                                   \
      static_cast<const ElementA*>(params.activations),               \
      static_cast<const ElementB*>(params.weights),                   \
      static_cast<const ElementS*>(params.scales),                    \
      static_cast<const ElementA*>(params.bias),                      \
      static_cast<ElementA*>(params.outputs),                         \
      params.gemm_n,                                                  \
      params.gemm_k,                                                  \
      params.rows_per_expert,                                         \
      params.num_experts,                                             \
      params.group_size,                                              \
      params.atomic_buffer,                                           \
      params.gemm1_clamp_limit)

#define FUSED_MOE_UP_LAUNCH_W4(HasClamping, ActType, Policy)                   \
  if (params.uses_first_weight_encoding) {                                     \
    if (params.activations_are_bfloat16) {                                     \
      FUSED_MOE_UP_LAUNCH(                                                     \
          HasClamping,                                                         \
          ActType,                                                             \
          'R',                                                                 \
          'C',                                                                 \
          Policy,                                                              \
          bfloat16_t,                                                          \
          uint8_t,                                                             \
          bfloat16_t);                                                         \
    } else {                                                                   \
      FUSED_MOE_UP_LAUNCH(                                                     \
          HasClamping, ActType, 'R', 'C', Policy, half_t, uint8_t, half_t);    \
    }                                                                          \
  } else if (params.activations_are_bfloat16) {                                \
    FUSED_MOE_UP_LAUNCH(                                                       \
        HasClamping, ActType, 'R', 'C', Policy, bfloat16_t, uint8_t, uint8_t); \
  } else {                                                                     \
    FUSED_MOE_UP_LAUNCH(                                                       \
        HasClamping, ActType, 'R', 'C', Policy, half_t, uint8_t, uint8_t);     \
  }

#define FUSED_MOE_UP_LAUNCH_W8(HasClamping, ActType, Policy)                  \
  if (params.uses_first_weight_encoding) {                                    \
    if (params.activations_are_bfloat16) {                                    \
      FUSED_MOE_UP_LAUNCH(                                                    \
          HasClamping,                                                        \
          ActType,                                                            \
          'R',                                                                \
          'R',                                                                \
          Policy,                                                             \
          bfloat16_t,                                                         \
          float_e4m3_t,                                                       \
          float);                                                             \
    } else {                                                                  \
      FUSED_MOE_UP_LAUNCH(                                                    \
          HasClamping,                                                        \
          ActType,                                                            \
          'R',                                                                \
          'R',                                                                \
          Policy,                                                             \
          half_t,                                                             \
          float_e4m3_t,                                                       \
          float);                                                             \
    }                                                                         \
  } else if (params.activations_are_bfloat16) {                               \
    FUSED_MOE_UP_LAUNCH(                                                      \
        HasClamping,                                                          \
        ActType,                                                              \
        'R',                                                                  \
        'R',                                                                  \
        Policy,                                                               \
        bfloat16_t,                                                           \
        float_e5m2_t,                                                         \
        float);                                                               \
  } else {                                                                    \
    FUSED_MOE_UP_LAUNCH(                                                      \
        HasClamping, ActType, 'R', 'R', Policy, half_t, float_e5m2_t, float); \
  }

#define FUSED_MOE_UP_LAUNCH_W16(HasClamping, ActType, Policy)            \
  if (params.activations_are_bfloat16) {                                 \
    FUSED_MOE_UP_LAUNCH(                                                 \
        HasClamping,                                                     \
        ActType,                                                         \
        'R',                                                             \
        'R',                                                             \
        Policy,                                                          \
        bfloat16_t,                                                      \
        bfloat16_t,                                                      \
        bfloat16_t);                                                     \
  } else {                                                               \
    FUSED_MOE_UP_LAUNCH(                                                 \
        HasClamping, ActType, 'R', 'R', Policy, half_t, half_t, half_t); \
  }

template <ActivationType activation_type, FusedMOEWeightType weight_type>
void FusedMOEUpLaunch(const FusedMOEUpLaunchParams& params) {
  if constexpr (weight_type == FusedMOEWeightType::W4A16) {
    if (params.average_rows_per_expert <= 4) {
      using policy = w4a16_policy_m_8;
      if (params.has_clamping) {
        FUSED_MOE_UP_LAUNCH_W4(true, activation_type, policy);
      } else {
        FUSED_MOE_UP_LAUNCH_W4(false, activation_type, policy);
      }
    } else if (params.average_rows_per_expert <= 8) {
      using policy = w4a16_policy_m_16;
      if (params.has_clamping) {
        FUSED_MOE_UP_LAUNCH_W4(true, activation_type, policy);
      } else {
        FUSED_MOE_UP_LAUNCH_W4(false, activation_type, policy);
      }
    } else if (params.average_rows_per_expert <= 128) {
      using policy = w4a16_policy_m_32;
      if (params.has_clamping) {
        FUSED_MOE_UP_LAUNCH_W4(true, activation_type, policy);
      } else {
        FUSED_MOE_UP_LAUNCH_W4(false, activation_type, policy);
      }
    } else {
      using policy = w4a16_policy;
      if (params.has_clamping) {
        FUSED_MOE_UP_LAUNCH_W4(true, activation_type, policy);
      } else {
        FUSED_MOE_UP_LAUNCH_W4(false, activation_type, policy);
      }
    }
  } else if constexpr (weight_type == FusedMOEWeightType::W8A16) {
    if (params.average_rows_per_expert <= 8) {
      using policy = w8a16_policy_m_16;
      if (params.has_clamping) {
        FUSED_MOE_UP_LAUNCH_W8(true, activation_type, policy);
      } else {
        FUSED_MOE_UP_LAUNCH_W8(false, activation_type, policy);
      }
    } else if (params.average_rows_per_expert <= 32) {
      using policy = w8a16_policy_m_32;
      if (params.has_clamping) {
        FUSED_MOE_UP_LAUNCH_W8(true, activation_type, policy);
      } else {
        FUSED_MOE_UP_LAUNCH_W8(false, activation_type, policy);
      }
    } else {
      using policy = w8a16_policy;
      if (params.has_clamping) {
        FUSED_MOE_UP_LAUNCH_W8(true, activation_type, policy);
      } else {
        FUSED_MOE_UP_LAUNCH_W8(false, activation_type, policy);
      }
    }
  } else if (params.average_rows_per_expert <= 8) {
    using policy = w16a16_policy_m_16;
    if (params.has_clamping) {
      FUSED_MOE_UP_LAUNCH_W16(true, activation_type, policy);
    } else {
      FUSED_MOE_UP_LAUNCH_W16(false, activation_type, policy);
    }
  } else if (params.average_rows_per_expert <= 16) {
    using policy = w16a16_policy_m_32;
    if (params.has_clamping) {
      FUSED_MOE_UP_LAUNCH_W16(true, activation_type, policy);
    } else {
      FUSED_MOE_UP_LAUNCH_W16(false, activation_type, policy);
    }
  } else if (params.weight_n <= 64) {
    using policy = w16a16_policy_n_64;
    if (params.has_clamping) {
      FUSED_MOE_UP_LAUNCH_W16(true, activation_type, policy);
    } else {
      FUSED_MOE_UP_LAUNCH_W16(false, activation_type, policy);
    }
  } else if (params.weight_n <= 512) {
    using policy = w16a16_policy_n_128;
    if (params.has_clamping) {
      FUSED_MOE_UP_LAUNCH_W16(true, activation_type, policy);
    } else {
      FUSED_MOE_UP_LAUNCH_W16(false, activation_type, policy);
    }
  } else {
    using policy = w16a16_policy;
    if (params.has_clamping) {
      FUSED_MOE_UP_LAUNCH_W16(true, activation_type, policy);
    } else {
      FUSED_MOE_UP_LAUNCH_W16(false, activation_type, policy);
    }
  }
}

#undef FUSED_MOE_UP_LAUNCH_W16
#undef FUSED_MOE_UP_LAUNCH_W8
#undef FUSED_MOE_UP_LAUNCH_W4
#undef FUSED_MOE_UP_LAUNCH

}  // namespace FusedMOE
