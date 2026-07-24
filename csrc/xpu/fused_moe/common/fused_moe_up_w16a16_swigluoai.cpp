#include "fused_moe_up_launcher.hpp"

namespace FusedMOE {
void fused_moe_up_w16a16_swigluoai(const FusedMOEUpLaunchParams& params) {
  FusedMOEUpLaunch<ActivationType::SWIGLUOAI, FusedMOEWeightType::W16A16>(
      params);
}
}  // namespace FusedMOE
