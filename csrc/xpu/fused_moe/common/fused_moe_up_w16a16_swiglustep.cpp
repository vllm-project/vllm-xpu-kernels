#include "fused_moe_up_launcher.hpp"

namespace FusedMOE {
void fused_moe_up_w16a16_swiglustep(const FusedMOEUpLaunchParams& params) {
  FusedMOEUpLaunch<ActivationType::SWIGLUSTEP, FusedMOEWeightType::W16A16>(
      params);
}
}  // namespace FusedMOE
