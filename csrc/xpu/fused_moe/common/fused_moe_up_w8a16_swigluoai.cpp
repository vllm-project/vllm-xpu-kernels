#include "fused_moe_up_launcher.hpp"

namespace FusedMOE {
void fused_moe_up_w8a16_swigluoai(const FusedMOEUpLaunchParams& params) {
  FusedMOEUpLaunch<ActivationType::SWIGLUOAI, FusedMOEWeightType::W8A16>(
      params);
}
}  // namespace FusedMOE
