#include "fused_moe_up_launcher.hpp"

namespace FusedMOE {
void fused_moe_up_w16a16_gelu(const FusedMOEUpLaunchParams& params) {
  FusedMOEUpLaunch<ActivationType::GELU, FusedMOEWeightType::W16A16>(params);
}
}  // namespace FusedMOE
