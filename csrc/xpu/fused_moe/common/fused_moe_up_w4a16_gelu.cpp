#include "fused_moe_up_launcher.hpp"

namespace FusedMOE {
void fused_moe_up_w4a16_gelu(const FusedMOEUpLaunchParams& params) {
  FusedMOEUpLaunch<ActivationType::GELU, FusedMOEWeightType::W4A16>(params);
}
}  // namespace FusedMOE
