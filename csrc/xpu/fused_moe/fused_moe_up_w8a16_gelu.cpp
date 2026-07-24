#include "fused_moe_up_launcher.hpp"

namespace FusedMOE {
void fused_moe_up_w8a16_gelu(const FusedMOEUpLaunchParams& params) {
  FusedMOEUpLaunch<ActivationType::GELU, FusedMOEWeightType::W8A16>(params);
}
}  // namespace FusedMOE
