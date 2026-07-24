#include "fused_moe_up_launcher.hpp"

namespace FusedMOE {
void fused_moe_up_w4a16_silu(const FusedMOEUpLaunchParams& params) {
  FusedMOEUpLaunch<ActivationType::SILU, FusedMOEWeightType::W4A16>(params);
}
}  // namespace FusedMOE
