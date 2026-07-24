#include "fused_moe_up_launcher.hpp"

namespace FusedMOE {
void fused_moe_up_w8a16_silu(const FusedMOEUpLaunchParams& params) {
  FusedMOEUpLaunch<ActivationType::SILU, FusedMOEWeightType::W8A16>(params);
}
}  // namespace FusedMOE
