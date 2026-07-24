#include "fused_moe_up_launcher.hpp"

namespace FusedMOE {
void fused_moe_up_w16a16_gelu_tanh(const FusedMOEUpLaunchParams& params) {
  FusedMOEUpLaunch<ActivationType::GELU_TANH, FusedMOEWeightType::W16A16>(
      params);
}
}  // namespace FusedMOE
