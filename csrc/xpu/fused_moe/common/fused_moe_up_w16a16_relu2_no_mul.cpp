#include "fused_moe_up_launcher.hpp"

namespace FusedMOE {
void fused_moe_up_w16a16_relu2_no_mul(const FusedMOEUpLaunchParams& params) {
  FusedMOEUpLaunch<ActivationType::RELU2_NO_MUL, FusedMOEWeightType::W16A16>(
      params);
}
}  // namespace FusedMOE
