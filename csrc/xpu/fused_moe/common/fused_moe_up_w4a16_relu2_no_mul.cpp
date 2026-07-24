#include "fused_moe_up_launcher.hpp"

namespace FusedMOE {
void fused_moe_up_w4a16_relu2_no_mul(const FusedMOEUpLaunchParams& params) {
  FusedMOEUpLaunch<ActivationType::RELU2_NO_MUL, FusedMOEWeightType::W4A16>(
      params);
}
}  // namespace FusedMOE
