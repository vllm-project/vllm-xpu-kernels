#pragma once

namespace FusedMOE {
static constexpr int MaxThreadsPerSM = 512;
static constexpr int sub_group_size = 16;
static constexpr int grf_size = 256;
static constexpr int systolic_m = 8;
}  // namespace FusedMOE