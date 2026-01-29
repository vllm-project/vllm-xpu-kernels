#include "utils.h"

using namespace vllm::xpu;

bool is_bmg(int64_t device_index = -1) {
  at::DeviceIndex dev_idx = device_index;
  return vllm::xpu::is_bmg(dev_idx);
}

bool is_pvc(int64_t device_index = -1) {
  at::DeviceIndex dev_idx = device_index;
  return vllm::xpu::is_pvc(dev_idx);
}