#include "utils.h"

using namespace vllm::xpu;

bool is_bmg_g21(int64_t device_index = -1) {
  at::DeviceIndex dev_idx = device_index;
  return vllm::xpu::is_bmg_g21(dev_idx);
}

bool is_bmg_g31(int64_t device_index = -1) {
  at::DeviceIndex dev_idx = device_index;
  return vllm::xpu::is_bmg_g31(dev_idx);
}

bool is_bmg(int64_t device_index = -1) {
  at::DeviceIndex dev_idx = device_index;
  return vllm::xpu::is_bmg(dev_idx);
}

bool is_pvc(int64_t device_index = -1) {
  at::DeviceIndex dev_idx = device_index;
  return vllm::xpu::is_pvc(dev_idx);
}

bool is_xe2_arch(int64_t device_index = -1) {
  at::DeviceIndex dev_idx = device_index;
  return vllm::xpu::is_xe2_arch(dev_idx);
}

bool is_xe3_arch(int64_t device_index = -1) {
  at::DeviceIndex dev_idx = device_index;
  return vllm::xpu::is_xe3_arch(dev_idx);
}
