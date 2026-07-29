#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <iostream>

#ifdef VLLM_XPU_HAS_ZE_USABLEMEM
size_t getTotalMemory(ze_device_handle_t& device) {
  uint32_t memoryCount = 0;
  zeDeviceGetMemoryProperties(device, &memoryCount, nullptr);
  auto pMemoryProperties = new ze_device_memory_properties_t[memoryCount];
  for (uint32_t mem = 0; mem < memoryCount; ++mem) {
    pMemoryProperties[mem].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    pMemoryProperties[mem].pNext = nullptr;
  }
  zeDeviceGetMemoryProperties(device, &memoryCount, pMemoryProperties);
  size_t totalMemory = 0;
  for (uint32_t mem = 0; mem < memoryCount; ++mem) {
    totalMemory += pMemoryProperties[mem].totalSize;
  }
  delete[] pMemoryProperties;

  return totalMemory;
}

size_t getUsableMemory(ze_device_handle_t& device) {
  ze_device_properties_t deviceProperties{};
  ze_device_usablemem_size_ext_properties_t usableMemProps{};

  usableMemProps.stype = ZE_STRUCTURE_TYPE_DEVICE_USABLEMEM_SIZE_EXT_PROPERTIES;
  deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  deviceProperties.pNext = &usableMemProps;

  zeDeviceGetProperties(device, &deviceProperties);
  return usableMemProps.currUsableMemSize;
}
#endif

std::tuple<int64_t, int64_t> getMemoryInfo(int64_t device_index) {
#ifdef VLLM_XPU_HAS_ZE_USABLEMEM
  const auto& device =
      c10::xpu::get_raw_device(static_cast<c10::DeviceIndex>(device_index));
  auto level_zero_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
  const size_t free = getUsableMemory(level_zero_device);
  const size_t total = getTotalMemory(level_zero_device);
#else
  // Level Zero loader < 1.27.0: the device-usablemem-size extension is not
  // available in the headers. Defer to PyTorch's XPU caching allocator, which
  // reports {free, total} and raises a descriptive error if the device cannot
  // report free memory.
  const auto [free, total] =
      c10::xpu::XPUCachingAllocator::get()->getMemoryInfo(
          static_cast<c10::DeviceIndex>(device_index));
#endif

  if (total > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      free > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    std::cerr << "Memory size exceeds int64_t max value!" << std::endl;
    return {-1, -1};  // or handle this case as appropriate
  }
  return {static_cast<int64_t>(free), static_cast<int64_t>(total)};
}
