#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <iostream>

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

// A driver that does not implement the extension leaves the zero-initialized
// size untouched and still returns success, so zero means unavailable.
size_t getUsableMemory(ze_device_handle_t& device) {
#ifdef ZE_DEVICE_USABLEMEM_SIZE_PROPERTIES_EXT_NAME
  ze_device_properties_t deviceProperties{};
  ze_device_usablemem_size_ext_properties_t usableMemProps{};

  usableMemProps.stype = ZE_STRUCTURE_TYPE_DEVICE_USABLEMEM_SIZE_EXT_PROPERTIES;
  deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  deviceProperties.pNext = &usableMemProps;

  zeDeviceGetProperties(device, &deviceProperties);
  return usableMemProps.currUsableMemSize;
#else
  return 0;
#endif
}

std::tuple<int64_t, int64_t> getMemoryInfo(int64_t device_index) {
  const auto& device =
      c10::xpu::get_raw_device(static_cast<c10::DeviceIndex>(device_index));
  auto level_zero_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);

  size_t free = getUsableMemory(level_zero_device);
  size_t total = 0;
  if (free == 0) {
    std::tie(free, total) = c10::xpu::XPUCachingAllocator::get()->getMemoryInfo(
        static_cast<c10::DeviceIndex>(device_index));
  } else {
    total = getTotalMemory(level_zero_device);
  }

  if (total > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      free > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    std::cerr << "Memory size exceeds int64_t max value!" << std::endl;
    return {-1, -1};  // or handle this case as appropriate
  }
  return {static_cast<int64_t>(free), static_cast<int64_t>(total)};
}
