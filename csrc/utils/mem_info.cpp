#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

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

#ifdef ZE_DEVICE_USABLEMEM_SIZE_PROPERTIES_EXT_NAME
static bool driverSupportsUsableMem(ze_driver_handle_t driver) {
  uint32_t extCount = 0;
  if (zeDriverGetExtensionProperties(driver, &extCount, nullptr) !=
      ZE_RESULT_SUCCESS) {
    return false;
  }
  std::vector<ze_driver_extension_properties_t> extProps(extCount);
  if (zeDriverGetExtensionProperties(driver, &extCount, extProps.data()) !=
      ZE_RESULT_SUCCESS) {
    return false;
  }
  for (const auto& ext : extProps) {
    if (std::strcmp(ext.name, ZE_DEVICE_USABLEMEM_SIZE_PROPERTIES_EXT_NAME) ==
            0 &&
        ext.version >= ZE_DEVICE_USABLEMEM_SIZE_PROPERTIES_EXT_VERSION_1_0) {
      return true;
    }
  }
  return false;
}
#endif

// Returns nullopt when the usable-memory extension is unavailable, which is a
// different thing from a device that genuinely has zero usable memory left.
std::optional<size_t>
getUsableMemory(ze_device_handle_t& device, ze_driver_handle_t& driver) {
#ifdef ZE_DEVICE_USABLEMEM_SIZE_PROPERTIES_EXT_NAME
  if (!driverSupportsUsableMem(driver)) {
    return std::nullopt;
  }
  ze_device_properties_t deviceProperties{};
  ze_device_usablemem_size_ext_properties_t usableMemProps{};

  usableMemProps.stype = ZE_STRUCTURE_TYPE_DEVICE_USABLEMEM_SIZE_EXT_PROPERTIES;
  deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  deviceProperties.pNext = &usableMemProps;

  if (zeDeviceGetProperties(device, &deviceProperties) != ZE_RESULT_SUCCESS) {
    return std::nullopt;
  }
  return usableMemProps.currUsableMemSize;
#else
  return std::nullopt;
#endif
}

std::tuple<int64_t, int64_t> getMemoryInfo(int64_t device_index) {
  const auto& device =
      c10::xpu::get_raw_device(static_cast<c10::DeviceIndex>(device_index));
  auto level_zero_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
  auto level_zero_driver =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          device.get_platform());

  size_t free = 0;
  size_t total = 0;
  if (auto usable = getUsableMemory(level_zero_device, level_zero_driver)) {
    free = *usable;
    total = getTotalMemory(level_zero_device);
  } else {
    std::tie(free, total) = c10::xpu::XPUCachingAllocator::get()->getMemoryInfo(
        static_cast<c10::DeviceIndex>(device_index));
  }

  if (total > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      free > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    std::cerr << "Memory size exceeds int64_t max value!" << std::endl;
    return {-1, -1};  // or handle this case as appropriate
  }
  return {static_cast<int64_t>(free), static_cast<int64_t>(total)};
}
