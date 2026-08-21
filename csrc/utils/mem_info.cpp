#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <iostream>

// Level Zero headers predating the device-usablemem-size-properties extension
// declare neither of these, which is a compile error rather than something a
// runtime check can guard, so spell out the spec-fixed stype and layout here.
namespace {
constexpr auto kUsableMemStype = static_cast<ze_structure_type_t>(0x00020041);

struct UsableMemProps {
  ze_structure_type_t stype;
  void* pNext;
  uint64_t currUsableMemSize;
};
}  // namespace

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

// Zero means the extension is unavailable: a driver that does not implement it
// ignores the unrecognized pNext and still reports success, leaving the
// zero-initialized field untouched.
size_t getUsableMemory(ze_device_handle_t& device) {
  ze_device_properties_t deviceProperties{};
  UsableMemProps usableMemProps{};

  usableMemProps.stype = kUsableMemStype;
  deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  deviceProperties.pNext = &usableMemProps;

  if (zeDeviceGetProperties(device, &deviceProperties) != ZE_RESULT_SUCCESS) {
    return 0;
  }
  return usableMemProps.currUsableMemSize;
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
