#pragma once

#include <torch/extension.h>
#include <c10/xpu/XPUFunctions.h>
#include <sycl/sycl.hpp>

namespace vllm::xpu {

/**
 * @class DummyAllocator
 * @brief A singleton allocator class for XPU memory management using SYCL
 * 
 * This class provides thread-safe memory allocation and deallocation for XPU
 * devices using SYCL's unified shared memory (USM). It tracks allocation and
 * deallocation call counts for debugging/monitoring purposes.
 * 
 * Usage example:
 *   // C++ style
 *   auto& allocator = vllm::xpu::DummyAllocator::Instance();
 *   void* ptr = allocator.allocate(1024, 0);
 *   allocator.free(ptr, 1024, 0);
 *   int count = allocator.get_alloc_count();
 *   
 *   // C style (backward compatible)
 *   void* ptr = dummy_alloc(1024, 0, nullptr);
 *   dummy_free(ptr, 1024, 0, nullptr);
 *   int count = called_dummy_alloc;  // Access via global variable
 */
class DummyAllocator {
 public:
  /**
   * @brief Get the singleton instance
   * @return Reference to the singleton instance
   */
  static DummyAllocator& Instance() {
    static DummyAllocator instance;
    return instance;
  }

  /**
   * @brief Allocate memory on XPU device
   * @param size Size of memory to allocate in bytes
   * @param device Device index
   * @param queue SYCL queue pointer (can be nullptr)
   * @return Pointer to allocated memory, or nullptr on failure
   */
  void* allocate(size_t size, int device, sycl::queue* queue = nullptr) {
    called_dummy_alloc_ = 123;
    auto& sycl_device = c10::xpu::get_raw_device(device);
    auto& sycl_context = c10::xpu::get_device_context();
    void* ptr = sycl::malloc_shared(size, sycl_device, sycl_context);
    return ptr;
  }

  /**
   * @brief Free memory on XPU device
   * @param ptr Pointer to memory to free
   * @param size Size of memory (for tracking purposes, not used by SYCL)
   * @param device Device index (for tracking purposes, not used by SYCL)
   * @param queue SYCL queue pointer (can be nullptr)
   */
  void free(void* ptr, size_t size, int device, sycl::queue* queue = nullptr) {
    called_dummy_free_ = 321;
    sycl::free(ptr, c10::xpu::get_device_context());
  }

  /**
   * @brief Get the number of times allocate() has been called
   * @return Allocation call count
   */
  int get_alloc_count() const { return called_dummy_alloc_; }

  /**
   * @brief Get the number of times free() has been called
   * @return Deallocation call count
   */
  int get_free_count() const { return called_dummy_free_; }

  /**
   * @brief Reset call counters
   */
  void reset_counters() {
    called_dummy_alloc_ = 0;
    called_dummy_free_ = 0;
  }

  // Delete copy constructor and assignment operator
  DummyAllocator(DummyAllocator const&) = delete;
  DummyAllocator& operator=(DummyAllocator const&) = delete;

 private:
  // Private constructor for singleton pattern
  DummyAllocator() : called_dummy_alloc_(0), called_dummy_free_(0) {}
  ~DummyAllocator() = default;

  int called_dummy_alloc_;  // Counter for allocation calls
  int called_dummy_free_;   // Counter for deallocation calls
};

}  // namespace vllm::xpu

// C interface for backward compatibility
extern "C" {
  // Global variables for C interface compatibility
  // These are automatically updated by the C interface functions below
  C10_EXPORT int called_dummy_alloc = 0;
  C10_EXPORT int called_dummy_free = 0;

  C10_EXPORT void* dummy_alloc(size_t size, int device, sycl::queue* queue) {
    auto& allocator = vllm::xpu::DummyAllocator::Instance();
    void* ptr = allocator.allocate(size, device, queue);
    // Update global variable to match allocator's internal state
    called_dummy_alloc = allocator.get_alloc_count();
    return ptr;
  }

  C10_EXPORT void dummy_free(void* ptr, size_t size, int device, sycl::queue* queue) {
    auto& allocator = vllm::xpu::DummyAllocator::Instance();
    allocator.free(ptr, size, device, queue);
    // Update global variable to match allocator's internal state
    called_dummy_free = allocator.get_free_count();
  }
}
