# Example implementation (this would typically be in C/C++ code that uses the allocator)
# The actual my_malloc and my_free functions are exported from the C extension

def example_usage():
    """
    Example of how to use the xpumem_allocator module.
    Note: The actual my_malloc and my_free functions need to be called from C/C++ code.
    """
    # Try to import from vllm_xpu_kernels package first
    try:
        from vllm_xpu_kernels import xpumem_allocator
    except ImportError:
        # Fallback to direct import if installed to system path
        import xpumem_allocator
    
    # Track allocations in a dictionary
    allocation_map = {}
    
    def python_malloc(device, size, ptr, handle):
        """Callback when memory is allocated."""
        print(f"[Python] Allocating {size} bytes on device {device}, ptr=0x{ptr:x}")
        allocation_map[ptr] = (device, size)
        return None
    
    def python_free(ptr):
        """Callback when memory is freed. Must return (device, size, ptr, handle)."""
        print(f"[Python] Freeing ptr=0x{ptr:x}")
        if ptr in allocation_map:
            device, size = allocation_map.pop(ptr)
            return (device, size, ptr, 0)
        else:
            # Fallback if not found
            return (0, 0, ptr, 0)
    
    # Initialize the module with callbacks
    xpumem_allocator.init_module(python_malloc, python_free)
    
    print("Module initialized. C/C++ code can now use my_malloc and my_free.")
    print("These functions will call the Python callbacks for tracking.")
    device_id = 0
    size_bytes = 1024

    try:
        xpumem_allocator.python_create_and_allocate(device_id, size_bytes, 0, 0)
    except Exception as e:
        print(f"{allocation fails: str(e)}")
        return

    # check allocation_map
    assert len(allocation_map) == 1, "allocation not tracked"
    ptr = next(iter(allocation_map))
    device, size = allocation_map[ptr]
    print(f"allocation succeeds: ptr=0x{ptr:x}, device={device}, size={size}")

    xpumem_allocator.python_unmap_and_release(device, size, ptr, 0)
    print("after release, allocation_map:", allocation_map)


if __name__ == "__main__":
    example_usage()
