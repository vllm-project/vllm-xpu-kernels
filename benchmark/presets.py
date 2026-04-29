# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Hardware Presets
hardware_presets: dict[str, dict[str, float]] = {
    "b60": {
        "tflops": 98, # BF16
        "memory_bandwidth_GBs": 456,
    },
    "b70": {
        "tflops": 182, # BF16
        "memory_bandwidth_GBs": 608,
    },
}

def get_hardware_preset(device_name: str) -> dict[str, float] | None:
    device_name = device_name.lower().split()
    for token in device_name:
         if token in hardware_presets:
            return hardware_presets[token]
    return None
