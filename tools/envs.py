# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    CMAKE_BUILD_TYPE: Optional[str] = None
    VLLM_TARGET_DEVICE: str = "xpu"
    MAX_JOBS: Optional[str] = None
    VLLM_USE_PRECOMPILED: bool = False
    VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL: bool = False
    VERBOSE: bool = False


def get_default_cache_root():
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root():
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def get_vllm_port() -> Optional[int]:
    """Get the port from VLLM_PORT environment variable.

    Returns:
        The port number as an integer if VLLM_PORT is set, None otherwise.

    Raises:
        ValueError: If VLLM_PORT is a URI, suggest k8s service discovery issue.
    """
    if 'VLLM_PORT' not in os.environ:
        return None

    port = os.getenv('VLLM_PORT', '0')

    try:
        return int(port)
    except ValueError as err:
        from urllib.parse import urlparse
        try:
            parsed = urlparse(port)
            if parsed.scheme:
                raise ValueError(
                    f"VLLM_PORT '{port}' appears to be a URI. "
                    "This may be caused by a Kubernetes service discovery issue"
                    "check the warning in: https://docs.vllm.ai/en/stable/usage/env_vars.html"
                )
        except Exception:
            pass

        raise ValueError(
            f"VLLM_PORT '{port}' must be a valid integer") from err


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# --8<-- [start:env-vars-definition]

environment_variables: dict[str, Callable[[], Any]] = {

    # ================== Installation Time Env Vars ==================

    # Target device of vLLM, supporting [xpu (by default)]
    "VLLM_TARGET_DEVICE":
    lambda: os.getenv("VLLM_TARGET_DEVICE", "xpu"),

    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),

    # If set, vllm-xpu-kernels will use precompiled binaries (*.so)
    "VLLM_USE_PRECOMPILED":
    lambda: os.environ.get("VLLM_USE_PRECOMPILED", "").strip().lower() in
    ("1", "true") or bool(os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION")),

    # Whether to force using nightly wheel in python build.
    # This is used for testing the nightly wheel in python build.
    "VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL":
    lambda: bool(int(os.getenv("VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL", "0"))
                 ),

    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),

    # ================== Kernel Build Options ==================
    # These control which kernel extensions are built. Set to "0" or "OFF"
    # to disable. They are forwarded to CMake as -D flags by setup.py.

    # Build SYCL-TLA based kernels (attention, grouped_gemm shared libs)
    "BUILD_SYCL_TLA_KERNELS":
    lambda: os.getenv("BUILD_SYCL_TLA_KERNELS", "ON"),

    # Architecture options
    "VLLM_XPU_ENABLE_XE2":
    lambda: os.getenv("VLLM_XPU_ENABLE_XE2", "ON"),
    "VLLM_XPU_ENABLE_XE_DEFAULT":
    lambda: os.getenv("VLLM_XPU_ENABLE_XE_DEFAULT", "ON"),

    # Individual kernel extension toggles
    "BASIC_KERNELS_ENABLED":
    lambda: os.getenv("BASIC_KERNELS_ENABLED", "ON"),
    "FA2_KERNELS_ENABLED":
    lambda: os.getenv("FA2_KERNELS_ENABLED", "ON"),
    "MOE_KERNELS_ENABLED":
    lambda: os.getenv("MOE_KERNELS_ENABLED", "ON"),
    "GDN_KERNELS_ENABLED":
    lambda: os.getenv("GDN_KERNELS_ENABLED", "ON"),
    "XPU_SPECIFIC_KERNELS_ENABLED":
    lambda: os.getenv("XPU_SPECIFIC_KERNELS_ENABLED", "ON"),
    "XPUMEM_ALLOCATOR_ENABLED":
    lambda: os.getenv("XPUMEM_ALLOCATOR_ENABLED", "ON"),

    # If set, vllm will print verbose logs during installation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),

    # Root directory for vLLM configuration files
    # Defaults to `~/.config/vllm` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how vllm finds its configuration files
    # during runtime, but also affects how vllm installs its configuration
    # files during **installation**.
    "VLLM_CONFIG_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "vllm"),
        )),
}

# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
