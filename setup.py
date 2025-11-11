# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from shutil import which

from packaging.version import Version
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import SYCL_HOME


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

envs = load_module_from_path('envs', os.path.join(ROOT_DIR, 'tools',
                                                  'envs.py'))

VLLM_TARGET_DEVICE = envs.VLLM_TARGET_DEVICE


def is_sccache_available() -> bool:
    return which("sccache") is not None


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def is_url_available(url: str) -> bool:
    from urllib.request import urlopen

    status = None
    try:
        with urlopen(url) as f:
            status = f.status
    except Exception:
        return False
    return status == 200


def get_oneapi_version() -> Version:
    """Get the oneapi version from
    """
    assert SYCL_HOME is not None, "SYCL_HOME environment variable is not set."
    icpx_output = subprocess.check_output([SYCL_HOME + "/bin/icpx", "-v"],
                                          universal_newlines=True)
    print("=============== icpx version ===============")
    print(f"sycl home: {SYCL_HOME}")
    print(icpx_output)
    print("=============== icpx version ===============")
    # output = icpx_output.split()


def _build_custom_ops() -> bool:
    return True


class CMakeExtension(Extension):

    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=True, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = envs.MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()

        nvcc_threads = None
        get_oneapi_version()

        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "Release"
        cfg = envs.CMAKE_BUILD_TYPE or default_cfg

        cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),
            '-DVLLM_TARGET_DEVICE={}'.format(VLLM_TARGET_DEVICE),
            '-DCMAKE_TOOLCHAIN_FILE=cmake/toolchain.cmake'
        ]

        verbose = envs.VERBOSE
        if verbose:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=ON']

        if is_sccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=sccache',
            ]
        elif is_ccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ['-DVLLM_PYTHON_EXECUTABLE={}'.format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ['-DVLLM_PYTHON_PATH={}'.format(":".join(sys.path))]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ['-DFETCHCONTENT_BASE_DIR={}'.format(fc_base_dir)]

        #
        # Setup parallelism and build tool
        #
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ['-DNVCC_THREADS={}'.format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ['-G', 'Ninja']
            cmake_args += [
                '-DCMAKE_JOB_POOL_COMPILE:STRING=compile',
                '-DCMAKE_JOB_POOLS:STRING=compile={}'.format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []
        my_env = os.environ.copy()
        icx_path = shutil.which('icx')
        icpx_path = shutil.which('icpx')
        build_option_gpu = {
            "CMAKE_C_COMPILER": f"{icx_path}",
            "CMAKE_CXX_COMPILER": f"{icpx_path}",
        }
        for key, value in build_option_gpu.items():
            if value is not None:
                cmake_args.append("-D{}={}".format(key, value))
        subprocess.check_call(
            ['cmake', ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
            env=my_env)

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError as e:
            raise RuntimeError('Cannot find CMake executable') from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(s: str) -> str:
            return s.removeprefix("vllm_xpu_kernels.")

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == self.build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            # We assume only the final component of extension prefix is added by
            # CMake, this is currently true for current extensions but may not
            # always be the case.
            prefix = outdir
            if '.' in ext.name:
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = [
                "cmake", "--install", ".", "--prefix", prefix, "--component",
                target_name(ext.name)
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self):
        self.build_temp = "build/temp"
        # First, run the standard build_ext command to compile the extensions
        super().run()

        # copy vllm/vllm_flash_attn/**/*.py from self.build_lib to current
        # directory so that they can be included in the editable build
        import glob
        files = glob.glob(os.path.join(self.build_lib, "vllm",
                                       "vllm_flash_attn", "**", "*.py"),
                          recursive=True)
        for file in files:
            dst_file = os.path.join("vllm/vllm_flash_attn",
                                    file.split("vllm/vllm_flash_attn/")[-1])
            print(f"Copying {file} to {dst_file}")
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            self.copy_file(file, dst_file)


class precompiled_wheel_utils:
    """Extracts libraries and other files from an existing wheel."""

    @staticmethod
    def extract_precompiled_and_patch_package(wheel_url_or_path: str) -> dict:
        import tempfile
        import zipfile

        temp_dir = None
        try:
            if not os.path.isfile(wheel_url_or_path):
                wheel_filename = wheel_url_or_path.split("/")[-1]
                temp_dir = tempfile.mkdtemp(prefix="vllm-wheels")
                wheel_path = os.path.join(temp_dir, wheel_filename)
                print(f"Downloading wheel from {wheel_url_or_path}"
                      f"to {wheel_path}")
                from urllib.request import urlretrieve

                urlretrieve(wheel_url_or_path, filename=wheel_path)
            else:
                wheel_path = wheel_url_or_path
                print(f"Using existing wheel at {wheel_path}")

            package_data_patch = {}

            with zipfile.ZipFile(wheel_path) as wheel:
                files_to_copy = [
                    "vllm_xpu_kernels/_C.abi3.so",
                    "vllm_xpu_kernels/_moe_C.abi3.so",
                    "vllm_xpu_kernels/_vllm_fa2_C.abi3.so",
                    "vllm_xpu_kernels/_xpu_C.abi3.so",
                ]

                file_members = list(
                    filter(lambda x: x.filename in files_to_copy,
                           wheel.filelist))

                for file in file_members:
                    print(f"[extract] {file.filename}")
                    target_path = os.path.join(".", file.filename)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with (
                            wheel.open(file.filename) as src,
                            open(target_path, "wb") as dst,
                    ):
                        shutil.copyfileobj(src, dst)

                    pkg = os.path.dirname(file.filename).replace("/", ".")
                    package_data_patch.setdefault(pkg, []).append(
                        os.path.basename(file.filename))

            return package_data_patch
        finally:
            if temp_dir is not None:
                print(f"Removing temporary directory {temp_dir}")
                shutil.rmtree(temp_dir)

    # TODO: not used currently.
    @staticmethod
    def get_base_commit_in_main_branch() -> str:
        # Force to use the nightly wheel. This is mainly used for CI testing.
        if envs.VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL:
            return "nightly"

        try:
            # Get the latest commit hash of the upstream main branch.
            resp_json = subprocess.check_output([
                "curl",
                "-s",
                "https://api.github.com/repos/vllm-project/vllm/commits/main",
            ]).decode("utf-8")
            upstream_main_commit = json.loads(resp_json)["sha"]

            # In Docker build context, .git may be immutable or missing.
            if envs.VLLM_DOCKER_BUILD_CONTEXT:
                return upstream_main_commit

            # Check if the upstream_main_commit exists in the local repo
            try:
                subprocess.check_output(
                    ["git", "cat-file", "-e", f"{upstream_main_commit}"])
            except subprocess.CalledProcessError:
                # If not present, fetch it from the remote repository.
                # Note that this does not update any local branches,
                # but ensures that this commit ref and its history are
                # available in our local repo.
                subprocess.check_call([
                    "git", "fetch",
                    "https://github.com/vllm-project/vllm-xpu-kernels", "main"
                ])

            # Then get the commit hash of the current branch that is the same as
            # the upstream main commit.
            current_branch = (subprocess.check_output(
                ["git", "branch", "--show-current"]).decode("utf-8").strip())

            base_commit = (subprocess.check_output([
                "git", "merge-base", f"{upstream_main_commit}", current_branch
            ]).decode("utf-8").strip())
            return base_commit
        except ValueError as err:
            raise ValueError(err) from None
        except Exception as err:
            logger.warning(
                "Failed to get the base commit in the main branch. "
                "Using the nightly wheel. The libraries in this "
                "wheel may not be compatible with your dev branch: %s",
                err,
            )
            return "nightly"


package_data = {
    "vllm-xpu-kernels": [
        "py.typed",
    ]
}

# If using precompiled, extract and patch package_data (in advance of setup)
if envs.VLLM_USE_PRECOMPILED:
    # for now, we force use local wheel path
    wheel_location = os.getenv(
        "VLLM_PRECOMPILED_WHEEL_LOCATION",
        "./vllm_xpu_kernels-0.0.1-cp312-cp312-linux_x86_64.whl")
    if wheel_location is not None:
        wheel_url = wheel_location
    else:
        import platform

        arch = platform.machine()
        if arch == "x86_64":
            wheel_tag = "manylinux1_x86_64"
        elif arch == "aarch64":
            wheel_tag = "manylinux2014_aarch64"
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        base_commit = precompiled_wheel_utils.get_base_commit_in_main_branch()
        # TODO: update the URL when hosting the wheels
        wheel_url = "https://to-be-add.whl"
        nightly_wheel_url = ("https://to-be-add.whl")
        from urllib.request import urlopen

        try:
            with urlopen(wheel_url) as resp:
                if resp.status != 200:
                    wheel_url = nightly_wheel_url
        except Exception as e:
            print(f"[warn] Falling back to nightly wheel: {e}")
            wheel_url = nightly_wheel_url

    patch = precompiled_wheel_utils.extract_precompiled_and_patch_package(
        wheel_url)
    for pkg, files in patch.items():
        package_data.setdefault(pkg, []).extend(files)


class precompiled_build_ext(build_ext):
    """Disables extension building when using precompiled binaries."""

    def run(self) -> None:
        print("Skipping build")
        pass

    def build_extensions(self) -> None:
        print("Skipping build_ext: using precompiled extensions.")
        return


ext_modules = []

if _build_custom_ops():
    ext_modules.append(CMakeExtension(name="vllm_xpu_kernels._C"))
    ext_modules.append(CMakeExtension(name="vllm_xpu_kernels._vllm_fa2_C"))
    ext_modules.append(CMakeExtension(name="vllm_xpu_kernels._moe_C"))
    ext_modules.append(CMakeExtension(name="vllm_xpu_kernels._xpu_C"))

if ext_modules:
    cmdclass = {
        "build_ext":
        precompiled_build_ext if envs.VLLM_USE_PRECOMPILED else cmake_build_ext
    }

setup(
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    package_data=package_data,
)
