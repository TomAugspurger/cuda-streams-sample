"""Setup script for nvcomp_minimal."""

import os
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


class CustomBuildExt(build_ext):
    """Custom build_ext to filter out problematic compiler flags."""

    def build_extensions(self):
        # Remove problematic flags from compiler
        if hasattr(self.compiler, "compiler_so"):
            self.compiler.compiler_so = [
                arg
                for arg in self.compiler.compiler_so
                if not arg.startswith("-fdebug-default-version")
            ]
        super().build_extensions()


# Try to find CUDA and nvcomp paths
cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
nvcomp_home = os.environ.get("NVCOMP_HOME", None)

# Try common nvcomp locations if not specified
if nvcomp_home is None:
    # First, try nvidia-nvcomp-cu12 wheel installation
    site_packages = sysconfig.get_path("purelib")
    wheel_nvcomp = os.path.join(site_packages, "nvidia", "nvcomp")

    if os.path.exists(os.path.join(wheel_nvcomp, "include", "nvcomp.h")):
        print(f"Found nvcomp from nvidia-nvcomp-cu12 wheel at: {wheel_nvcomp}")
        nvcomp_home = wheel_nvcomp
    else:
        # Try system paths
        for possible_path in [
            "/usr/local/nvcomp",
            "/usr/local",
            "/opt/nvcomp",
            os.path.expanduser("~/.local"),
        ]:
            if os.path.exists(os.path.join(possible_path, "include", "nvcomp.h")):
                nvcomp_home = possible_path
                break

if nvcomp_home is None:
    print("Warning: Could not find nvcomp. Set NVCOMP_HOME environment variable.")
    print("Trying to build anyway assuming system paths...")
    nvcomp_include = []
    nvcomp_libdir = []
else:
    nvcomp_include = [os.path.join(nvcomp_home, "include")]
    # For wheel installation, libs are in the same directory
    if "site-packages" in nvcomp_home:
        nvcomp_libdir = [nvcomp_home]
    else:
        nvcomp_libdir = [os.path.join(nvcomp_home, "lib")]

cuda_include = [os.path.join(cuda_home, "include")]
cuda_libdir = [os.path.join(cuda_home, "lib64")]

# Determine library name - wheel uses libnvcomp.so.5, system may use libnvcomp.so
nvcomp_lib = "nvcomp"
if nvcomp_libdir:
    lib_path = nvcomp_libdir[0]
    if os.path.exists(os.path.join(lib_path, "libnvcomp.so.5")):
        # Use direct reference to versioned library for wheel installation
        nvcomp_lib = ":libnvcomp.so.5"
    elif os.path.exists(os.path.join(lib_path, "libnvcomp.so")):
        nvcomp_lib = "nvcomp"

extensions = [
    Extension(
        name="nvcomp_minimal.zstd",
        sources=["nvcomp_minimal/zstd.pyx"],
        include_dirs=nvcomp_include + cuda_include,
        library_dirs=nvcomp_libdir + cuda_libdir,
        libraries=[nvcomp_lib, "cudart"],
        runtime_library_dirs=nvcomp_libdir + cuda_libdir,
        language="c++",
        extra_compile_args=["-std=c++11"],
    )
]

setup(
    name="nvcomp-minimal",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
    cmdclass={"build_ext": CustomBuildExt},
)
