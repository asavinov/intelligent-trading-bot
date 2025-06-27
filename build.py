import os
import re
import shutil
import sys
import platform
import datetime as dt
from pathlib import Path
import numpy as np
from Cython.Compiler import Options
from Cython.Build import cythonize
from Cython.Compiler.Version import version as cython_compiler_version
from setuptools import Distribution, Extension

# Platform constants
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"
IS_ARM64 = platform.machine() == "arm64"

# Configuration flags
PROFILE_MODE = bool(os.getenv("PROFILE_MODE", ""))
ANNOTATION_MODE = bool(os.getenv("ANNOTATION_MODE", ""))

# Build directory
if PROFILE_MODE:
    BUILD_DIR = None
elif ANNOTATION_MODE:
    BUILD_DIR = "build/annotated"
else:
    BUILD_DIR = "build/optimized"

################################################################################
#  CYTHON BUILD
################################################################################
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html

Options.docstrings = True  # Include docstrings in modules
Options.fast_fail = True  # Abort the compilation on the first error occurred
Options.annotate = ANNOTATION_MODE  # Create annotated HTML files for each .pyx
Options.warning_errors = True
Options.extra_warnings = True

CYTHON_COMPILER_DIRECTIVES = {
    "language_level": "3",
    "cdivision": True,
    "nonecheck": True,
    "embedsignature": True,
    "profile": PROFILE_MODE,
    "linetrace": PROFILE_MODE,
    "warn.maybe_uninitialized": True,
}

# TODO: Temporarily separate Cython configuration while we require v3.0.11 for coverage
if cython_compiler_version == "3.1.0a1":
    Options.warning_errors = True  # Treat compiler warnings as errors
    Options.extra_warnings = True
    CYTHON_COMPILER_DIRECTIVES["warn.deprecated.IF"] = False

compile_args = ["-O3"]
link_args = []
include_dirs = [np.get_include()]
libraries = ["m"]

def _get_version() -> str:
    with open("pyproject.toml", encoding="utf-8") as f:
        pyproject_content = f.read().strip()
    if not pyproject_content:
        raise ValueError("pyproject.toml is empty or not properly formatted")

    version_match = re.search(r'version\s*=\s*"(.*?)"', pyproject_content)
    if not version_match:
        raise ValueError("Version not found in pyproject.toml")

    return version_match.group(1)


def build_extensions() -> list[Extension]:
    define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    if PROFILE_MODE or ANNOTATION_MODE:
        define_macros.append(("CYTHON_TRACE", "1"))

    extra_compile_args = []
    extra_link_args = []
    if platform.system() != "Windows":
        extra_compile_args.append("-Wno-unreachable-code")
        if not PROFILE_MODE:
            extra_compile_args.extend(["-O3", "-pipe"])

    if IS_WINDOWS:
        extra_link_args += [
            "AdvAPI32.Lib",
            "bcrypt.lib",
            "Crypt32.lib",
            "Iphlpapi.lib",
            "Kernel32.lib",
            "ncrypt.lib",
            "Netapi32.lib",
            "ntdll.lib",
            "Ole32.lib",
            "OleAut32.lib",
            "Pdh.lib",
            "PowrProf.lib",
            "Propsys.lib",
            "Psapi.lib",
            "runtimeobject.lib",
            "schannel.lib",
            "secur32.lib",
            "Shell32.lib",
            "User32.Lib",
            "UserEnv.Lib",
            "WS2_32.Lib",
        ]

    print("Creating C extension modules...")
    print(f"define_macros={define_macros}")
    print(f"extra_compile_args={extra_compile_args}")
    
    return [
        Extension(
            name=str(pyx.relative_to(".")).replace(os.path.sep, ".")[:-4],
            sources=[str(pyx)],
            include_dirs=[np.get_include()],
            define_macros=define_macros,
            language="c",
            extra_link_args=extra_link_args,
            extra_compile_args=extra_compile_args,
        )
        for pyx in Path("intelligent_trading_bot").rglob("*.pyx")
    ]


def copy_build_to_source(cmd) -> None:
    for output in cmd.get_outputs():
        relative_extension = Path(output).relative_to(cmd.build_lib)
        if Path(output).exists():
            shutil.copyfile(output, relative_extension)
            mode = relative_extension.stat().st_mode
            mode |= (mode & 0o444) >> 2
            relative_extension.chmod(mode)
    print("Copied all compiled dynamic library files into the source directory")


def build() -> None:
    extensions = build_extensions()
    if not extensions:
        raise ValueError(
            "No extensions found to build. Ensure .pyx files are in the correct location."
        )

    ext_modules = cythonize(
        extensions,
        compiler_directives=CYTHON_COMPILER_DIRECTIVES,
        nthreads=os.cpu_count(),
        build_dir=BUILD_DIR,
    )
    if not ext_modules:
        raise ValueError("Cythonize returned no extensions. Check your configurations.")

    distribution = Distribution(
        {
            "name": "intelligent_trading_bot",
            "ext_modules": ext_modules,
            "zip_safe": False,
        }
    )
    
    cmd = distribution.get_command_obj("build_ext")
    cmd.inplace = True
    if not getattr(cmd, "extensions", None):
        cmd.extensions = ext_modules  # Fallback to manually set extensions
        
    cmd.ensure_finalized()
    cmd.cython_include_dirs = cmd.cython_include_dirs or [] 
    cmd.run()

    copy_build_to_source(cmd)


if __name__ == "__main__":
    print("\033[36m")
    print("=====================================================================")
    print(f"Sage Builder {_get_version()}")
    print(
        "=====================================================================\033[0m"
    )
    print(f"System: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Cython: {cython_compiler_version}")
    print(f"NumPy:  {np.__version__}")

    print(f"\nPROFILE_MODE={PROFILE_MODE}")
    print(f"ANNOTATION_MODE={ANNOTATION_MODE}")
    print(f"BUILD_DIR={BUILD_DIR}")
    print("\nStarting build...")
    ts_start = dt.datetime.now(dt.timezone.utc)
    build()
    print(f"Build time: {dt.datetime.now(dt.timezone.utc) - ts_start}")
    print("\033[32m" + "Build completed" + "\033[0m")
