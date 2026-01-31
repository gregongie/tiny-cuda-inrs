"""
Setup script for tinycudann_inrs_fp32 - the float32 precision variant.

This package can be installed alongside tinycudann_inrs (fp16) for
applications requiring higher numerical precision.

Installation:
    cd bindings/torch_fp32
    pip install .
"""
import os
import re
from setuptools import setup
from pkg_resources import parse_version
import subprocess
import shutil
import sys
import torch
from glob import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def min_supported_compute_capability(cuda_version):
	if cuda_version >= parse_version("13.0"):
		return 75
	elif cuda_version >= parse_version("12.0"):
		return 50
	else:
		return 20

def max_supported_compute_capability(cuda_version):
	if cuda_version < parse_version("11.0"):
		return 75
	elif cuda_version < parse_version("11.1"):
		return 80
	elif cuda_version < parse_version("11.8"):
		return 86
	elif cuda_version < parse_version("12.8"):
		return 90
	else:
		return 120

# Find version of tinycudann by scraping CMakeLists.txt
with open(os.path.join(ROOT_DIR, "CMakeLists.txt"), "r") as cmakelists:
	for line in cmakelists.readlines():
		if line.strip().startswith("VERSION"):
			VERSION = line.split("VERSION")[-1].strip()
			break

print(f"Building PyTorch extension for tiny-cuda-nn (FP32) version {VERSION}")

ext_modules = []

if "TCNN_CUDA_ARCHITECTURES" in os.environ and os.environ["TCNN_CUDA_ARCHITECTURES"]:
	compute_capabilities = [int(x) for x in os.environ["TCNN_CUDA_ARCHITECTURES"].replace(";", ",").split(",")]
	print(f"Obtained compute capabilities {compute_capabilities} from environment variable TCNN_CUDA_ARCHITECTURES")
elif torch.cuda.is_available():
	major, minor = torch.cuda.get_device_capability()
	compute_capabilities = [major * 10 + minor]
	print(f"Obtained compute capability {compute_capabilities[0]} from PyTorch")
else:
	raise EnvironmentError("Unknown compute capability. Specify the target compute capabilities in the TCNN_CUDA_ARCHITECTURES environment variable or install PyTorch with the CUDA backend to detect it automatically.")

include_networks = True
if "--no-networks" in sys.argv:
	include_networks = False
	sys.argv.remove("--no-networks")
	print("Building >> without << neural networks (just the input encodings)")

if os.name == "nt":
	def find_cl_path():
		import glob
		for executable in ["Program Files (x86)", "Program Files"]:
			for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
				paths = sorted(glob.glob(f"C:\\{executable}\\Microsoft Visual Studio\\*\\{edition}\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64"), reverse=True)
				if paths:
					return paths[0]

	# If cl.exe is not on path, try to find it.
	if os.system("where cl.exe >nul 2>nul") != 0:
		cl_path = find_cl_path()
		if cl_path is None:
			raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
		os.environ["PATH"] += ";" + cl_path
	else:
		os.environ["DISTUTILS_USE_SDK"] = "1"

cpp_standard = 14

# Get CUDA version and make sure the targeted compute capability is compatible
if os.system("nvcc --version") == 0:
	nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode()
	cuda_version = re.search(r"release (\S+),", nvcc_out)

	if cuda_version:
		cuda_version = parse_version(cuda_version.group(1))
		print(f"Detected CUDA version {cuda_version}")
		if cuda_version >= parse_version("11.0"):
			cpp_standard = 17

		supported_compute_capabilities = [
			cc for cc in compute_capabilities if cc >= min_supported_compute_capability(cuda_version) and cc <= max_supported_compute_capability(cuda_version)
		]

		if not supported_compute_capabilities:
			supported_compute_capabilities = [max_supported_compute_capability(cuda_version)]

		if supported_compute_capabilities != compute_capabilities:
			print(f"WARNING: Compute capabilities {compute_capabilities} are not all supported by the installed CUDA version {cuda_version}. Targeting {supported_compute_capabilities} instead.")
			compute_capabilities = supported_compute_capabilities

min_compute_capability = min(compute_capabilities)

print(f"Targeting C++ standard {cpp_standard}")

base_nvcc_flags = [
	f"-std=c++{cpp_standard}",
	"--extended-lambda",
	"--use_fast_math",
	"--expt-relaxed-constexpr",
	"-U__CUDA_NO_HALF_OPERATORS__",
	"-U__CUDA_NO_HALF_CONVERSIONS__",
	"-U__CUDA_NO_HALF2_OPERATORS__",
]

if os.name == "posix":
	base_cflags = [f"-std=c++{cpp_standard}"]
	base_nvcc_flags += [
		"-Xcompiler=-Wno-float-conversion",
		"-Xcompiler=-fno-strict-aliasing",
	]
elif os.name == "nt":
	base_cflags = [f"/std:c++{cpp_standard}"]


# Some containers set this to contain old architectures that won't compile.
os.environ["TORCH_CUDA_ARCH_LIST"] = ""

# List of sources.
bindings_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(bindings_dir, "../.."))

base_definitions = [
	"-DTCNN_PARAMS_UNALIGNED",
	"-DTCNN_RTC",
	"-DTCNN_RTC_USE_FAST_MATH",
	# Force FP32 precision - this is the key difference from the fp16 build
	"-DTCNN_HALF_PRECISION=0",
]

print("Forcing TCNN_HALF_PRECISION=0 (FP32 precision)")

base_source_files = [
	"tinycudann_inrs_fp32/bindings.cpp",
	"../../dependencies/fmt/src/format.cc",
	"../../dependencies/fmt/src/os.cc",
	"../../src/cpp_api.cu",
	"../../src/common_host.cu",
	"../../src/encoding.cu",
	"../../src/object.cu",
	"../../src/rtc_kernel.cu",
]

if include_networks:
	base_source_files += [
		"../../src/network.cu",
		"../../src/cutlass_mlp.cu",
	]
	# Note: FullyFusedMLP is NOT included for FP32 builds because it
	# requires __half precision (hardcoded in kernels)
else:
	base_definitions.append("-DTCNN_NO_NETWORKS")

# Copy headers required by RTC at runtime
rtc_dir = os.path.join(bindings_dir, "tinycudann_inrs_fp32", "rtc")
rtc_include_dir = os.path.join(rtc_dir, "include")
rtc_cache_dir = os.path.join(rtc_dir, "cache")
shutil.rmtree(rtc_dir, ignore_errors=True)
os.makedirs(rtc_include_dir, exist_ok=True)
os.makedirs(rtc_cache_dir, exist_ok=True)

nvcc_path = shutil.which("nvcc")
if nvcc_path is None:
	print(f"WARNING: could not find CUDA include directory. JIT compilation will not be supported.")
else:
	cuda_include_dir = os.path.join(os.path.dirname(os.path.dirname(nvcc_path)), "include")

	cuda_headers = glob(f"{cuda_include_dir}/cuda_fp16*") + glob(f"{cuda_include_dir}/vector*")
	tcnn_headers = glob(f"{root_dir}/include/tiny-cuda-nn/*", recursive=True)
	pcg32_headers = glob(f"{root_dir}/dependencies/pcg32/*")

	def copy_files(whence, files):
		for h in files:
			if not os.path.isfile(h):
				continue
			tgt = os.path.join(rtc_include_dir, os.path.relpath(h, whence))
			os.makedirs(os.path.dirname(tgt), exist_ok=True)
			shutil.copyfile(h, tgt)

	copy_files(cuda_include_dir, cuda_headers)
	copy_files(f"{root_dir}/include", tcnn_headers)
	copy_files(f"{root_dir}/dependencies", pcg32_headers)

def make_extension(compute_capability):
	nvcc_flags = base_nvcc_flags + [f"-gencode=arch=compute_{compute_capability},code={code}_{compute_capability}" for code in ["compute", "sm"]]
	definitions = base_definitions + [f"-DTCNN_MIN_GPU_ARCH={compute_capability}"]

	# Note: FullyFusedMLP is excluded for FP32 - only CutlassMLP supports float
	source_files = base_source_files

	nvcc_flags = nvcc_flags + definitions
	cflags = base_cflags + definitions

	ext = CUDAExtension(
		name=f"tinycudann_inrs_fp32_bindings._{compute_capability}_C",
		sources=source_files,
		include_dirs=[
			f"{root_dir}/include",
			f"{root_dir}/dependencies",
			f"{root_dir}/dependencies/cutlass/include",
			f"{root_dir}/dependencies/cutlass/tools/util/include",
			f"{root_dir}/dependencies/fmt/include",
		],
		extra_compile_args={"cxx": cflags, "nvcc": nvcc_flags},
		libraries=["cuda", "nvrtc"],
	)
	return ext

ext_modules = [make_extension(comp) for comp in compute_capabilities]

def package_files(directory):
	paths = []
	for (path, _, filenames) in os.walk(directory):
		for filename in filenames:
			paths.append(os.path.join('..', path, filename))
	return paths

setup(
	name="tinycudann_inrs_fp32",
	version=VERSION,
	description="tiny-cuda-nn extension for PyTorch (FP32 precision)",
	long_description="tiny-cuda-nn extension for PyTorch with float32 precision for MLPs. "
	                 "Can be installed alongside tinycudann_inrs (fp16) for applications "
	                 "requiring higher numerical precision.",
	classifiers=[
		"Development Status :: 4 - Beta",
		"Environment :: GPU :: NVIDIA CUDA",
		"License :: BSD 3-Clause",
		"Programming Language :: C++",
		"Programming Language :: CUDA",
		"Programming Language :: Python :: 3 :: Only",
		"Topic :: Multimedia :: Graphics",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"Topic :: Scientific/Engineering :: Image Processing",
	],
	keywords="PyTorch,cutlass,machine learning,float32",
	url="https://github.com/nvlabs/tiny-cuda-nn",
	author="Thomas Müller, Jacob Munkberg, Jon Hasselgren, Or Perel",
	author_email="tmueller@nvidia.com, jmunkberg@nvidia.com, jhasselgren@nvidia.com, operel@nvidia.com",
	maintainer="Thomas Müller",
	maintainer_email="tmueller@nvidia.com",
	download_url=f"https://github.com/nvlabs/tiny-cuda-nn",
	license="BSD 3-Clause \"New\" or \"Revised\" License",
	packages=["tinycudann_inrs_fp32"],
	package_data={"": package_files(rtc_dir)},
	install_requires=[],
	include_package_data=True,
	zip_safe=False,
	ext_modules=ext_modules,
	cmdclass={"build_ext": BuildExtension}
)
