# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Float32 precision variant of tiny-cuda-nn modules.

This module is nearly identical to tinycudann_inrs.modules, but loads the
fp32-compiled extension instead of the fp16 version.

Key differences from fp16 version:
- Uses float32 for MLP weights, biases, and computations
- Only CutlassMLP is available (FullyFusedMLP requires fp16)
- No tensor core acceleration (uses SIMT instead)
- Higher numerical precision at the cost of performance
"""

import gc
import importlib
import math
import os
import tempfile
import warnings
from typing import Optional, Literal, Union

import torch

ALL_COMPUTE_CAPABILITIES = [20, 21, 30, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 120]

if not torch.cuda.is_available():
	raise EnvironmentError("Unknown compute capability. Ensure PyTorch with CUDA support is installed.")

def _get_device_compute_capability(idx):
	major, minor = torch.cuda.get_device_capability(idx)
	return major * 10 + minor

def _get_system_compute_capability():
	num_devices = torch.cuda.device_count()
	device_capability = [_get_device_compute_capability(i) for i in range(num_devices)]
	system_capability = min(device_capability)

	if not all(cc == system_capability for cc in device_capability):
		warnings.warn(
			f"System has multiple GPUs with different compute capabilities: {device_capability}. "
			f"Using compute capability {system_capability} for best compatibility. "
			f"This may result in suboptimal performance."
		)
	return system_capability

system_compute_capability = _get_system_compute_capability()
ALL_COMPUTE_CAPABILITIES.append(system_compute_capability)

# Try to import the fp32 extension
_C = None

for cc in reversed(ALL_COMPUTE_CAPABILITIES):
	if cc > system_compute_capability:
		continue

	try:
		# Key difference: import from tinycudann_inrs_fp32_bindings instead of tinycudann_inrs_bindings
		_C = importlib.import_module(f"tinycudann_inrs_fp32_bindings._{cc}_C")
		if cc != system_compute_capability:
			warnings.warn(f"tinycudann_fp32 was built for lower compute capability ({cc}) than the system's ({system_compute_capability}). Performance may be suboptimal.")
		break
	except ModuleNotFoundError:
		pass

if _C is None:
	raise EnvironmentError(f"Could not find compatible tinycudann_fp32 extension for compute capability {system_compute_capability}.")

def rtc_set_cache_dir(dir):
	if not dir:
		_C.rtc_set_cache_dir('')
		return

	if not os.path.isdir(dir):
		raise OSError(f"Missing RTC cache directory {dir}")

	try:
		with tempfile.TemporaryFile(dir=dir):
			pass
	except OSError as err:
		raise OSError(f"Invalid RTC cache directory {dir}") from err

	_C.rtc_set_cache_dir(str(dir))

# Set up JIT runtime compilation
_rtc_dir = os.path.join(os.path.dirname(__file__), "rtc")

_rtc_include_dir = os.path.join(_rtc_dir, "include")
_C.rtc_set_include_dir(_rtc_include_dir)

_rtc_cache_dir = os.path.join(_rtc_dir, "cache")
try:
	os.makedirs(_rtc_cache_dir, exist_ok=True)
	rtc_set_cache_dir(_rtc_cache_dir)
except OSError:
	pass

def _torch_precision(tcnn_precision):
	if tcnn_precision == _C.Precision.Fp16:
		return torch.half
	elif tcnn_precision == _C.Precision.Fp32:
		return torch.float
	else:
		raise ValueError(f"Unknown precision {tcnn_precision}")

def supports_jit_fusion():
	return _C.supports_jit_fusion()

def free_temporary_memory():
	gc.collect()
	_C.free_temporary_memory()

def null_tensor_like(tensor):
	return torch.empty([], dtype=tensor.dtype, device=tensor.device)

def null_tensor_to_none(tensor):
	if len(tensor.shape) == 0:
		return None
	return tensor

class _module_function(torch.autograd.Function):
	@staticmethod
	def forward(ctx, native_tcnn_module, input, params, loss_scale):
		ctx.set_materialize_grads(False)

		native_ctx, output = native_tcnn_module.fwd(input, params)
		ctx.save_for_backward(input, params, output)
		ctx.native_tcnn_module = native_tcnn_module
		ctx.native_ctx = native_ctx
		ctx.loss_scale = loss_scale

		return output

	@staticmethod
	def backward(ctx, doutput):
		if doutput is None:
			return None, None, None, None

		if not doutput.is_cuda:
			warnings.warn("doutput must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
			doutput = doutput.cuda()

		input, params, output = ctx.saved_tensors
		input_grad, params_grad = _module_function_backward.apply(ctx, doutput, input, params, output)

		return None, null_tensor_to_none(input_grad), null_tensor_to_none(params_grad), None

class _module_function_backward(torch.autograd.Function):
	@staticmethod
	def forward(ctx, ctx_fwd, doutput, input, params, output):
		ctx.ctx_fwd = ctx_fwd
		ctx.save_for_backward(input, params, doutput)
		with torch.no_grad():
			scaled_grad = doutput * ctx_fwd.loss_scale
			input_grad, params_grad = ctx_fwd.native_tcnn_module.bwd(ctx_fwd.native_ctx, input, params, output, scaled_grad)
			input_grad = null_tensor_like(input) if input_grad is None else (input_grad / ctx_fwd.loss_scale)
			params_grad = null_tensor_like(params) if params_grad is None else (params_grad / ctx_fwd.loss_scale)
		return input_grad, params_grad

	@staticmethod
	def backward(ctx, dinput_grad, dparams_grad):
		input, params, doutput = ctx.saved_tensors
		with torch.enable_grad():
			doutput = doutput * ctx.ctx_fwd.loss_scale
		with torch.no_grad():
			doutput_grad, params_grad, input_grad = ctx.ctx_fwd.native_tcnn_module.bwd_bwd_input(
				ctx.ctx_fwd.native_ctx,
				input,
				params,
				dinput_grad,
				doutput
			)
			params_grad = None if params_grad is None else (params_grad / ctx.ctx_fwd.loss_scale)
			input_grad = None if input_grad is None else (input_grad / ctx.ctx_fwd.loss_scale)

		return None, doutput_grad, input_grad, params_grad, None

class Module(torch.nn.Module):
	def __init__(self, seed=1337):
		super(Module, self).__init__()

		self.native_tcnn_module = self._native_tcnn_module()
		self.dtype = _torch_precision(self.native_tcnn_module.param_precision())

		self.seed = seed
		initial_params = self.native_tcnn_module.initial_params(seed)
		self.params = torch.nn.Parameter(initial_params, requires_grad=True)
		self.register_parameter(name="params", param=self.params)

		self.loss_scale = _C.default_loss_scale(self.native_tcnn_module.param_precision())

	def forward(self, x):
		if not x.is_cuda:
			warnings.warn("input must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
			x = x.cuda()

		batch_size = x.shape[0]
		batch_size_granularity = int(_C.batch_size_granularity())
		padded_batch_size = (batch_size + batch_size_granularity-1) // batch_size_granularity * batch_size_granularity

		x_padded = x if batch_size == padded_batch_size else torch.nn.functional.pad(x, [0, 0, 0, padded_batch_size - batch_size])
		output = _module_function.apply(
			self.native_tcnn_module,
			x_padded.to(torch.float).contiguous(),
			self.params.to(_torch_precision(self.native_tcnn_module.param_precision())).contiguous(),
			self.loss_scale
		)
		return output[:batch_size, :self.n_output_dims]

	def __getstate__(self):
		"""Return state values to be pickled."""
		state = self.__dict__.copy()
		del state["native_tcnn_module"]
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		self.native_tcnn_module = self._native_tcnn_module()

	def extra_repr(self):
		return f"n_input_dims={self.n_input_dims}, n_output_dims={self.n_output_dims}, seed={self.seed}, dtype={self.dtype}, hyperparams={self.native_tcnn_module.hyperparams()}"

	@property
	def jit_fusion(self):
		return self.native_tcnn_module.jit_fusion

	@jit_fusion.setter
	def jit_fusion(self, val):
		self.native_tcnn_module.jit_fusion = val

	@jit_fusion.deleter
	def jit_fusion(self):
		raise AttributeError("`jit_fusion` can not be deleted")

class NetworkWithInputEncoding(Module):
	"""
	Input encoding, followed by a neural network (FP32 precision).

	This module is more efficient than invoking individual `Encoding`
	and `Network` modules in sequence.

	Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
	it to a tensor of shape `[:, n_output_dims]`.

	Note: This FP32 version only supports CutlassMLP, not FullyFusedMLP.

	Parameters
	----------
	n_input_dims : `int`
		Determines the shape of input tensors as `[:, n_input_dims]`
	n_output_dims : `int`
		Determines the shape of output tensors as `[:, n_output_dims]`
	encoding_config: `dict`
		Configures the encoding. Possible configurations are documented at
		https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
	network_config: `dict`
		Configures the neural network. Must use "otype": "CutlassMLP" for FP32.
	seed: `int`
		Seed for pseudorandom parameter initialization
	"""
	def __init__(self, n_input_dims, n_output_dims, encoding_config, network_config, seed=1337):
		if not _C.has_networks():
			raise RuntimeError(f"Cannot create `NetworkWithInputEncoding` because tiny-cuda-nn was not compiled with neural network support.")

		self.n_input_dims = n_input_dims
		self.n_output_dims = n_output_dims
		self.encoding_config = encoding_config
		self.network_config = network_config

		super(NetworkWithInputEncoding, self).__init__(seed=seed)

	def _native_tcnn_module(self):
		return _C.create_network_with_input_encoding(self.n_input_dims, self.n_output_dims, self.encoding_config, self.network_config)

class Network(Module):
	"""
	Neural network (FP32 precision).

	Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
	it to a tensor of shape `[:, n_output_dims]`.

	Note: This FP32 version only supports CutlassMLP, not FullyFusedMLP.

	Parameters
	----------
	n_input_dims : `int`
		Determines the shape of input tensors as `[:, n_input_dims]`
	n_output_dims : `int`
		Determines the shape of output tensors as `[:, n_output_dims]`
	network_config: `dict`
		Configures the neural network. Must use "otype": "CutlassMLP" for FP32.
	seed: `int`
		Seed for pseudorandom parameter initialization
	"""
	def __init__(self, n_input_dims, n_output_dims, network_config, seed=1337):
		if not _C.has_networks():
			raise RuntimeError(f"Cannot create `Network` because tiny-cuda-nn was not compiled with neural network support.")

		self.n_input_dims = n_input_dims
		self.n_output_dims = n_output_dims
		self.network_config = network_config

		super(Network, self).__init__(seed=seed)

	def _native_tcnn_module(self):
		return _C.create_network(self.n_input_dims, self.n_output_dims, self.network_config)

class Encoding(Module):
	"""
	Input encoding to a neural network (FP32 precision).

	Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
	it to a `dtype` tensor of shape `[:, self.n_output_dims]`, where
	`self.n_output_dims` depends on `n_input_dims` and the configuration
	`encoding_config`.

	Parameters
	----------
	n_input_dims : `int`
		Determines the shape of input tensors as `[:, n_input_dims]`
	encoding_config: `dict`
		Configures the encoding. Possible configurations are documented at
		https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
	seed: `int`
		Seed for pseudorandom parameter initialization
	dtype: `torch.dtype`
		Precision of the output tensor and internal parameters. A value
		of `None` corresponds to `torch.float` for this FP32 build.
	"""
	def __init__(self, n_input_dims, encoding_config, seed=1337, dtype=None):
		self.n_input_dims = n_input_dims
		self.encoding_config = encoding_config
		if dtype is None:
			self.precision = _C.preferred_precision()
		else:
			if dtype == torch.float32:
				self.precision = _C.Precision.Fp32
			elif dtype == torch.float16:
				self.precision = _C.Precision.Fp16
			else:
				raise ValueError(f"Encoding only supports fp32 or fp16 precision, but got {dtype}")

		super(Encoding, self).__init__(seed=seed)

		self.n_output_dims = self.native_tcnn_module.n_output_dims()

	def _native_tcnn_module(self):
		return _C.create_encoding(self.n_input_dims, self.encoding_config, self.precision)


# =============================================================================
# SIREN Initialization Utilities
# =============================================================================

def _get_network_structure(model: Module) -> dict:
	"""
	Extract network structure from a tcnn Network or NetworkWithInputEncoding.
	"""
	hyperparams = model.native_tcnn_module.hyperparams()

	n_hidden_layers = hyperparams.get('n_hidden_layers', 0)
	network_width = hyperparams.get('n_neurons', 128)
	use_bias = hyperparams.get('use_bias', False)
	activation = hyperparams.get('activation', 'ReLU')

	input_width = model.n_input_dims
	output_width = model.n_output_dims
	padded_output_width = ((output_width + 15) // 16) * 16
	n_hidden_matmuls = max(0, n_hidden_layers - 1)

	return {
		'input_width': input_width,
		'network_width': network_width,
		'output_width': output_width,
		'padded_output_width': padded_output_width,
		'n_hidden_layers': n_hidden_layers,
		'n_hidden_matmuls': n_hidden_matmuls,
		'use_bias': use_bias,
		'activation': activation,
	}


def _get_layer_shapes(structure: dict) -> list:
	"""Get list of (fan_out, fan_in) tuples for each weight matrix."""
	shapes = []

	if structure['n_hidden_layers'] == 0:
		shapes.append((structure['padded_output_width'], structure['input_width']))
	else:
		shapes.append((structure['network_width'], structure['input_width']))
		for _ in range(structure['n_hidden_matmuls']):
			shapes.append((structure['network_width'], structure['network_width']))
		shapes.append((structure['padded_output_width'], structure['network_width']))

	return shapes


def _get_bias_sizes(structure: dict) -> list:
	"""Get list of bias sizes (padded to alignment of 16)."""
	ALIGNMENT = 16
	sizes = []

	if not structure['use_bias']:
		return sizes

	if structure['n_hidden_layers'] == 0:
		size = ((structure['padded_output_width'] + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
		sizes.append(size)
	else:
		hidden_bias_size = ((structure['network_width'] + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
		sizes.append(hidden_bias_size)
		for _ in range(structure['n_hidden_matmuls']):
			sizes.append(hidden_bias_size)
		output_bias_size = ((structure['padded_output_width'] + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
		sizes.append(output_bias_size)

	return sizes


def siren_init(
	model: Module,
	omega_0: float = 30.0,
	first_layer_omega_0: Optional[float] = None,
	bias_init: Literal['zero', 'siren', 'uniform'] = 'zero',
	seed: Optional[int] = None,
) -> None:
	"""
	Re-initialize a tiny-cuda-nn network with SIREN initialization.

	Since tcnn's Sine activation computes sin(x) without omega_0 scaling,
	we absorb omega_0 into the weights and biases.

	Parameters
	----------
	model : Network or NetworkWithInputEncoding
		The network to re-initialize. Should have activation='Sine'.
	omega_0 : float, default=30.0
		The omega_0 factor for hidden layers.
	first_layer_omega_0 : float, optional
		Separate omega_0 for the first layer. If None, uses omega_0.
	bias_init : {'zero', 'siren', 'uniform'}, default='zero'
		Bias initialization strategy.
	seed : int, optional
		Random seed for reproducibility.
	"""
	if first_layer_omega_0 is None:
		first_layer_omega_0 = omega_0

	if seed is not None:
		torch.manual_seed(seed)

	structure = _get_network_structure(model)
	layer_shapes = _get_layer_shapes(structure)
	bias_sizes = _get_bias_sizes(structure)

	if structure['activation'] != 'Sine':
		warnings.warn(
			f"SIREN initialization is designed for Sine activation, "
			f"but network has activation='{structure['activation']}'"
		)

	if bias_init != 'zero' and not structure['use_bias']:
		raise ValueError(
			f"bias_init='{bias_init}' requires use_bias=True, "
			f"but network has use_bias=False"
		)

	params = model.params.data
	total_weight_params = sum(fo * fi for fo, fi in layer_shapes)
	total_bias_params = sum(bias_sizes) if structure['use_bias'] else 0
	expected_total = total_weight_params + total_bias_params

	if params.numel() != expected_total:
		raise ValueError(
			f"Parameter count mismatch: model has {params.numel()} params, "
			f"but computed {expected_total}"
		)

	new_params = torch.empty(params.numel(), dtype=torch.float32)

	offset = 0
	n_layers = len(layer_shapes)

	for i, (fan_out, fan_in) in enumerate(layer_shapes):
		n_elements = fan_out * fan_in
		weight_slice = new_params[offset:offset + n_elements]

		is_first_layer = (i == 0)
		is_output_layer = (i == n_layers - 1)

		if is_first_layer:
			bound = first_layer_omega_0 / fan_in
			weight_slice.uniform_(-bound, bound)
		elif is_output_layer:
			bound = math.sqrt(6.0 / (fan_in + fan_out))
			weight_slice.uniform_(-bound, bound)
		else:
			bound = math.sqrt(6.0 / fan_in)
			weight_slice.uniform_(-bound, bound)

		offset += n_elements

	if structure['use_bias']:
		for i, bias_size in enumerate(bias_sizes):
			bias_slice = new_params[offset:offset + bias_size]
			fan_out, fan_in = layer_shapes[i]
			is_first_layer = (i == 0)
			is_output_layer = (i == len(bias_sizes) - 1)
			layer_omega = first_layer_omega_0 if is_first_layer else omega_0

			if bias_init == 'zero':
				bias_slice.zero_()
			elif bias_init == 'siren':
				if is_output_layer:
					bound = 1.0 / math.sqrt(fan_in)
				else:
					bound = layer_omega / math.sqrt(fan_in)
				bias_slice.uniform_(-bound, bound)
			elif bias_init == 'uniform':
				if is_output_layer:
					bound = 1.0 / fan_in
				else:
					bound = layer_omega / fan_in
				bias_slice.uniform_(-bound, bound)

			offset += bias_size

	with torch.no_grad():
		params.copy_(new_params.to(params.device).to(params.dtype))


def siren_init_first_layer(
	model: Module,
	omega_0: float = 30.0,
	bias_init: Literal['zero', 'siren', 'uniform'] = 'zero',
	seed: Optional[int] = None,
) -> None:
	"""Re-initialize only the first layer with SIREN initialization."""
	if seed is not None:
		torch.manual_seed(seed)

	structure = _get_network_structure(model)
	layer_shapes = _get_layer_shapes(structure)

	fan_out, fan_in = layer_shapes[0]
	n_elements = fan_out * fan_in

	bound = omega_0 / fan_in
	new_weights = torch.empty(n_elements, dtype=torch.float32)
	new_weights.uniform_(-bound, bound)

	params = model.params.data
	with torch.no_grad():
		params[:n_elements].copy_(new_weights.to(params.device).to(params.dtype))

	if structure['use_bias'] and bias_init != 'zero':
		total_weight_params = sum(fo * fi for fo, fi in layer_shapes)
		bias_sizes = _get_bias_sizes(structure)

		bias_offset = total_weight_params
		bias_size = bias_sizes[0]

		new_bias = torch.empty(bias_size, dtype=torch.float32)

		if bias_init == 'siren':
			bias_bound = omega_0 / math.sqrt(fan_in)
		elif bias_init == 'uniform':
			bias_bound = omega_0 / fan_in
		else:
			bias_bound = 0.0

		if bias_bound > 0:
			new_bias.uniform_(-bias_bound, bias_bound)
		else:
			new_bias.zero_()

		with torch.no_grad():
			params[bias_offset:bias_offset + bias_size].copy_(
				new_bias.to(params.device).to(params.dtype)
			)


def inspect_network_params(model: Module, verbose: bool = True) -> dict:
	"""Inspect the parameter layout of a tcnn network."""
	structure = _get_network_structure(model)
	layer_shapes = _get_layer_shapes(structure)
	bias_sizes = _get_bias_sizes(structure)

	info = {
		'structure': structure,
		'layer_shapes': layer_shapes,
		'bias_sizes': bias_sizes,
		'total_params': model.params.numel(),
	}

	if verbose:
		print("Network Structure (FP32):")
		print(f"  Input width: {structure['input_width']}")
		print(f"  Network width: {structure['network_width']}")
		print(f"  Output width: {structure['output_width']} "
			  f"(padded: {structure['padded_output_width']})")
		print(f"  Hidden layers: {structure['n_hidden_layers']}")
		print(f"  Use bias: {structure['use_bias']}")
		print(f"  Activation: {structure['activation']}")
		print()
		print("Weight Matrices (Row-Major, shape = (fan_out, fan_in)):")
		offset = 0
		for i, (fo, fi) in enumerate(layer_shapes):
			n_elem = fo * fi
			if i == 0:
				layer_type = "Input"
			elif i == len(layer_shapes) - 1:
				layer_type = "Output"
			else:
				layer_type = f"Hidden {i}"
			print(f"  Layer {i} ({layer_type}): ({fo}, {fi}) = {n_elem} params, "
				  f"offset {offset}")
			offset += n_elem

		if structure['use_bias']:
			print()
			print("Bias Vectors (padded to 16):")
			for i, size in enumerate(bias_sizes):
				if i == 0:
					layer_type = "Input"
				elif i == len(bias_sizes) - 1:
					layer_type = "Output"
				else:
					layer_type = f"Hidden {i}"
				print(f"  Layer {i} ({layer_type}): {size} params, offset {offset}")
				offset += size

		print()
		print(f"Total parameters: {model.params.numel()}")

	return info


def get_weight_matrix(model: Module, layer_idx: int) -> torch.Tensor:
	"""Extract a weight matrix from a tcnn network as a 2D tensor."""
	structure = _get_network_structure(model)
	layer_shapes = _get_layer_shapes(structure)

	if layer_idx < 0:
		layer_idx = len(layer_shapes) + layer_idx

	if layer_idx < 0 or layer_idx >= len(layer_shapes):
		raise IndexError(
			f"layer_idx {layer_idx} out of range for network with "
			f"{len(layer_shapes)} layers"
		)

	offset = sum(fo * fi for fo, fi in layer_shapes[:layer_idx])
	fan_out, fan_in = layer_shapes[layer_idx]
	n_elements = fan_out * fan_in

	return model.params.data[offset:offset + n_elements].view(fan_out, fan_in)


def get_bias_vector(model: Module, layer_idx: int) -> torch.Tensor:
	"""Extract a bias vector from a tcnn network."""
	structure = _get_network_structure(model)

	if not structure['use_bias']:
		raise ValueError("Network does not have biases enabled")

	layer_shapes = _get_layer_shapes(structure)
	bias_sizes = _get_bias_sizes(structure)

	if layer_idx < 0:
		layer_idx = len(bias_sizes) + layer_idx

	if layer_idx < 0 or layer_idx >= len(bias_sizes):
		raise IndexError(
			f"layer_idx {layer_idx} out of range for network with "
			f"{len(bias_sizes)} bias vectors"
		)

	total_weight_params = sum(fo * fi for fo, fi in layer_shapes)
	bias_offset = total_weight_params + sum(bias_sizes[:layer_idx])
	bias_size = bias_sizes[layer_idx]

	return model.params.data[bias_offset:bias_offset + bias_size]


# =============================================================================
# Muon Optimizer Utilities
# =============================================================================

def get_weight_matrices(model: Module, requires_grad: bool = True) -> list:
	"""
	Get all weight matrices as 2D tensor views suitable for optimizers.

	Unlike `get_weight_matrix()` which returns a view into `.data`, this
	function returns views that preserve gradient tracking, making them
	suitable for use with optimizers like torch.optim.Muon.

	Parameters
	----------
	model : Network or NetworkWithInputEncoding
		The tiny-cuda-nn network model.

	requires_grad : bool, default=True
		If True, returns views of `model.params` (preserves gradients).
		If False, returns views of `model.params.data` (no gradients).

	Returns
	-------
	list of torch.Tensor
		List of 2D weight matrices, one per layer.
	"""
	structure = _get_network_structure(model)
	layer_shapes = _get_layer_shapes(structure)

	params = model.params if requires_grad else model.params.data

	weights = []
	offset = 0
	for fan_out, fan_in in layer_shapes:
		n_elements = fan_out * fan_in
		weight = params[offset:offset + n_elements].view(fan_out, fan_in)
		weights.append(weight)
		offset += n_elements

	return weights


def get_bias_vectors(model: Module, requires_grad: bool = True) -> list:
	"""
	Get all bias vectors as 1D tensor views suitable for optimizers.

	Parameters
	----------
	model : Network or NetworkWithInputEncoding
		The tiny-cuda-nn network model.

	requires_grad : bool, default=True
		If True, returns views of `model.params` (preserves gradients).
		If False, returns views of `model.params.data` (no gradients).

	Returns
	-------
	list of torch.Tensor
		List of 1D bias vectors, one per layer.
		Empty list if the network has use_bias=False.
	"""
	structure = _get_network_structure(model)

	if not structure['use_bias']:
		return []

	layer_shapes = _get_layer_shapes(structure)
	bias_sizes = _get_bias_sizes(structure)

	params = model.params if requires_grad else model.params.data

	total_weight_params = sum(fo * fi for fo, fi in layer_shapes)

	biases = []
	offset = total_weight_params
	for bias_size in bias_sizes:
		bias = params[offset:offset + bias_size]
		biases.append(bias)
		offset += bias_size

	return biases


def get_muon_param_groups(
	model: Module,
	lr: float = 0.02,
	momentum: float = 0.95,
	weight_decay: float = 0.0,
	adamw_lr: Optional[float] = None,
	adamw_weight_decay: float = 0.0,
) -> list:
	"""
	Get parameter groups configured for the Muon optimizer.

	Muon is designed for 2D weight matrices. Biases and other 1D parameters
	should use AdamW (via Muon's built-in adamw_params support) or be placed
	in a separate param group with use_muon=False.

	Parameters
	----------
	model : Network or NetworkWithInputEncoding
		The tiny-cuda-nn network model.
	lr : float, default=0.02
		Learning rate for Muon (applied to weight matrices).
	momentum : float, default=0.95
		Momentum for Muon.
	weight_decay : float, default=0.0
		Weight decay for weight matrices.
	adamw_lr : float, optional
		Learning rate for biases (via AdamW). If None, uses lr * 0.1.
	adamw_weight_decay : float, default=0.0
		Weight decay for biases.

	Returns
	-------
	list of dict
		Parameter groups ready for torch.optim.Muon.
	"""
	weights = get_weight_matrices(model, requires_grad=True)
	biases = get_bias_vectors(model, requires_grad=True)

	if adamw_lr is None:
		adamw_lr = lr * 0.1

	param_groups = [
		{
			'params': weights,
			'lr': lr,
			'momentum': momentum,
			'weight_decay': weight_decay,
			'use_muon': True,
		},
	]

	if biases:
		param_groups.append({
			'params': biases,
			'lr': adamw_lr,
			'weight_decay': adamw_weight_decay,
			'use_muon': False,
		})

	return param_groups


def create_muon_optimizer(
	model: Module,
	lr: float = 0.02,
	momentum: float = 0.95,
	nesterov: bool = True,
	ns_steps: int = 5,
	weight_decay: float = 0.0,
	adamw_lr: Optional[float] = None,
	adamw_betas: tuple = (0.9, 0.95),
	adamw_wd: float = 0.0,
):
	"""
	Create a Muon optimizer configured for a tcnn network.

	This is a convenience function that sets up the Muon optimizer with
	appropriate parameter groups for weight matrices (Muon) and biases (AdamW).

	Parameters
	----------
	model : Network or NetworkWithInputEncoding
		The tiny-cuda-nn network model.
	lr : float, default=0.02
		Learning rate for Muon (weight matrices).
	momentum : float, default=0.95
		Momentum for Muon.
	nesterov : bool, default=True
		Whether to use Nesterov momentum.
	ns_steps : int, default=5
		Number of Newton-Schulz iteration steps.
	weight_decay : float, default=0.0
		Weight decay for weight matrices.
	adamw_lr : float, optional
		Learning rate for biases. If None, uses lr * 0.1.
	adamw_betas : tuple, default=(0.9, 0.95)
		Betas for AdamW (used for biases).
	adamw_wd : float, default=0.0
		Weight decay for AdamW (biases).

	Returns
	-------
	torch.optim.Muon
		Configured Muon optimizer.

	Notes
	-----
	Requires PyTorch >= 2.6 with Muon optimizer support.
	"""
	if not hasattr(torch.optim, 'Muon'):
		raise ImportError(
			"torch.optim.Muon is not available. "
			"Please upgrade to PyTorch >= 2.6 or install a version with Muon support."
		)

	weights = get_weight_matrices(model, requires_grad=True)
	biases = get_bias_vectors(model, requires_grad=True)

	if adamw_lr is None:
		adamw_lr = lr * 0.1

	adamw_params = biases if biases else None

	optimizer = torch.optim.Muon(
		weights,
		lr=lr,
		momentum=momentum,
		nesterov=nesterov,
		ns_steps=ns_steps,
		weight_decay=weight_decay,
		adamw_params=adamw_params,
		adamw_lr=adamw_lr,
		adamw_betas=adamw_betas,
		adamw_wd=adamw_wd,
	)

	return optimizer
