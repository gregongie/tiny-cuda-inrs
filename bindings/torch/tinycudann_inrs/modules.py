# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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

# Determine the capability of the system as the minimum of all
# devices, ensuring that we have no runtime errors.
system_compute_capability = _get_system_compute_capability()

# Ensure the system's compute capability is represented in the list to avoid
# total failure if a new capability is released without tiny-cuda-nn being updated.
ALL_COMPUTE_CAPABILITIES.append(system_compute_capability)

# Try to import the highest compute capability version of tcnn that
# we can find and is compatible with the system's compute capability.
_C = None

for cc in reversed(ALL_COMPUTE_CAPABILITIES):
	if cc > system_compute_capability:
		# incompatible
		continue

	try:
		_C = importlib.import_module(f"tinycudann_inrs_bindings._{cc}_C")
		if cc != system_compute_capability:
			warnings.warn(f"tinycudann was built for lower compute capability ({cc}) than the system's ({system_compute_capability}). Performance may be suboptimal.")
		break
	except ModuleNotFoundError:
		pass

if _C is None:
	raise EnvironmentError(f"Could not find compatible tinycudann extension for compute capability {system_compute_capability}.")

# Pipe tcnn warnings and errors into Python
# def _log(severity, msg):
# 	if severity == _C.LogSeverity.Warning:
# 		warnings.warn(f"tinycudann warning: {msg}")
# 	elif severity == _C.LogSeverity.Error:
# 		warnings.warn(f"tinycudann error: {msg}")

# _C.set_log_callback(_log)

def rtc_set_cache_dir(dir):
	if not dir:
		_C.rtc_set_cache_dir('')
		return

	if not os.path.isdir(dir):
		raise OSError(f"Missing RTC cache directory {dir}")

	# check write permission
	try:
		with tempfile.TemporaryFile(dir=dir):
			pass
	except OSError as err:
		raise OSError(f"Invalid RTC cache directory {dir}") from err

	# `str` to handle pathlib.Path objects
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
	# Ensure all Python objects (potentially pointing
	# to temporary TCNN allocations) are cleaned up.
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
		# If no output gradient is provided, no need to
		# automatically materialize it as torch.zeros.
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
		# NOTE: currently support:
		#       ✓   d(dL_dinput)_d(dL_doutput)  doutput_grad
		#       ✓   d(dL_dinput)_d(params)      params_grad
		#       ✓   d(dL_dinput)_d(input)       input_grad
		#       x   d(dL_dparam)_d(...)
		input, params, doutput = ctx.saved_tensors
		# assert dparams_grad is None, "currently do not support 2nd-order gradients from gradient of grid"
		with torch.enable_grad():
			# NOTE: preserves requires_grad info (this function is in no_grad() context by default when invoking loss.backward())
			doutput = doutput * ctx.ctx_fwd.loss_scale
		with torch.no_grad():
			doutput_grad, params_grad, input_grad = ctx.ctx_fwd.native_tcnn_module.bwd_bwd_input(
				ctx.ctx_fwd.native_ctx,
				input,
				params,
				dinput_grad,
				doutput
			)
			# NOTE: be cautious when multiplying and dividing loss_scale
			#       doutput_grad uses dinput_grad
			#       params_grad  uses dinput_grad * doutput
			#       input_grad   uses dinput_grad * doutput
			params_grad = None if params_grad is None else (params_grad / ctx.ctx_fwd.loss_scale)
			input_grad = None if input_grad is None else (input_grad / ctx.ctx_fwd.loss_scale)

		# ctx_fwd,   doutput,      input,      params,      output
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
		# Avoid pickling native objects
		del state["native_tcnn_module"]
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		# Reconstruct native entries
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
	Input encoding, followed by a neural network.

	This module is more efficient than invoking individual `Encoding`
	and `Network` modules in sequence.

	Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
	it to a tensor of shape `[:, n_output_dims]`.

	The output tensor can be either of type `torch.float` or `torch.half`,
	depending on which performs better on the system.

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
		Configures the neural network. Possible configurations are documented at
		https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
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
	Neural network.

	Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
	it to a tensor of shape `[:, n_output_dims]`.

	The output tensor can be either of type `torch.float` or `torch.half`,
	depending on which performs better on the system.

	Parameters
	----------
	n_input_dims : `int`
		Determines the shape of input tensors as `[:, n_input_dims]`
	n_output_dims : `int`
		Determines the shape of output tensors as `[:, n_output_dims]`
	network_config: `dict`
		Configures the neural network. Possible configurations are documented at
		https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
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
	Input encoding to a neural network.

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
		of `None` corresponds to the optimally performing precision,
		which is `torch.half` on most systems. A value of `torch.float`
		may yield higher numerical accuracy, but is generally slower.
		A value of `torch.half` may not be supported on all systems.
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

def _get_encoding_info(hyperparams: dict, model_n_input_dims: int) -> dict:
	"""
	Extract encoding information from hyperparams.

	Parameters
	----------
	hyperparams : dict
		The full hyperparams from model.native_tcnn_module.hyperparams().
	model_n_input_dims : int
		The model's n_input_dims (encoding input dimensions).

	Returns
	-------
	dict or None
		If model has an encoding, returns dict with:
		- n_output_dims: int (unpadded encoding output)
		- padded_output_width: int (padded encoding output, used as network input)
		- n_params: int (number of encoding parameters)
		Returns None if no encoding present.
	"""
	if 'encoding' not in hyperparams:
		return None

	encoding_config = hyperparams['encoding']
	otype = encoding_config.get('otype', '')

	# Compute encoding output dimensions and params based on encoding type
	# Note: Most encodings have n_params=0 (frequencies/features are not trainable)
	# Only grid-based encodings (HashGrid, etc.) have trainable parameters
	if otype == 'RandomFourierFeatures':
		n_features = encoding_config.get('n_features', 128)
		n_output_dims = n_features * 2
		n_params = 0  # Frequencies are fixed, not trainable

	elif otype == 'Frequency':
		n_frequencies = encoding_config.get('n_frequencies', 12)
		n_output_dims = model_n_input_dims * n_frequencies * 2
		n_params = 0  # Frequency encoding has no learnable params

	elif otype == 'Identity':
		n_output_dims = model_n_input_dims
		n_params = 0

	elif otype == 'SphericalHarmonics':
		degree = encoding_config.get('degree', 4)
		n_output_dims = degree * degree
		n_params = 0

	elif otype == 'OneBlob':
		n_bins = encoding_config.get('n_bins', 16)
		n_output_dims = model_n_input_dims * n_bins
		n_params = 0

	elif otype in ('HashGrid', 'Grid', 'DenseGrid', 'TiledGrid'):
		# Grid encodings are complex - use a rough estimate
		# These have learnable parameters stored in the grid
		n_levels = encoding_config.get('n_levels', 16)
		n_features_per_level = encoding_config.get('n_features_per_level', 2)
		n_output_dims = n_levels * n_features_per_level
		# Grid params are complex to compute exactly; we'll handle this case specially
		# For now, set to -1 to indicate we need to compute it differently
		n_params = -1

	else:
		# Unknown encoding type - try to infer from common patterns
		n_output_dims = encoding_config.get('n_output_dims', model_n_input_dims)
		n_params = 0

	# Padded output width (aligned to 16)
	padded_output_width = ((n_output_dims + 15) // 16) * 16

	return {
		'n_output_dims': n_output_dims,
		'padded_output_width': padded_output_width,
		'n_params': n_params,
		'otype': otype,
	}


def _get_network_structure(model: Module) -> dict:
	"""
	Extract network structure from a tcnn Network or NetworkWithInputEncoding.

	Parameters
	----------
	model : Network or NetworkWithInputEncoding
		The tiny-cuda-nn network model.

	Returns
	-------
	dict
		Dictionary with keys:
		- input_width: int (network input width, which is encoding output for NetworkWithInputEncoding)
		- padded_input_width: int
		- network_width: int
		- output_width: int
		- padded_output_width: int
		- n_hidden_layers: int
		- n_hidden_matmuls: int
		- use_bias: bool
		- activation: str
		- encoding_info: dict or None (encoding details if present)
	"""
	hyperparams = model.native_tcnn_module.hyperparams()

	# Network config is nested under 'network' key for both Network and NetworkWithInputEncoding
	network_config = hyperparams.get('network', hyperparams)

	n_hidden_layers = network_config.get('n_hidden_layers', 0)
	network_width = network_config.get('n_neurons', 128)
	use_bias = network_config.get('use_bias', False)
	activation = network_config.get('activation', 'ReLU')

	# Check for encoding (NetworkWithInputEncoding)
	encoding_info = _get_encoding_info(hyperparams, model.n_input_dims)

	if encoding_info is not None:
		# For NetworkWithInputEncoding, the network's input is the encoding's output
		input_width = encoding_info['n_output_dims']
		padded_input_width = encoding_info['padded_output_width']
	else:
		# For standalone Network, use model's input dims
		input_width = model.n_input_dims
		padded_input_width = ((input_width + 15) // 16) * 16

	output_width = model.n_output_dims
	padded_output_width = ((output_width + 15) // 16) * 16

	# Number of hidden matmuls
	n_hidden_matmuls = max(0, n_hidden_layers - 1)

	return {
		'input_width': input_width,
		'padded_input_width': padded_input_width,
		'network_width': network_width,
		'output_width': output_width,
		'padded_output_width': padded_output_width,
		'n_hidden_layers': n_hidden_layers,
		'n_hidden_matmuls': n_hidden_matmuls,
		'use_bias': use_bias,
		'activation': activation,
		'encoding_info': encoding_info,
	}


def _get_layer_shapes(structure: dict) -> list:
	"""
	Get list of (fan_out, fan_in) tuples for each weight matrix.

	Parameters
	----------
	structure : dict
		Network structure from _get_network_structure().

	Returns
	-------
	list of tuple
		List of (fan_out, fan_in) shapes for each weight matrix.
	"""
	shapes = []

	if structure['n_hidden_layers'] == 0:
		# Single layer network
		shapes.append((structure['padded_output_width'], structure['padded_input_width']))
	else:
		# Input layer (uses padded input width for CutlassMLP alignment)
		shapes.append((structure['network_width'], structure['padded_input_width']))

		# Hidden layers
		for _ in range(structure['n_hidden_matmuls']):
			shapes.append((structure['network_width'], structure['network_width']))

		# Output layer
		shapes.append((structure['padded_output_width'], structure['network_width']))

	return shapes


def _get_bias_sizes(structure: dict) -> list:
	"""
	Get list of bias sizes (padded to alignment of 16).

	Parameters
	----------
	structure : dict
		Network structure from _get_network_structure().

	Returns
	-------
	list of int
		List of bias vector sizes for each layer.
	"""
	ALIGNMENT = 16
	sizes = []

	if not structure['use_bias']:
		return sizes

	if structure['n_hidden_layers'] == 0:
		# Single layer
		size = ((structure['padded_output_width'] + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
		sizes.append(size)
	else:
		# Input layer bias
		hidden_bias_size = ((structure['network_width'] + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
		sizes.append(hidden_bias_size)

		# Hidden layer biases
		for _ in range(structure['n_hidden_matmuls']):
			sizes.append(hidden_bias_size)

		# Output layer bias
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
	we absorb omega_0 into the weights and biases:
		sin(omega_0 * (Wx + b)) = sin(W'x + b')
	where W' = omega_0 * W and b' = omega_0 * b

	Parameters
	----------
	model : Network or NetworkWithInputEncoding
		The network to re-initialize. Should have activation='Sine'.

	omega_0 : float, default=30.0
		The omega_0 factor for hidden layers. Standard SIREN uses 30.0.

	first_layer_omega_0 : float, optional
		Separate omega_0 for the first layer. If None, uses omega_0.
		Standard SIREN uses the same value for all layers.

	bias_init : {'zero', 'siren', 'uniform'}, default='zero'
		Bias initialization strategy:
		- 'zero': Initialize biases to zero (tcnn default)
		- 'siren': Initialize as omega_0 * U[-1/sqrt(fan_in), 1/sqrt(fan_in)]
		- 'uniform': Initialize as U[-1/fan_in, 1/fan_in] (simpler scheme)

	seed : int, optional
		Random seed for reproducibility. If None, uses current RNG state.

	Notes
	-----
	Weight initialization (matching SIREN paper):

	- First layer: W ~ U[-1/fan_in, 1/fan_in], then scaled by omega_0
	  Result: W' ~ U[-omega_0/fan_in, omega_0/fan_in]

	- Hidden layers: W ~ U[-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0],
	  then scaled by omega_0
	  Result: W' ~ U[-sqrt(6/fan_in), sqrt(6/fan_in)]

	- Output layer: Xavier uniform (no omega_0 scaling, since output often
	  has no activation or a different activation)

	Examples
	--------
	>>> import tinycudann_inrs as tcnn
	>>> model = tcnn.Network(
	...     n_input_dims=2,
	...     n_output_dims=3,
	...     network_config={
	...         "otype": "CutlassMLP",
	...         "activation": "Sine",
	...         "output_activation": "None",
	...         "n_neurons": 256,
	...         "n_hidden_layers": 5,
	...         "use_bias": True,
	...     }
	... )
	>>> tcnn.siren_init(model, omega_0=30.0, bias_init='siren')
	"""
	if first_layer_omega_0 is None:
		first_layer_omega_0 = omega_0

	if seed is not None:
		torch.manual_seed(seed)

	structure = _get_network_structure(model)
	layer_shapes = _get_layer_shapes(structure)
	bias_sizes = _get_bias_sizes(structure)

	# Validate
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

	# Get the flat parameter tensor
	params = model.params.data

	# Compute network params (weights + biases)
	total_weight_params = sum(fo * fi for fo, fi in layer_shapes)
	total_bias_params = sum(bias_sizes) if structure['use_bias'] else 0
	total_network_params = total_weight_params + total_bias_params

	# Handle encoding parameters for NetworkWithInputEncoding
	# In tcnn, parameter layout is: [network_params][encoding_params]
	encoding_info = structure.get('encoding_info')
	if encoding_info is not None:
		encoding_n_params = encoding_info['n_params']
		if encoding_n_params == -1:
			# For complex encodings (e.g., HashGrid), compute by subtraction
			encoding_n_params = params.numel() - total_network_params
		expected_total = total_network_params + encoding_n_params
	else:
		encoding_n_params = 0
		expected_total = total_network_params

	if params.numel() != expected_total:
		raise ValueError(
			f"Parameter count mismatch: model has {params.numel()} params, "
			f"but computed {expected_total} "
			f"(network weights: {total_weight_params}, biases: {total_bias_params}, "
			f"encoding: {encoding_n_params})"
		)

	# Initialize network params on CPU then copy to GPU
	# Only initialize network portion; leave encoding params unchanged
	new_network_params = torch.empty(total_network_params, dtype=torch.float32)

	offset = 0
	n_layers = len(layer_shapes)

	# Initialize weight matrices
	for i, (fan_out, fan_in) in enumerate(layer_shapes):
		n_elements = fan_out * fan_in
		weight_slice = new_network_params[offset:offset + n_elements]

		is_first_layer = (i == 0)
		is_output_layer = (i == n_layers - 1)

		# Use logical (unpadded) dimensions for initialization bounds
		# since SIREN formulas are defined in terms of actual dimensions
		if is_first_layer:
			logical_fan_in = structure['input_width']
		else:
			logical_fan_in = fan_in  # Hidden layers use network_width (typically already aligned)

		if is_output_layer:
			logical_fan_out = structure['output_width']
		else:
			logical_fan_out = fan_out

		if is_first_layer:
			# First layer: U[-omega_0/fan_in, omega_0/fan_in]
			# This comes from: omega_0 * U[-1/fan_in, 1/fan_in]
			bound = first_layer_omega_0 / logical_fan_in
			weight_slice.uniform_(-bound, bound)

		elif is_output_layer:
			# Output layer: U[-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0]
			# Reference SIREN divides by omega_0 even for linear output layer
			# to keep output magnitudes in a reasonable range
			bound = math.sqrt(6.0 / logical_fan_in) / omega_0
			weight_slice.uniform_(-bound, bound)

		else:
			# Hidden layers: U[-sqrt(6/fan_in), sqrt(6/fan_in)]
			# This comes from: omega_0 * U[-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0]
			bound = math.sqrt(6.0 / logical_fan_in)
			weight_slice.uniform_(-bound, bound)

		offset += n_elements

	# Initialize bias vectors
	if structure['use_bias']:
		for i, bias_size in enumerate(bias_sizes):
			bias_slice = new_network_params[offset:offset + bias_size]

			# Get the corresponding weight matrix fan_in
			fan_out, fan_in = layer_shapes[i]
			is_first_layer = (i == 0)
			is_output_layer = (i == len(bias_sizes) - 1)

			# Use logical (unpadded) fan_in for initialization bounds
			if is_first_layer:
				logical_fan_in = structure['input_width']
			else:
				logical_fan_in = fan_in  # Hidden layers use network_width

			# Determine omega for this layer
			layer_omega = first_layer_omega_0 if is_first_layer else omega_0

			if bias_init == 'zero':
				bias_slice.zero_()

			elif bias_init == 'siren':
				# SIREN-style: omega_0 * U[-1/sqrt(fan_in), 1/sqrt(fan_in)]
				if is_output_layer:
					# Output layer: no omega scaling
					bound = 1.0 / math.sqrt(logical_fan_in)
				else:
					bound = layer_omega / math.sqrt(logical_fan_in)
				bias_slice.uniform_(-bound, bound)

			elif bias_init == 'uniform':
				# Simple uniform: U[-1/fan_in, 1/fan_in]
				if is_output_layer:
					bound = 1.0 / logical_fan_in
				else:
					bound = layer_omega / logical_fan_in
				bias_slice.uniform_(-bound, bound)

			offset += bias_size

	# Copy to model parameters (only network portion; encoding params unchanged)
	with torch.no_grad():
		params[:total_network_params].copy_(
			new_network_params.to(params.device).to(params.dtype)
		)


def siren_init_first_layer(
	model: Module,
	omega_0: float = 30.0,
	bias_init: Literal['zero', 'siren', 'uniform'] = 'zero',
	seed: Optional[int] = None,
) -> None:
	"""
	Re-initialize only the first layer of a tcnn network with SIREN initialization.

	This is useful when you want to use a different omega_0 than the default 30.0
	that tcnn uses, without re-initializing hidden layers.

	Parameters
	----------
	model : Network or NetworkWithInputEncoding
		The network to partially re-initialize.

	omega_0 : float, default=30.0
		The omega_0 factor for the first layer.

	bias_init : {'zero', 'siren', 'uniform'}, default='zero'
		Bias initialization strategy for the first layer bias.

	seed : int, optional
		Random seed for reproducibility.
	"""
	if seed is not None:
		torch.manual_seed(seed)

	structure = _get_network_structure(model)
	layer_shapes = _get_layer_shapes(structure)

	# Get first layer shape (padded) and logical input width
	fan_out, fan_in = layer_shapes[0]
	logical_fan_in = structure['input_width']
	n_elements = fan_out * fan_in

	# Initialize first layer weights using logical fan_in for bounds
	bound = omega_0 / logical_fan_in
	new_weights = torch.empty(n_elements, dtype=torch.float32)
	new_weights.uniform_(-bound, bound)

	# Copy to model
	params = model.params.data
	with torch.no_grad():
		params[:n_elements].copy_(new_weights.to(params.device).to(params.dtype))

	# Also re-initialize first layer bias if present and requested
	if structure['use_bias'] and bias_init != 'zero':
		total_weight_params = sum(fo * fi for fo, fi in layer_shapes)
		bias_sizes = _get_bias_sizes(structure)

		bias_offset = total_weight_params
		bias_size = bias_sizes[0]

		new_bias = torch.empty(bias_size, dtype=torch.float32)

		# Use logical fan_in for bias initialization bounds
		if bias_init == 'siren':
			bias_bound = omega_0 / math.sqrt(logical_fan_in)
		elif bias_init == 'uniform':
			bias_bound = omega_0 / logical_fan_in
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


def pytorch_init(
	model: Module,
	seed: Optional[int] = None,
) -> None:
	"""
	Re-initialize a tiny-cuda-nn network with PyTorch's standard MLP initialization.

	This matches the default initialization used by torch.nn.Linear:
	- Weights: Kaiming uniform with a=sqrt(5), which gives U[-1/sqrt(fan_in), 1/sqrt(fan_in)]
	- Biases: U[-1/sqrt(fan_in), 1/sqrt(fan_in)]

	Parameters
	----------
	model : Network or NetworkWithInputEncoding
		The network to re-initialize.

	seed : int, optional
		Random seed for reproducibility. If None, uses current RNG state.

	Notes
	-----
	This function uses the logical (unpadded) input dimensions for computing
	initialization bounds, since the padded dimensions are implementation details
	of CutlassMLP and should not affect the effective initialization scale.

	For the first layer, this uses the actual input_width (not padded_input_width).
	For hidden and output layers, fan_in is the network_width.

	Examples
	--------
	>>> import tinycudann_inrs as tcnn
	>>> model = tcnn.Network(
	...     n_input_dims=3,
	...     n_output_dims=1,
	...     network_config={
	...         "otype": "CutlassMLP",
	...         "activation": "ReLU",
	...         "output_activation": "None",
	...         "n_neurons": 64,
	...         "n_hidden_layers": 2,
	...         "use_bias": True,
	...     }
	... )
	>>> tcnn.pytorch_init(model, seed=42)
	"""
	if seed is not None:
		torch.manual_seed(seed)

	structure = _get_network_structure(model)
	layer_shapes = _get_layer_shapes(structure)
	bias_sizes = _get_bias_sizes(structure)

	# Get the flat parameter tensor
	params = model.params.data

	# Compute network params (weights + biases)
	total_weight_params = sum(fo * fi for fo, fi in layer_shapes)
	total_bias_params = sum(bias_sizes) if structure['use_bias'] else 0
	total_network_params = total_weight_params + total_bias_params

	# Handle encoding parameters for NetworkWithInputEncoding
	# In tcnn, parameter layout is: [network_params][encoding_params]
	encoding_info = structure.get('encoding_info')
	if encoding_info is not None:
		encoding_n_params = encoding_info['n_params']
		if encoding_n_params == -1:
			# For complex encodings (e.g., HashGrid), compute by subtraction
			encoding_n_params = params.numel() - total_network_params
		expected_total = total_network_params + encoding_n_params
	else:
		encoding_n_params = 0
		expected_total = total_network_params

	if params.numel() != expected_total:
		raise ValueError(
			f"Parameter count mismatch: model has {params.numel()} params, "
			f"but computed {expected_total} "
			f"(network weights: {total_weight_params}, biases: {total_bias_params}, "
			f"encoding: {encoding_n_params})"
		)

	# Initialize network params on CPU then copy to GPU
	# Only initialize network portion; leave encoding params unchanged
	new_network_params = torch.empty(total_network_params, dtype=torch.float32)

	offset = 0

	# Initialize weight matrices with Kaiming uniform (a=sqrt(5))
	# This gives bound = 1/sqrt(fan_in)
	for i, (fan_out, fan_in) in enumerate(layer_shapes):
		n_elements = fan_out * fan_in
		weight_slice = new_network_params[offset:offset + n_elements]

		is_first_layer = (i == 0)

		# Use logical (unpadded) fan_in for initialization bounds
		if is_first_layer:
			logical_fan_in = structure['input_width']
		else:
			logical_fan_in = fan_in  # Hidden layers use network_width

		# Kaiming uniform with a=sqrt(5): bound = 1/sqrt(fan_in)
		bound = 1.0 / math.sqrt(logical_fan_in)
		weight_slice.uniform_(-bound, bound)

		offset += n_elements

	# Initialize bias vectors with U[-1/sqrt(fan_in), 1/sqrt(fan_in)]
	# This matches PyTorch's nn.Linear bias initialization
	if structure['use_bias']:
		for i, bias_size in enumerate(bias_sizes):
			bias_slice = new_network_params[offset:offset + bias_size]

			# Get the corresponding weight matrix fan_in
			fan_out, fan_in = layer_shapes[i]
			is_first_layer = (i == 0)

			# Use logical (unpadded) fan_in for initialization bounds
			if is_first_layer:
				logical_fan_in = structure['input_width']
			else:
				logical_fan_in = fan_in  # Hidden layers use network_width

			# Same bound as weights: 1/sqrt(fan_in)
			bound = 1.0 / math.sqrt(logical_fan_in)
			bias_slice.uniform_(-bound, bound)

			offset += bias_size

	# Copy to model parameters (only network portion; encoding params unchanged)
	with torch.no_grad():
		params[:total_network_params].copy_(
			new_network_params.to(params.device).to(params.dtype)
		)


def inspect_network_params(model: Module, verbose: bool = True) -> dict:
	"""
	Inspect the parameter layout of a tcnn network.

	Parameters
	----------
	model : Network or NetworkWithInputEncoding
		The tiny-cuda-nn network model.

	verbose : bool, default=True
		If True, print a summary of the parameter layout.

	Returns
	-------
	dict
		Dictionary with keys:
		- structure: Network structure dict
		- layer_shapes: List of (fan_out, fan_in) tuples
		- bias_sizes: List of bias sizes
		- total_params: Total number of parameters
	"""
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
		print("Network Structure:")
		print(f"  Input width: {structure['input_width']} "
			  f"(padded: {structure['padded_input_width']})")
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


def get_weight_matrix(
	model: Module,
	layer_idx: int,
	use_logical_shapes: bool = True,
) -> torch.Tensor:
	"""
	Extract a weight matrix from a tcnn network as a 2D tensor.

	Parameters
	----------
	model : Network or NetworkWithInputEncoding
		The tiny-cuda-nn network model.

	layer_idx : int
		Index of the layer (0 = input layer, -1 = output layer).

	use_logical_shapes : bool, default=True
		If True, returns the matrix sliced to its logical (unpadded) shape.
		- Input layer: (network_width, input_width) instead of (network_width, padded_input_width)
		- Output layer: (output_width, network_width) instead of (padded_output_width, network_width)
		If False, returns the full padded matrix.

	Returns
	-------
	torch.Tensor
		Weight matrix of shape (fan_out, fan_in).

	Notes
	-----
	This returns a view into the parameter tensor, so modifications
	will affect the model parameters directly.
	"""
	structure = _get_network_structure(model)
	layer_shapes = _get_layer_shapes(structure)
	n_layers = len(layer_shapes)

	# Handle negative indexing
	if layer_idx < 0:
		layer_idx = n_layers + layer_idx

	if layer_idx < 0 or layer_idx >= n_layers:
		raise IndexError(
			f"layer_idx {layer_idx} out of range for network with "
			f"{n_layers} layers"
		)

	# Compute offset
	offset = sum(fo * fi for fo, fi in layer_shapes[:layer_idx])
	fan_out, fan_in = layer_shapes[layer_idx]
	n_elements = fan_out * fan_in

	# Get reshaped view
	weight = model.params.data[offset:offset + n_elements].view(fan_out, fan_in)

	if use_logical_shapes:
		is_first_layer = (layer_idx == 0)
		is_output_layer = (layer_idx == n_layers - 1)

		if is_first_layer:
			# Slice to logical input width
			logical_fan_in = structure['input_width']
			weight = weight[:, :logical_fan_in]
		elif is_output_layer:
			# Slice to logical output width
			logical_fan_out = structure['output_width']
			weight = weight[:logical_fan_out, :]

	return weight


def get_bias_vector(
	model: Module,
	layer_idx: int,
	use_logical_shapes: bool = True,
) -> torch.Tensor:
	"""
	Extract a bias vector from a tcnn network.

	Parameters
	----------
	model : Network or NetworkWithInputEncoding
		The tiny-cuda-nn network model.

	layer_idx : int
		Index of the layer (0 = input layer, -1 = output layer).

	use_logical_shapes : bool, default=True
		If True, returns the bias vector sliced to its logical (unpadded) size.
		- Hidden layers: network_width (typically already aligned)
		- Output layer: output_width instead of padded_output_width
		If False, returns the full padded bias vector.

	Returns
	-------
	torch.Tensor
		Bias vector.

	Raises
	------
	ValueError
		If the network does not have biases enabled.

	Notes
	-----
	This returns a view into the parameter tensor, so modifications
	will affect the model parameters directly.
	"""
	structure = _get_network_structure(model)

	if not structure['use_bias']:
		raise ValueError("Network does not have biases enabled")

	layer_shapes = _get_layer_shapes(structure)
	bias_sizes = _get_bias_sizes(structure)
	n_biases = len(bias_sizes)

	# Handle negative indexing
	if layer_idx < 0:
		layer_idx = n_biases + layer_idx

	if layer_idx < 0 or layer_idx >= n_biases:
		raise IndexError(
			f"layer_idx {layer_idx} out of range for network with "
			f"{n_biases} bias vectors"
		)

	# Compute offset (weights come first, then biases)
	total_weight_params = sum(fo * fi for fo, fi in layer_shapes)
	bias_offset = total_weight_params + sum(bias_sizes[:layer_idx])
	bias_size = bias_sizes[layer_idx]

	bias = model.params.data[bias_offset:bias_offset + bias_size]

	if use_logical_shapes:
		is_output_layer = (layer_idx == n_biases - 1)
		if is_output_layer:
			# Slice to logical output width
			logical_size = structure['output_width']
			bias = bias[:logical_size]
		else:
			# Hidden layers: slice to network_width (in case it's padded)
			logical_size = structure['network_width']
			bias = bias[:logical_size]

	return bias


# =============================================================================
# Muon Optimizer Utilities
# =============================================================================

def get_weight_matrices(
	model: Module,
	requires_grad: bool = True,
	use_logical_shapes: bool = True,
) -> list:
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

	use_logical_shapes : bool, default=True
		If True, returns matrices sliced to their logical (unpadded) shapes.
		This is important for optimizers like Muon that are shape-sensitive.
		- Input layer: (network_width, input_width) instead of (network_width, padded_input_width)
		- Output layer: (output_width, network_width) instead of (padded_output_width, network_width)
		If False, returns the full padded matrices.

	Returns
	-------
	list of torch.Tensor
		List of 2D weight matrices, one per layer.
		Each tensor has shape (fan_out, fan_in).

	Examples
	--------
	>>> weights = tcnn.get_weight_matrices(model)
	>>> print([w.shape for w in weights])
	[(64, 3), (64, 64), (1, 64)]  # Logical shapes (unpadded)
	"""
	structure = _get_network_structure(model)
	layer_shapes = _get_layer_shapes(structure)

	params = model.params if requires_grad else model.params.data

	weights = []
	offset = 0
	n_layers = len(layer_shapes)

	for i, (fan_out, fan_in) in enumerate(layer_shapes):
		n_elements = fan_out * fan_in
		weight = params[offset:offset + n_elements].view(fan_out, fan_in)

		if use_logical_shapes:
			is_first_layer = (i == 0)
			is_output_layer = (i == n_layers - 1)

			if is_first_layer:
				# Slice to logical input width
				logical_fan_in = structure['input_width']
				weight = weight[:, :logical_fan_in]
			elif is_output_layer:
				# Slice to logical output width
				logical_fan_out = structure['output_width']
				weight = weight[:logical_fan_out, :]

		weights.append(weight)
		offset += n_elements

	return weights


def get_bias_vectors(
	model: Module,
	requires_grad: bool = True,
	use_logical_shapes: bool = True,
) -> list:
	"""
	Get all bias vectors as 1D tensor views suitable for optimizers.

	Unlike `get_bias_vector()` which returns a view into `.data`, this
	function returns views that preserve gradient tracking, making them
	suitable for use with optimizers.

	Parameters
	----------
	model : Network or NetworkWithInputEncoding
		The tiny-cuda-nn network model.

	requires_grad : bool, default=True
		If True, returns views of `model.params` (preserves gradients).
		If False, returns views of `model.params.data` (no gradients).

	use_logical_shapes : bool, default=True
		If True, returns bias vectors sliced to their logical (unpadded) sizes.
		- Hidden layers: network_width (typically already aligned)
		- Output layer: output_width instead of padded_output_width
		If False, returns the full padded bias vectors.

	Returns
	-------
	list of torch.Tensor
		List of 1D bias vectors, one per layer.
		Empty list if the network has use_bias=False.

	Examples
	--------
	>>> biases = tcnn.get_bias_vectors(model)
	>>> print([b.shape for b in biases])
	[(64,), (64,), (1,)]  # Logical shapes (unpadded output)
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
	n_biases = len(bias_sizes)

	for i, bias_size in enumerate(bias_sizes):
		bias = params[offset:offset + bias_size]

		if use_logical_shapes:
			is_output_layer = (i == n_biases - 1)
			if is_output_layer:
				# Slice to logical output width
				logical_size = structure['output_width']
				bias = bias[:logical_size]
			else:
				# Hidden layers: slice to network_width (in case it's padded)
				logical_size = structure['network_width']
				bias = bias[:logical_size]

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
	use_logical_shapes: bool = True,
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

	use_logical_shapes : bool, default=True
		If True, weight matrices are sliced to their logical (unpadded) shapes.
		This is important because Muon's Newton-Schulz orthogonalization is
		shape-sensitive. For example, with output_width=1, the output layer
		is returned as (1, network_width) rather than (16, network_width).
		If False, returns the full padded matrices.

	Returns
	-------
	list of dict
		Parameter groups ready for torch.optim.Muon.
		First group: weight matrices (use_muon=True)
		Second group: biases (use_muon=False), if model has biases

	Examples
	--------
	>>> import torch
	>>> import tinycudann_inrs as tcnn
	>>>
	>>> model = tcnn.Network(
	...     n_input_dims=3, n_output_dims=1,
	...     network_config={
	...         "otype": "CutlassMLP",
	...         "activation": "ReLU",
	...         "n_neurons": 64,
	...         "n_hidden_layers": 2,
	...         "use_bias": True,
	...     }
	... )
	>>>
	>>> param_groups = tcnn.get_muon_param_groups(model, lr=0.02)
	>>> optimizer = torch.optim.Muon(param_groups)

	Notes
	-----
	The Muon optimizer applies Newton-Schulz orthogonalization to weight
	matrices, which helps with optimization stability and generalization.
	It should only be applied to 2D parameters (weight matrices), not to
	biases or other 1D parameters.

	Weight matrices are returned with logical (unpadded) shapes by default
	because Muon's orthogonalization behavior differs based on matrix shape.
	The unused padded dimensions in CutlassMLP would otherwise affect the
	orthogonalization computation.

	See: https://kellerjordan.github.io/posts/muon/
	"""
	weights = get_weight_matrices(model, requires_grad=True, use_logical_shapes=use_logical_shapes)
	biases = get_bias_vectors(model, requires_grad=True, use_logical_shapes=use_logical_shapes)

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
	use_logical_shapes: bool = True,
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

	use_logical_shapes : bool, default=True
		If True, weight matrices are sliced to their logical (unpadded) shapes.
		This is important because Muon's Newton-Schulz orthogonalization is
		shape-sensitive. See get_muon_param_groups for details.

	Returns
	-------
	torch.optim.Muon
		Configured Muon optimizer.

	Examples
	--------
	>>> import tinycudann_inrs as tcnn
	>>>
	>>> model = tcnn.NetworkWithInputEncoding(
	...     n_input_dims=3, n_output_dims=1,
	...     encoding_config={"otype": "Frequency", "n_frequencies": 6},
	...     network_config={
	...         "otype": "CutlassMLP",
	...         "activation": "ReLU",
	...         "n_neurons": 64,
	...         "n_hidden_layers": 3,
	...         "use_bias": True,
	...     }
	... )
	>>>
	>>> optimizer = tcnn.create_muon_optimizer(model, lr=0.02)
	>>> # Training loop
	>>> for batch in dataloader:
	...     optimizer.zero_grad()
	...     loss = compute_loss(model, batch)
	...     loss.backward()
	...     optimizer.step()

	Notes
	-----
	Requires PyTorch >= 2.0 with Muon optimizer support.
	If torch.optim.Muon is not available, this will raise an ImportError.

	Weight matrices are returned with logical (unpadded) shapes by default
	to ensure Muon's orthogonalization operates on the semantically correct
	matrix dimensions.
	"""
	if not hasattr(torch.optim, 'Muon'):
		raise ImportError(
			"torch.optim.Muon is not available. "
			"Please upgrade to PyTorch >= 2.6 or install a version with Muon support."
		)

	weights = get_weight_matrices(model, requires_grad=True, use_logical_shapes=use_logical_shapes)
	biases = get_bias_vectors(model, requires_grad=True, use_logical_shapes=use_logical_shapes)

	if adamw_lr is None:
		adamw_lr = lr * 0.1

	# Muon handles biases via adamw_params argument
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
