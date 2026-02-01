# tinycudann_inrs

A fork of [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) with extensions for Implicit Neural Representations (INRs).

## Main Features

This fork extends tiny-cuda-nn with:

- **Random Fourier Features encoding** - Gaussian random frequency encoding from [Tancik et al. (2020)](https://arxiv.org/abs/2006.10739)
- **SIREN initialization** - Proper initialization for sinusoidal networks ([Sitzmann et al. 2020](https://github.com/vsitzmann/siren))
- **PyTorch-style initialization** - Re-initialize networks with PyTorch's default `nn.Linear` scheme
- **Muon optimizer utilities** - Helper functions for using the [Muon optimizer](https://kellerjordan.github.io/posts/muon/) with tcnn networks
- **Parameter inspection** - Utilities to access individual weight matrices and bias vectors

## Installation

### Prerequisites

- CUDA Toolkit 11.0+
- PyTorch with CUDA support
- C++17 compatible compiler

### Install as `tinycudann_inrs` (FP16 - Default)

This installs alongside any existing `tinycudann` installation:

```bash
# Clone and enter the repository
git clone <repo-url> tiny-cuda-inrs
cd tiny-cuda-inrs
git submodule update --init --recursive

# Install PyTorch bindings (FP16 precision, uses tensor cores)
cd bindings/torch
pip install .
```

## Random Fourier Features Encoding

Implements random Fourier features from [Rahimi & Recht (2007)](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html), proposed for neural networks in [Tancik et al. (2020)](https://arxiv.org/abs/2006.10739).

### Mathematical Background

Unlike the standard `FrequencyEncoding` which uses axis-aligned frequencies at powers of 2, this encoding uses random Gaussian frequency vectors:

```
γ(x) = [cos(2πBx), sin(2πBx)]
```

Where:
- `x` is the input vector (n_dims_to_encode dimensions)
- `B` is a (n_features × n_dims_to_encode) matrix with entries sampled from N(0, σ²)
- `σ` (scale parameter) controls the bandwidth
- Output dimension: `2 * n_features`

### Configuration

```json
{
    "otype": "RandomFourierFeatures",
    "n_features": 128,
    "scale": 10.0,
    "seed": 1337
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_features` | 128 | Number of random frequency vectors |
| `scale` | 10.0 | Standard deviation of Gaussian frequencies (controls bandwidth) |
| `seed` | 1337 | Random seed for reproducibility |

### Usage

#### Standalone Encoding

```python
import tinycudann_inrs as tcnn
import torch

encoding = tcnn.Encoding(
    n_input_dims=3,
    encoding_config={
        "otype": "RandomFourierFeatures",
        "n_features": 128,
        "scale": 10.0,
        "seed": 42,
    },
    dtype=torch.float32,  # or torch.float16
)

x = torch.rand(1024, 3, device="cuda")
y = encoding(x)  # shape: (1024, 256)
```

#### With Network

```python
model = tcnn.NetworkWithInputEncoding(
    n_input_dims=3,
    n_output_dims=1,
    encoding_config={
        "otype": "RandomFourierFeatures",
        "n_features": 64,
        "scale": 10.0,
    },
    network_config={
        "otype": "FullyFusedMLP",  # or "CutlassMLP" for larger widths
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2,
    },
)

x = torch.rand(1024, 3, device="cuda")
y = model(x)  # shape: (1024, 1)
```

## SIREN Initialization

Utilities for [SIREN](https://github.com/vsitzmann/siren) (Sinusoidal Representation Networks) initialization. Since tcnn's `Sine` activation computes `sin(x)` without the ω₀ scaling factor, these utilities absorb ω₀ into the weights and biases.

### Basic Usage

```python
import tinycudann_inrs as tcnn

model = tcnn.Network(
    n_input_dims=2,
    n_output_dims=3,
    network_config={
        "otype": "CutlassMLP",
        "activation": "Sine",
        "output_activation": "None",
        "n_neurons": 256,
        "n_hidden_layers": 5,
        "use_bias": True,
    }
)

# Re-initialize with SIREN scheme
tcnn.siren_init(model, omega_0=30.0, bias_init='siren')
```

### SIREN Functions

| Function | Description |
|----------|-------------|
| `siren_init(model, omega_0=30.0, first_layer_omega_0=None, bias_init='zero', seed=None)` | Full SIREN initialization for all layers |
| `siren_init_first_layer(model, omega_0=30.0, bias_init='zero', seed=None)` | Re-initialize only the first layer |

### Bias Initialization Options

- `'zero'`: Initialize biases to zero (default)
- `'siren'`: Initialize as ω₀ × U[-1/√fan_in, 1/√fan_in]
- `'uniform'`: Initialize as U[-1/fan_in, 1/fan_in]

## PyTorch-Style Initialization

If you want to match PyTorch's default `nn.Linear` initialization instead of tcnn's Xavier uniform, use `pytorch_init()`:

```python
import tinycudann_inrs as tcnn

model = tcnn.Network(
    n_input_dims=3,
    n_output_dims=1,
    network_config={
        "otype": "CutlassMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2,
        "use_bias": True,
    }
)

# Re-initialize with PyTorch's default scheme
tcnn.pytorch_init(model, seed=42)
```

This applies Kaiming uniform initialization (with `a=sqrt(5)`) to both weights and biases:
- **Weights**: U[-1/√fan_in, 1/√fan_in]
- **Biases**: U[-1/√fan_in, 1/√fan_in]

The function uses logical (unpadded) dimensions for computing bounds, so initialization is independent of CutlassMLP's internal padding.

| Function | Description |
|----------|-------------|
| `pytorch_init(model, seed=None)` | Re-initialize with PyTorch's default nn.Linear scheme |

## Parameter Inspection

Utilities for inspecting and accessing the internal parameter layout of tcnn networks.

### Inspect Network Structure

```python
import tinycudann_inrs as tcnn

model = tcnn.Network(n_input_dims=3, n_output_dims=1, network_config={...})

# Print detailed parameter layout
info = tcnn.inspect_network_params(model)
# Output:
# Network Structure:
#   Input width: 3
#   Network width: 64
#   Output width: 1 (padded: 16)
#   Hidden layers: 2
#   Use bias: True
#   Activation: ReLU
# ...
```

### Access Individual Layers

```python
# Get weight matrix for layer 0 (input layer)
W0 = tcnn.get_weight_matrix(model, layer_idx=0)  # shape: (64, 3)

# Get output layer weights
W_out = tcnn.get_weight_matrix(model, layer_idx=-1)  # shape: (16, 64)

# Get bias vector for layer 0
b0 = tcnn.get_bias_vector(model, layer_idx=0)  # shape: (64,)
```

## Muon Optimizer Support

Utilities for using the [Muon optimizer](https://kellerjordan.github.io/posts/muon/) with tcnn networks. Muon applies Newton-Schulz orthogonalization to weight matrices and is designed for 2D parameters only.

### Quick Start

```python
import tinycudann_inrs as tcnn

model = tcnn.NetworkWithInputEncoding(
    n_input_dims=3,
    n_output_dims=1,
    encoding_config={"otype": "Frequency", "n_frequencies": 6},
    network_config={
        "otype": "CutlassMLP",
        "activation": "ReLU",
        "n_neurons": 64,
        "n_hidden_layers": 3,
        "use_bias": True,
    }
)

# Create Muon optimizer (handles weight matrices + biases automatically)
optimizer = tcnn.create_muon_optimizer(model, lr=0.02)

# Training loop
for x, y in dataloader:
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
```

### Muon Functions

| Function | Description |
|----------|-------------|
| `create_muon_optimizer(model, lr=0.02, ...)` | Create a fully configured Muon optimizer |
| `get_muon_param_groups(model, lr=0.02, ...)` | Get param groups for manual optimizer setup |
| `get_weight_matrices(model, requires_grad=True)` | Get all weight matrices as 2D tensor views |
| `get_bias_vectors(model, requires_grad=True)` | Get all bias vectors as 1D tensor views |

### Manual Setup (Advanced)

For more control over the optimizer configuration:

```python
import torch
import tinycudann_inrs as tcnn

# Get structured parameters with gradient tracking
weights = tcnn.get_weight_matrices(model)  # List of 2D tensors
biases = tcnn.get_bias_vectors(model)      # List of 1D tensors

# Option 1: Use Muon's built-in AdamW for biases
optimizer = torch.optim.Muon(
    weights,
    lr=0.02,
    momentum=0.95,
    adamw_params=biases,
    adamw_lr=0.002,
)

# Option 2: Use param groups with use_muon flag
param_groups = tcnn.get_muon_param_groups(model, lr=0.02, adamw_lr=0.002)
optimizer = torch.optim.Muon(param_groups)
```

### create_muon_optimizer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 0.02 | Learning rate for weight matrices |
| `momentum` | 0.95 | Momentum for Muon |
| `nesterov` | True | Use Nesterov momentum |
| `ns_steps` | 5 | Newton-Schulz iteration steps |
| `weight_decay` | 0.0 | Weight decay for weight matrices |
| `adamw_lr` | lr × 0.1 | Learning rate for biases (AdamW) |
| `adamw_betas` | (0.9, 0.95) | Betas for AdamW |
| `adamw_wd` | 0.0 | Weight decay for biases |

**Note:** Requires PyTorch >= 2.6 with Muon optimizer support.

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `NetworkWithInputEncoding` | Encoding + MLP in a single efficient module |
| `Network` | Standalone MLP network |
| `Encoding` | Standalone input encoding |

### Initialization Functions

| Function | Description |
|----------|-------------|
| `siren_init(model, ...)` | SIREN initialization for sinusoidal networks |
| `siren_init_first_layer(model, ...)` | Re-initialize only the first layer with SIREN |
| `pytorch_init(model, seed=None)` | PyTorch's default nn.Linear initialization |

### Utility Functions

| Function | Description |
|----------|-------------|
| `supports_jit_fusion()` | Check if JIT fusion is available |
| `free_temporary_memory()` | Free tcnn's temporary GPU allocations |
| `inspect_network_params(model)` | Inspect network parameter layout |
| `get_weight_matrix(model, layer_idx)` | Get a single weight matrix |
| `get_bias_vector(model, layer_idx)` | Get a single bias vector |
| `get_weight_matrices(model)` | Get all weight matrices (for optimizers) |
| `get_bias_vectors(model)` | Get all bias vectors (for optimizers) |
| `get_muon_param_groups(model, ...)` | Get param groups for Muon optimizer |
| `create_muon_optimizer(model, ...)` | Create a configured Muon optimizer |

## Notes

- **Reproducibility**: Same seed produces identical frequency matrices
- **Scale parameter**: Higher values = higher frequency content = more detail but potentially harder to optimize
- **Large n_features**: No hard limit, but very large values may slow JIT compilation. For >64 features (>128 output dims), consider using `CutlassMLP` instead of `FullyFusedMLP`
- **Precision**: Encodings support runtime dtype selection (`torch.float32` or `torch.float16`). Network weights use FP16 with tensor core acceleration.
