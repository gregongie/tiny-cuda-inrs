# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
tinycudann_inrs_fp32 - Float32 precision variant of tiny-cuda-nn

This package provides MLPs with float32 weights and biases, offering higher
numerical precision at the cost of performance (no tensor core acceleration).

Can be installed alongside tinycudann_inrs (fp16) for applications that need both.

Usage:
    import tinycudann_inrs_fp32 as tcnn_fp32

    model = tcnn_fp32.NetworkWithInputEncoding(
        n_input_dims=3,
        n_output_dims=1,
        encoding_config={"otype": "Frequency", "n_frequencies": 12},
        network_config={
            "otype": "CutlassMLP",  # Note: FullyFusedMLP not available in fp32
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        },
    )
"""

from tinycudann_inrs_fp32.modules import (
    supports_jit_fusion,
    free_temporary_memory,
    NetworkWithInputEncoding,
    Network,
    Encoding,
    # SIREN initialization utilities
    siren_init,
    siren_init_first_layer,
    inspect_network_params,
    get_weight_matrix,
    get_bias_vector,
    # Muon optimizer utilities
    get_weight_matrices,
    get_bias_vectors,
    get_muon_param_groups,
    create_muon_optimizer,
)

__all__ = [
    "supports_jit_fusion",
    "free_temporary_memory",
    "NetworkWithInputEncoding",
    "Network",
    "Encoding",
    # SIREN initialization utilities
    "siren_init",
    "siren_init_first_layer",
    "inspect_network_params",
    "get_weight_matrix",
    "get_bias_vector",
    # Muon optimizer utilities
    "get_weight_matrices",
    "get_bias_vectors",
    "get_muon_param_groups",
    "create_muon_optimizer",
]
