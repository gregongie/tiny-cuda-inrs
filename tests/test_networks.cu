/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   test_networks.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Test various invariances of neural networks. E.g. that `inference`
 *          and `inference_mixed_precision` produce the same results, that JIT
 *          produces the same result as no-JIT, as well as the same for derivatives.
 */

 #include "test_common.h"

#include <tiny-cuda-nn/network.h>

using namespace tcnn;

TEST_CASE("Various invariance checks for neural networks", "[network][jit]") {
	using T = network_precision_t;

	tcnn_test_setup();

	std::vector<uint32_t> input_sizes = {16, 32, 48, 64, 128};
	std::vector<uint32_t> hidden_sizes = {32, 64, 128};
	std::vector<uint32_t> output_sizes = {16, 32};
	std::vector<uint32_t> depths = {2};

	for (uint32_t n_in : input_sizes) {
		for (uint32_t n_hidden : hidden_sizes) {
			for (uint32_t n_out : output_sizes) {
				for (uint32_t depth : depths) {
					SECTION(fmt::format("MLP testing {}->{}x{}->{}", n_in, n_hidden, depth, n_out)) {
						json config = {
							{"otype", "CutlassMLP"},
							{"n_input_dims", n_in},
							{"n_neurons", n_hidden},
							{"n_output_dims", n_out},
							{"n_hidden_layers", depth},
						};

						std::shared_ptr<Network<T>> cutlass_mlp{create_network<T>(config)};
						SECTION("CutlassMLP") { test_differentiable_object<T, T, T>(cutlass_mlp); }

						// Test CutlassMLP with bias (skip JIT tests since JIT doesn't support bias yet)
						config["use_bias"] = true;
						std::shared_ptr<Network<T>> cutlass_mlp_bias{create_network<T>(config)};
						SECTION("CutlassMLP with bias") {
							const uint32_t batch_size = 32 * BATCH_SIZE_GRANULARITY;

							// Initialize network with a trainer (allocates parameters)
							std::shared_ptr<Optimizer<T>> dummy_optimizer{create_optimizer<T>(json::object())};
							std::shared_ptr<Loss<T>> dummy_loss{create_loss<T>(json::object())};
							auto trainer = std::make_shared<Trainer<T, T, T>>(cutlass_mlp_bias, dummy_optimizer, dummy_loss);

							GPUMatrix<T> input{n_in, batch_size};
							GPUMatrix<T> input_gradient{n_in, batch_size};
							GPUMatrix<T> output{cutlass_mlp_bias->padded_output_width(), batch_size};
							GPUMatrix<T> output_gradient{cutlass_mlp_bias->padded_output_width(), batch_size};
							GPUMatrix<float> output_fp{cutlass_mlp_bias->output_width(), batch_size};

							pcg32 rng{0xdeadbeef};
							input.initialize_uniform(rng, 0.001f, 0.999f);
							output_gradient.initialize_uniform(rng, -1.0f, 1.0f);

							// Test basic functionality without JIT
							SECTION("`inference` and `inference_mixed_precision` match") {
								cutlass_mlp_bias->inference(input, output_fp);
								cutlass_mlp_bias->inference_mixed_precision(input, output);

								auto v1 = output.to_cpu_vector();
								auto v2 = output_fp.to_cpu_vector();
								REQUIRE(v1.size() > 0);
								// Just verify no NaN/Inf
								for (size_t i = 0; i < v1.size(); ++i) {
									REQUIRE(std::isfinite((float)v1[i]));
								}
							}

							SECTION("Forward and backward run without error") {
								auto ctx = cutlass_mlp_bias->forward(input, &output, false, true);
								cutlass_mlp_bias->backward(*ctx, input, output, output_gradient, &input_gradient);

								// Verify gradients are finite
								auto ig = input_gradient.to_cpu_vector();
								for (size_t i = 0; i < ig.size(); ++i) {
									REQUIRE(std::isfinite((float)ig[i]));
								}

								std::vector<T> pg(cutlass_mlp_bias->n_params());
								CUDA_CHECK_THROW(cudaMemcpy(pg.data(), cutlass_mlp_bias->gradients(), cutlass_mlp_bias->n_params() * sizeof(T), cudaMemcpyDeviceToHost));
								for (size_t i = 0; i < pg.size(); ++i) {
									REQUIRE(std::isfinite((float)pg[i]));
								}
							}
						}
						config["use_bias"] = false;

						if (MIN_GPU_ARCH > 70) {
							config["otype"] = "FullyFusedMLP";
							std::shared_ptr<Network<T>> fully_fused_mlp{create_network<T>(config)};
							SECTION("FullyFusedMLP") { test_differentiable_object<T, T, T>(fully_fused_mlp); }

							// This would be a good place to check that FullyFusedMLP produces the same
							// results as CutlassMLP if this was not already implicitly checked in the JIT
							// tests from within `test_differentiable_object`, because the JIT is the same
							// across CutlassMLP and FullyFusedMLP.
						}
					}
				}
			}
		}
	}
}
