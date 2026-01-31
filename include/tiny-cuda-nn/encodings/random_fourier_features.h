/** @file   random_fourier_features.h
 *  @author Greg Ongie (with assitance from Claude Code)
 *  @brief  Implementation of random Fourier features encoding from Rahimi & Recht / Tancik et al.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>

#include <pcg32/pcg32.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>
#include <cmath>

namespace tcnn {

template <typename T>
__global__ void random_fourier_features_encoding(
	const uint32_t num_elements,
	const uint32_t n_features,
	const uint32_t n_dims_to_encode,
	const uint32_t n_to_pad,
	MatrixView<const float> data_in,
	const float* __restrict__ frequencies,
	MatrixView<T> data_out,
	float* __restrict__ dy_dx)
{
	const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (encoded_index >= num_elements) return;

	const uint32_t fan_out_encoded = n_features * 2;
	const uint32_t fan_out = fan_out_encoded + n_to_pad;

	const uint32_t i = encoded_index / fan_out;
	const uint32_t j = encoded_index - i * fan_out;

	if (j >= fan_out_encoded) {
		data_out(j, i) = (T)1.0f;
	} else {
		const uint32_t feature_idx = j / 2;
		const bool is_sin = (j % 2) == 1;

		// Compute dot product: B[feature_idx, :] dot x[i, :]
		float dot_product = 0.0f;
		for (uint32_t k = 0; k < n_dims_to_encode; ++k) {
			dot_product += frequencies[feature_idx * n_dims_to_encode + k] * data_in(k, i);
		}

		const float phase = 2.0f * PI() * dot_product;
		if (is_sin) {
			data_out(j, i) = (T)__sinf(phase);
		} else {
			data_out(j, i) = (T)__cosf(phase);
		}

		if (dy_dx != nullptr) {
			// Store intermediate values for backward pass
			// dy_dx layout: [batch, n_features * 2, n_dims_to_encode]
			const float cos_phase = __cosf(phase);
			const float sin_phase = __sinf(phase);
			for (uint32_t k = 0; k < n_dims_to_encode; ++k) {
				const float freq = frequencies[feature_idx * n_dims_to_encode + k];
				const float factor = 2.0f * PI() * freq;
				if (is_sin) {
					// d(sin(phase))/dx[k] = cos(phase) * 2*pi * B[j,k]
					dy_dx[i * fan_out_encoded * n_dims_to_encode + j * n_dims_to_encode + k] = factor * cos_phase;
				} else {
					// d(cos(phase))/dx[k] = -sin(phase) * 2*pi * B[j,k]
					dy_dx[i * fan_out_encoded * n_dims_to_encode + j * n_dims_to_encode + k] = -factor * sin_phase;
				}
			}
		}
	}
}

template <typename T>
__global__ void random_fourier_features_encoding_backward(
	const uint32_t num_elements,
	const uint32_t n_dims_to_encode,
	const uint32_t n_features,
	MatrixView<const T> dL_dy,
	const float* __restrict__ dy_dx,
	MatrixView<float> dL_dx
) {
	const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (encoded_index >= num_elements) return;

	const uint32_t i = encoded_index / n_dims_to_encode;
	const uint32_t k = encoded_index - i * n_dims_to_encode;

	const uint32_t outputs_per_feature = 2;
	const uint32_t n_outputs = n_features * outputs_per_feature;

	float result = 0.0f;
	for (uint32_t j = 0; j < n_outputs; ++j) {
		result += (float)dL_dy(j, i) * dy_dx[i * n_outputs * n_dims_to_encode + j * n_dims_to_encode + k];
	}
	dL_dx(k, i) = result;
}

template <typename T>
class RandomFourierFeaturesEncoding : public Encoding<T> {
public:
	RandomFourierFeaturesEncoding(uint32_t n_features, uint32_t n_dims_to_encode, float scale, uint32_t seed)
	: m_n_features{n_features}, m_n_dims_to_encode{n_dims_to_encode}, m_scale{scale}, m_seed{seed} {
		m_n_output_dims = m_n_features * 2;

		// Generate random Gaussian frequencies using Box-Muller transform
		const uint32_t n_freq_elements = m_n_features * m_n_dims_to_encode;
		std::vector<float> frequencies_host(n_freq_elements);

		pcg32 rng{(uint64_t)seed};

		// Box-Muller transform to generate Gaussian random numbers
		for (uint32_t i = 0; i < n_freq_elements; i += 2) {
			float u1 = rng.next_float();
			float u2 = rng.next_float();

			// Ensure u1 is not zero to avoid log(0)
			while (u1 == 0.0f) {
				u1 = rng.next_float();
			}

			float r = std::sqrt(-2.0f * std::log(u1)) * scale;
			float theta = 2.0f * PI() * u2;

			frequencies_host[i] = r * std::cos(theta);
			if (i + 1 < n_freq_elements) {
				frequencies_host[i + 1] = r * std::sin(theta);
			}
		}

		// Upload to GPU
		m_frequencies.resize(n_freq_elements);
		m_frequencies.copy_from_host(frequencies_host.data());

		// Store frequencies for JIT compilation
		m_frequencies_host = std::move(frequencies_host);
	}

#if !defined(TCNN_NO_FWD_BWD)
	std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		auto forward = std::make_unique<ForwardContext>();

		if (!output || padded_output_width() == 0) {
			return forward;
		}

		if (prepare_input_gradients) {
			forward->dy_dx = GPUMatrix<float>{m_n_output_dims * m_n_dims_to_encode, input.n(), stream};
		}

		linear_kernel(random_fourier_features_encoding<T>, 0, stream,
			input.n() * padded_output_width(),
			m_n_features,
			m_n_dims_to_encode,
			m_n_to_pad,
			input.view(),
			m_frequencies.data(),
			output->view(),
			forward->dy_dx.data()
		);

		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override {
		if (!dL_dinput) {
			return;
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		linear_kernel(random_fourier_features_encoding_backward<T>, 0, stream,
			input.n() * m_n_dims_to_encode,
			m_n_dims_to_encode,
			m_n_features,
			dL_doutput.view(),
			forward.dy_dx.data(),
			dL_dinput->view()
		);
	}
#endif // !defined(TCNN_NO_FWD_BWD)

	uint32_t input_width() const override {
		return m_n_dims_to_encode;
	}

	uint32_t padded_output_width() const override {
		return m_n_output_dims + m_n_to_pad;
	}

	uint32_t output_width() const override {
		return padded_output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1;
	}

	void set_padded_output_width(uint32_t padded_output_width) override {
		CHECK_THROW(padded_output_width >= m_n_output_dims);
		m_n_to_pad = padded_output_width - m_n_output_dims;
	}

	uint32_t required_output_alignment() const override {
		return 1;
	}

	MatrixLayout preferred_output_layout() const override {
		return AoS;
	}

	json hyperparams() const override {
		return {
			{"otype", "RandomFourierFeatures"},
			{"n_features", m_n_features},
			{"scale", m_scale},
			{"seed", m_seed},
		};
	}

	std::string generate_device_function(const std::string& name) const override {
		// Generate frequency array as literals
		std::ostringstream freq_array;
		freq_array << std::setprecision(9);
		freq_array << "constexpr float frequencies[] = {";
		for (uint32_t i = 0; i < m_frequencies_host.size(); ++i) {
			if (i > 0) freq_array << ", ";
			freq_array << m_frequencies_host[i] << "f";
		}
		freq_array << "};";

		std::ostringstream body;
		body << dfmt(1, R"(
				if (fwd_ctx) {{
					input.to_array((float*)fwd_ctx);
				}}

				{FREQ_ARRAY}

				{VEC_OUT} result;
				TCNN_PRAGMA_UNROLL
				for (uint32_t j = 0; j < {N_FEATURES}; ++j) {{
					float dot_product = 0.0f;
					TCNN_PRAGMA_UNROLL
					for (uint32_t k = 0; k < {N_DIMS}; ++k) {{
						dot_product += frequencies[j * {N_DIMS} + k] * input[k];
					}}
					const float phase = 2.0f * PI() * dot_product;
					result[2 * j]     = ({T})__cosf(phase);
					result[2 * j + 1] = ({T})__sinf(phase);
				}}
			)",
			"FREQ_ARRAY"_a = freq_array.str(),
			"VEC_OUT"_a = this->generate_vec_out(),
			"N_FEATURES"_a = m_n_features,
			"N_DIMS"_a = m_n_dims_to_encode,
			"T"_a = type_to_string<T>()
		) << "\n" << dfmt(1, R"(
				TCNN_PRAGMA_UNROLL
				for (uint32_t i = {N_OUT}; i < {N_PADDED_OUT}; ++i) {{
					result[i] = ({T})1.0f;
				}}
				return result;
			)",
			"N_OUT"_a = m_n_output_dims,
			"N_PADDED_OUT"_a = this->padded_output_width(),
			"T"_a = type_to_string<T>()
		);

		return this->generate_device_function_from_body(name, body.str());
	}

	std::string generate_backward_device_function(const std::string& name, uint32_t n_threads) const override {
		// Generate frequency array as literals
		std::ostringstream freq_array;
		freq_array << std::setprecision(9);
		freq_array << "constexpr float frequencies[] = {";
		for (uint32_t i = 0; i < m_frequencies_host.size(); ++i) {
			if (i > 0) freq_array << ", ";
			freq_array << m_frequencies_host[i] << "f";
		}
		freq_array << "};";

		return this->generate_backward_device_function_from_body(name, dfmt(1, R"(
				if (!dL_dx) {{
					return;
				}}

				{FREQ_ARRAY}

				{VEC_IN} input((float*)fwd_ctx), result(0.0f);

				TCNN_PRAGMA_UNROLL
				for (uint32_t j = 0; j < {N_FEATURES}; ++j) {{
					float dot_product = 0.0f;
					TCNN_PRAGMA_UNROLL
					for (uint32_t k = 0; k < {N_DIMS}; ++k) {{
						dot_product += frequencies[j * {N_DIMS} + k] * input[k];
					}}
					const float phase = 2.0f * PI() * dot_product;
					const float cos_phase = __cosf(phase);
					const float sin_phase = __sinf(phase);

					TCNN_PRAGMA_UNROLL
					for (uint32_t k = 0; k < {N_DIMS}; ++k) {{
						const float freq = frequencies[j * {N_DIMS} + k];
						const float factor = 2.0f * PI() * freq;
						// d(cos(phase))/dx[k] = -sin(phase) * 2*pi * B[j,k]
						// d(sin(phase))/dx[k] = cos(phase) * 2*pi * B[j,k]
						result[k] += (float)dL_dy[2 * j] * (-factor * sin_phase);
						result[k] += (float)dL_dy[2 * j + 1] * (factor * cos_phase);
					}}
				}}

				*dL_dx = result;
			)",
			"FREQ_ARRAY"_a = freq_array.str(),
			"VEC_IN"_a = this->generate_vec_in(),
			"N_FEATURES"_a = m_n_features,
			"N_DIMS"_a = m_n_dims_to_encode
		));
	}

	uint32_t device_function_fwd_ctx_bytes() const override {
		return this->input_width() * sizeof(float);
	}

private:
	struct ForwardContext : public Context {
		GPUMatrix<float> dy_dx;
	};

	uint32_t m_n_features;
	uint32_t m_n_dims_to_encode;
	float m_scale;
	uint32_t m_seed;

	// Frequencies matrix B: (n_features x n_dims_to_encode)
	GPUMemory<float> m_frequencies;
	std::vector<float> m_frequencies_host;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;
};

}
