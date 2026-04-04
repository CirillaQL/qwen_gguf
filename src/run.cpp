#include "run.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

float fp16_to_fp32(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1FU;
    const uint32_t mantissa = bits & 0x03FFU;

    uint32_t out_bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            out_bits = sign;
        } else {
            uint32_t mantissa_normalized = mantissa;
            int32_t exp = -14;
            while ((mantissa_normalized & 0x0400U) == 0) {
                mantissa_normalized <<= 1;
                --exp;
            }
            mantissa_normalized &= 0x03FFU;
            out_bits = sign
                     | (static_cast<uint32_t>(exp + 127) << 23)
                     | (mantissa_normalized << 13);
        }
    } else if (exponent == 0x1FU) {
        out_bits = sign | 0x7F800000U | (mantissa << 13);
    } else {
        out_bits = sign
                 | ((exponent - 15U + 127U) << 23)
                 | (mantissa << 13);
    }

    float value = 0.0f;
    std::memcpy(&value, &out_bits, sizeof(value));
    return value;
}

std::vector<float> read_embedding_row(
    const gguf_tensor_data &tensor,
    uint64_t row_index
) {
    if (tensor.info.dimensions.size() != 2) {
        throw std::runtime_error("embedding tensor must be 2D");
    }

    const uint64_t hidden_size = tensor.info.dimensions[0];
    const uint64_t vocab_size = tensor.info.dimensions[1];
    if (row_index >= vocab_size) {
        throw std::runtime_error("token id out of range for embedding tensor");
    }

    if (hidden_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("embedding hidden size is too large");
    }

    const size_t hidden = static_cast<size_t>(hidden_size);
    std::vector<float> result(hidden);

    switch (tensor.info.type) {
        case GGML_TYPE_F32: {
            const size_t offset = static_cast<size_t>(row_index) * hidden * sizeof(float);
            const float *data = reinterpret_cast<const float *>(tensor.raw_data.data() + offset);
            result.assign(data, data + hidden);
            return result;
        }
        case GGML_TYPE_F16: {
            const size_t offset = static_cast<size_t>(row_index) * hidden * sizeof(uint16_t);
            const uint16_t *data = reinterpret_cast<const uint16_t *>(tensor.raw_data.data() + offset);
            for (size_t i = 0; i < hidden; ++i) {
                result[i] = fp16_to_fp32(data[i]);
            }
            return result;
        }
        default:
            throw std::runtime_error("unsupported embedding tensor type");
    }
}

float dot_product_f32_row(
    const float *row_data,
    const float *input_data,
    uint32_t input_dim
) {
    float sum = 0.0f;
    for (uint32_t in_index = 0; in_index < input_dim; ++in_index) {
        sum += input_data[in_index] * row_data[in_index];
    }
    return sum;
}

float dot_product_f16_row(
    const uint16_t *row_data,
    const float *input_data,
    uint32_t input_dim
) {
    float sum = 0.0f;
    for (uint32_t in_index = 0; in_index < input_dim; ++in_index) {
        sum += input_data[in_index] * fp16_to_fp32(row_data[in_index]);
    }
    return sum;
}

std::vector<float> read_tensor_vector(const gguf_tensor_data &tensor) {
    if (tensor.info.dimensions.size() != 1) {
        throw std::runtime_error("tensor must be 1D");
    }

    const uint64_t width_u64 = tensor.info.dimensions[0];
    if (width_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("tensor width is too large");
    }

    const size_t width = static_cast<size_t>(width_u64);
    std::vector<float> values(width);

    switch (tensor.info.type) {
        case GGML_TYPE_F32: {
            const float *data = reinterpret_cast<const float *>(tensor.raw_data.data());
            values.assign(data, data + width);
            return values;
        }
        case GGML_TYPE_F16: {
            const uint16_t *data = reinterpret_cast<const uint16_t *>(tensor.raw_data.data());
            for (size_t i = 0; i < width; ++i) {
                values[i] = fp16_to_fp32(data[i]);
            }
            return values;
        }
        default:
            throw std::runtime_error("unsupported tensor type");
    }
}

void gemm_batch(
    const gguf_tensor_data &weight,
    const std::vector<float> &input,
    uint32_t batch_size,
    uint32_t input_dim,
    uint32_t output_dim,
    std::vector<float> &output,
    const std::vector<float> *bias = nullptr
) {
    if (weight.info.dimensions.size() != 2) {
        throw std::runtime_error("projection weight must be 2D");
    }
    if (weight.info.dimensions[0] != input_dim) {
        throw std::runtime_error("projection input dimension mismatch");
    }
    if (weight.info.dimensions[1] != output_dim) {
        throw std::runtime_error("projection output dimension mismatch");
    }
    if (input.size() != static_cast<size_t>(batch_size) * input_dim) {
        throw std::runtime_error("projection input buffer size mismatch");
    }
    if (bias != nullptr && bias->size() != output_dim) {
        throw std::runtime_error("projection bias size mismatch");
    }

    output.assign(static_cast<size_t>(batch_size) * output_dim, 0.0f);

    switch (weight.info.type) {
        case GGML_TYPE_F32: {
            const float *weight_data = reinterpret_cast<const float *>(weight.raw_data.data());
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (uint32_t out_index = 0; out_index < output_dim; ++out_index) {
                const float *row_data = weight_data + static_cast<size_t>(out_index) * input_dim;
                for (uint32_t batch_index = 0; batch_index < batch_size; ++batch_index) {
                    const size_t input_offset = static_cast<size_t>(batch_index) * input_dim;
                    const size_t output_offset = static_cast<size_t>(batch_index) * output_dim;

                    float sum = bias == nullptr ? 0.0f : (*bias)[out_index];
                    sum += dot_product_f32_row(row_data, input.data() + input_offset, input_dim);
                    output[output_offset + out_index] = sum;
                }
            }
            return;
        }
        case GGML_TYPE_F16: {
            const uint16_t *weight_data = reinterpret_cast<const uint16_t *>(weight.raw_data.data());
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (uint32_t out_index = 0; out_index < output_dim; ++out_index) {
                const uint16_t *row_data = weight_data + static_cast<size_t>(out_index) * input_dim;
                for (uint32_t batch_index = 0; batch_index < batch_size; ++batch_index) {
                    const size_t input_offset = static_cast<size_t>(batch_index) * input_dim;
                    const size_t output_offset = static_cast<size_t>(batch_index) * output_dim;

                    float sum = bias == nullptr ? 0.0f : (*bias)[out_index];
                    sum += dot_product_f16_row(row_data, input.data() + input_offset, input_dim);
                    output[output_offset + out_index] = sum;
                }
            }
            return;
        }
        default:
            throw std::runtime_error("unsupported tensor type");
    }
}

void weighted_rmsnorm_batch(
    const std::vector<float> &input,
    uint32_t batch_size,
    uint32_t hidden_size,
    const std::vector<float> &weight,
    float eps,
    std::vector<float> &output
) {
    if (hidden_size == 0) {
        throw std::runtime_error("hidden size must be greater than zero");
    }
    if (input.size() != static_cast<size_t>(batch_size) * hidden_size) {
        throw std::runtime_error("RMSNorm input buffer size mismatch");
    }
    if (weight.size() != hidden_size) {
        throw std::runtime_error("RMSNorm weight size mismatch");
    }

    output.resize(input.size());
    for (uint32_t token_index = 0; token_index < batch_size; ++token_index) {
        const size_t begin = static_cast<size_t>(token_index) * hidden_size;
        const size_t end = begin + hidden_size;

        float sum_of_squares = 0.0f;
        for (size_t i = begin; i < end; ++i) {
            sum_of_squares += input[i] * input[i];
        }

        const float mean_of_squares = sum_of_squares / static_cast<float>(hidden_size);
        const float inv_rms = 1.0f / std::sqrt(mean_of_squares + eps);

        for (uint32_t hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
            output[begin + hidden_index] =
                input[begin + hidden_index] * inv_rms * weight[hidden_index];
        }
    }
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

}  // namespace

float embedding_batch::at(uint32_t token_index, uint32_t hidden_index) const {
    if (token_index >= token_count) {
        throw std::out_of_range("token index out of range");
    }
    if (hidden_index >= hidden_size) {
        throw std::out_of_range("hidden index out of range");
    }
    return values[static_cast<size_t>(token_index) * hidden_size + hidden_index];
}

std::vector<float> embedding_batch::token_embedding(uint32_t token_index) const {
    if (token_index >= token_count) {
        throw std::out_of_range("token index out of range");
    }

    const size_t begin = static_cast<size_t>(token_index) * hidden_size;
    const size_t end = begin + hidden_size;
    return std::vector<float>(values.begin() + begin, values.begin() + end);
}

embedding_batch lookup_embeddings(
    const gguf_model &model,
    const std::vector<int32_t> &token_ids,
    const std::string &tensor_name
) {
    const gguf_tensor_data &tensor = load_gguf_tensor_data(model, tensor_name);
    if (tensor.info.dimensions.size() != 2) {
        throw std::runtime_error("embedding tensor must be 2D");
    }

    const uint64_t hidden_size = tensor.info.dimensions[0];
    if (hidden_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("embedding hidden size does not fit in uint32_t");
    }
    if (token_ids.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("token batch size does not fit in uint32_t");
    }

    embedding_batch batch{};
    batch.token_count = static_cast<uint32_t>(token_ids.size());
    batch.hidden_size = static_cast<uint32_t>(hidden_size);
    batch.values.reserve(static_cast<size_t>(batch.token_count) * batch.hidden_size);

    for (int32_t token_id : token_ids) {
        if (token_id < 0) {
            throw std::runtime_error("token id must be non-negative");
        }
        std::vector<float> row = read_embedding_row(tensor, static_cast<uint64_t>(token_id));
        batch.values.insert(batch.values.end(), row.begin(), row.end());
    }

    return batch;
}

embedding_batch RMSNorm(const embedding_batch &token_embedding, float eps) {
    if (token_embedding.hidden_size == 0) {
        throw std::runtime_error("hidden size must be greater than zero");
    }

    embedding_batch result{};
    result.token_count = token_embedding.token_count;
    result.hidden_size = token_embedding.hidden_size;
    result.values.resize(token_embedding.values.size());

    for (uint32_t token_index = 0; token_index < token_embedding.token_count; ++token_index) {
        const size_t begin = static_cast<size_t>(token_index) * token_embedding.hidden_size;
        const size_t end = begin + token_embedding.hidden_size;

        float sum_of_squares = 0.0f;
        for (size_t i = begin; i < end; ++i) {
            sum_of_squares += token_embedding.values[i] * token_embedding.values[i];
        }

        const float mean_of_squares = sum_of_squares / static_cast<float>(token_embedding.hidden_size);
        const float inv_rms = 1.0f / std::sqrt(mean_of_squares + eps);

        for (size_t i = begin; i < end; ++i) {
            result.values[i] = token_embedding.values[i] * inv_rms;
        }
    }

    return result;
}

void run_state::compute_qkv(const gguf_model &model, uint32_t layer_index) {
    if (layer_index >= shape_.n_layers) {
        throw std::out_of_range("layer index out of range");
    }
    if (shape_.hidden_size == 0) {
        throw std::runtime_error("hidden size must be greater than zero");
    }
    if (norm_.size() != static_cast<size_t>(shape_.batch_size) * shape_.hidden_size) {
        throw std::runtime_error("norm buffer size mismatch");
    }

    const std::string prefix = "blk." + std::to_string(layer_index);
    const std::string q_weight_name = prefix + ".attn_q.weight";
    const std::string k_weight_name = prefix + ".attn_k.weight";
    const std::string v_weight_name = prefix + ".attn_v.weight";
    const gguf_tensor_data &q_weight = load_gguf_tensor_data(model, q_weight_name);
    const gguf_tensor_data &k_weight = load_gguf_tensor_data(model, k_weight_name);
    const gguf_tensor_data &v_weight = load_gguf_tensor_data(model, v_weight_name);
    const std::vector<float> q_bias =
        read_tensor_vector(load_gguf_tensor_data(model, prefix + ".attn_q.bias"));
    const std::vector<float> k_bias =
        read_tensor_vector(load_gguf_tensor_data(model, prefix + ".attn_k.bias"));
    const std::vector<float> v_bias =
        read_tensor_vector(load_gguf_tensor_data(model, prefix + ".attn_v.bias"));

    gemm_batch(
        q_weight,
        norm_,
        shape_.batch_size,
        shape_.hidden_size,
        shape_.q_dim(),
        q_,
        &q_bias
    );
    gemm_batch(
        k_weight,
        norm_,
        shape_.batch_size,
        shape_.hidden_size,
        shape_.kv_dim(),
        k_,
        &k_bias
    );
    gemm_batch(
        v_weight,
        norm_,
        shape_.batch_size,
        shape_.hidden_size,
        shape_.kv_dim(),
        v_,
        &v_bias
    );
}

void run_state::apply_rope(uint32_t position_index) {
    if ((shape_.head_dim % 2U) != 0U) {
        throw std::runtime_error("RoPE requires an even head_dim");
    }
    if (shape_.batch_size == 0) {
        return;
    }

    constexpr float kRopeBase = 1000000.0f;
    const uint32_t half_dim = shape_.head_dim / 2U;

    for (uint32_t token_index = 0; token_index < shape_.batch_size; ++token_index) {
        const uint32_t position = position_index + token_index;

        for (uint32_t head_index = 0; head_index < shape_.n_heads; ++head_index) {
            const size_t head_offset =
                (static_cast<size_t>(token_index) * shape_.n_heads + head_index) * shape_.head_dim;

            for (uint32_t pair_index = 0; pair_index < half_dim; ++pair_index) {
                const float exponent = (2.0f * static_cast<float>(pair_index))
                                     / static_cast<float>(shape_.head_dim);
                const float theta = static_cast<float>(position) / std::pow(kRopeBase, exponent);
                const float cos_theta = std::cos(theta);
                const float sin_theta = std::sin(theta);

                const size_t even_index = head_offset + pair_index;
                const size_t odd_index = head_offset + pair_index + half_dim;

                const float x_even = q_[even_index];
                const float x_odd = q_[odd_index];
                q_[even_index] = x_even * cos_theta - x_odd * sin_theta;
                q_[odd_index] = x_even * sin_theta + x_odd * cos_theta;
            }
        }

        for (uint32_t head_index = 0; head_index < shape_.n_kv_heads; ++head_index) {
            const size_t head_offset =
                (static_cast<size_t>(token_index) * shape_.n_kv_heads + head_index) * shape_.head_dim;

            for (uint32_t pair_index = 0; pair_index < half_dim; ++pair_index) {
                const float exponent = (2.0f * static_cast<float>(pair_index))
                                     / static_cast<float>(shape_.head_dim);
                const float theta = static_cast<float>(position) / std::pow(kRopeBase, exponent);
                const float cos_theta = std::cos(theta);
                const float sin_theta = std::sin(theta);

                const size_t even_index = head_offset + pair_index;
                const size_t odd_index = head_offset + pair_index + half_dim;

                const float x_even = k_[even_index];
                const float x_odd = k_[odd_index];
                k_[even_index] = x_even * cos_theta - x_odd * sin_theta;
                k_[odd_index] = x_even * sin_theta + x_odd * cos_theta;
            }
        }
    }
}

void run_state::update_kv_cache(uint32_t layer_index, uint32_t position_index) {
    if (layer_index >= shape_.n_layers) {
        throw std::out_of_range("layer index out of range");
    }
    if (position_index + shape_.batch_size > shape_.max_seq_len) {
        throw std::out_of_range("KV-cache position out of range");
    }
    if (k_.size() != kv_elements() || v_.size() != kv_elements()) {
        throw std::runtime_error("KV buffer size mismatch");
    }

    const size_t kv_stride = shape_.kv_dim();
    for (uint32_t token_index = 0; token_index < shape_.batch_size; ++token_index) {
        const size_t src_offset = static_cast<size_t>(token_index) * kv_stride;
        const size_t dst_offset = kv_cache_offset(layer_index, position_index + token_index);

        std::copy(
            k_.begin() + src_offset,
            k_.begin() + src_offset + kv_stride,
            key_cache_.begin() + dst_offset
        );
        std::copy(
            v_.begin() + src_offset,
            v_.begin() + src_offset + kv_stride,
            value_cache_.begin() + dst_offset
        );
    }
}

void run_state::compute_attention(const gguf_model &model, uint32_t layer_index, uint32_t position_index) {
    if (shape_.n_heads == 0 || shape_.n_kv_heads == 0 || shape_.head_dim == 0) {
        throw std::runtime_error("attention shape is not initialized");
    }
    if ((shape_.n_heads % shape_.n_kv_heads) != 0U) {
        throw std::runtime_error("n_heads must be divisible by n_kv_heads");
    }
    if (position_index + shape_.batch_size > shape_.max_seq_len) {
        throw std::out_of_range("attention position out of range");
    }

    apply_rope(position_index);
    update_kv_cache(layer_index, position_index);

    const uint32_t kv_group_size = shape_.n_heads / shape_.n_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape_.head_dim));

    std::fill(attn_scores_.begin(), attn_scores_.end(), 0.0f);
    std::fill(attn_probs_.begin(), attn_probs_.end(), 0.0f);
    std::fill(attn_ctx_.begin(), attn_ctx_.end(), 0.0f);

    for (uint32_t token_index = 0; token_index < shape_.batch_size; ++token_index) {
        const uint32_t query_position = position_index + token_index;
        const uint32_t seq_len = query_position + 1U;

        for (uint32_t head_index = 0; head_index < shape_.n_heads; ++head_index) {
            const uint32_t kv_head_index = head_index / kv_group_size;
            const size_t q_offset =
                (static_cast<size_t>(token_index) * shape_.n_heads + head_index) * shape_.head_dim;
            const size_t score_offset =
                (static_cast<size_t>(token_index) * shape_.n_heads + head_index) * shape_.max_seq_len;

            bool has_score = false;
            float max_score = 0.0f;
            for (uint32_t past_position = 0; past_position < seq_len; ++past_position) {
                const size_t k_offset = kv_cache_offset(layer_index, past_position)
                                      + static_cast<size_t>(kv_head_index) * shape_.head_dim;

                float score = 0.0f;
                for (uint32_t dim_index = 0; dim_index < shape_.head_dim; ++dim_index) {
                    score += q_[q_offset + dim_index] * key_cache_[k_offset + dim_index];
                }
                score *= scale;
                attn_scores_[score_offset + past_position] = score;
                if (!has_score || score > max_score) {
                    max_score = score;
                    has_score = true;
                }
            }
            if (!has_score) {
                throw std::runtime_error("attention produced no scores");
            }

            float exp_sum = 0.0f;
            for (uint32_t past_position = 0; past_position < seq_len; ++past_position) {
                const float prob = std::exp(attn_scores_[score_offset + past_position] - max_score);
                attn_probs_[score_offset + past_position] = prob;
                exp_sum += prob;
            }
            if (!(exp_sum > 0.0f) || !std::isfinite(exp_sum)) {
                throw std::runtime_error("attention softmax sum is invalid");
            }

            const size_t ctx_offset =
                (static_cast<size_t>(token_index) * shape_.n_heads + head_index) * shape_.head_dim;
            for (uint32_t past_position = 0; past_position < seq_len; ++past_position) {
                const float prob = attn_probs_[score_offset + past_position] / exp_sum;
                attn_probs_[score_offset + past_position] = prob;

                const size_t v_offset = kv_cache_offset(layer_index, past_position)
                                      + static_cast<size_t>(kv_head_index) * shape_.head_dim;
                for (uint32_t dim_index = 0; dim_index < shape_.head_dim; ++dim_index) {
                    attn_ctx_[ctx_offset + dim_index] += prob * value_cache_[v_offset + dim_index];
                }
            }
        }
    }

    const std::string tensor_name = "blk." + std::to_string(layer_index) + ".attn_output.weight";
    const gguf_tensor_data &output_weight = load_gguf_tensor_data(model, tensor_name);
    gemm_batch(
        output_weight,
        attn_ctx_,
        shape_.batch_size,
        shape_.q_dim(),
        shape_.hidden_size,
        attn_out_
    );
}

void run_state::run_block(const gguf_model &model, uint32_t layer_index, uint32_t position_index) {
    if (layer_index >= shape_.n_layers) {
        throw std::out_of_range("layer index out of range");
    }
    if (hidden_.size() != hidden_elements()) {
        throw std::runtime_error("hidden buffer size mismatch");
    }

    const std::string prefix = "blk." + std::to_string(layer_index);
    constexpr float kRmsNormEps = 1e-6f;

    const std::string attn_norm_weight_name = prefix + ".attn_norm.weight";
    const gguf_tensor_data &attn_norm_weight_tensor =
        load_gguf_tensor_data(model, attn_norm_weight_name);
    const std::vector<float> attn_norm_weight = read_tensor_vector(attn_norm_weight_tensor);
    weighted_rmsnorm_batch(
        hidden_,
        shape_.batch_size,
        shape_.hidden_size,
        attn_norm_weight,
        kRmsNormEps,
        norm_
    );

    compute_qkv(model, layer_index);
    compute_attention(model, layer_index, position_index);

    for (size_t i = 0; i < hidden_.size(); ++i) {
        hidden_[i] += attn_out_[i];
    }

    const std::string ffn_norm_weight_name = prefix + ".ffn_norm.weight";
    const gguf_tensor_data &ffn_norm_weight_tensor =
        load_gguf_tensor_data(model, ffn_norm_weight_name);
    const std::vector<float> ffn_norm_weight = read_tensor_vector(ffn_norm_weight_tensor);
    weighted_rmsnorm_batch(
        hidden_,
        shape_.batch_size,
        shape_.hidden_size,
        ffn_norm_weight,
        kRmsNormEps,
        norm_
    );

    const std::string gate_weight_name = prefix + ".ffn_gate.weight";
    const std::string up_weight_name = prefix + ".ffn_up.weight";
    const std::string down_weight_name = prefix + ".ffn_down.weight";
    const gguf_tensor_data &gate_weight = load_gguf_tensor_data(model, gate_weight_name);
    const gguf_tensor_data &up_weight = load_gguf_tensor_data(model, up_weight_name);
    const gguf_tensor_data &down_weight = load_gguf_tensor_data(model, down_weight_name);

    std::vector<float> gate_proj;
    std::vector<float> up_proj;
    gemm_batch(
        gate_weight,
        norm_,
        shape_.batch_size,
        shape_.hidden_size,
        shape_.ffn_hidden_size,
        gate_proj
    );
    gemm_batch(
        up_weight,
        norm_,
        shape_.batch_size,
        shape_.hidden_size,
        shape_.ffn_hidden_size,
        up_proj
    );

    std::vector<float> ffn_hidden(gate_proj.size(), 0.0f);
    for (size_t i = 0; i < ffn_hidden.size(); ++i) {
        ffn_hidden[i] = silu(gate_proj[i]) * up_proj[i];
    }

    std::vector<float> ffn_out;
    gemm_batch(
        down_weight,
        ffn_hidden,
        shape_.batch_size,
        shape_.ffn_hidden_size,
        shape_.hidden_size,
        ffn_out
    );

    for (size_t i = 0; i < hidden_.size(); ++i) {
        hidden_[i] += ffn_out[i];
    }
}

void run_state::apply_final_norm(const gguf_model &model) {
    if (hidden_.size() != hidden_elements()) {
        throw std::runtime_error("hidden buffer size mismatch");
    }

    constexpr float kRmsNormEps = 1e-6f;
    const std::string output_norm_weight_name = "output_norm.weight";
    const gguf_tensor_data &output_norm_weight_tensor =
        load_gguf_tensor_data(model, output_norm_weight_name);
    const std::vector<float> output_norm_weight = read_tensor_vector(output_norm_weight_tensor);
    weighted_rmsnorm_batch(
        hidden_,
        shape_.batch_size,
        shape_.hidden_size,
        output_norm_weight,
        kRmsNormEps,
        norm_
    );
}

void run_state::compute_logits(const gguf_model &model) {
    if (norm_.size() != hidden_elements()) {
        throw std::runtime_error("final norm buffer size mismatch");
    }

    const std::string output_weight_name = "output.weight";
    const gguf_tensor_data &output_weight = load_gguf_tensor_data(model, output_weight_name);
    if (output_weight.info.dimensions.size() != 2) {
        throw std::runtime_error("output weight must be 2D");
    }
    if (output_weight.info.dimensions[0] != shape_.hidden_size) {
        throw std::runtime_error("output weight hidden dimension mismatch");
    }
    if (output_weight.info.dimensions[1] > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("vocab size does not fit in uint32_t");
    }

    const uint32_t vocab_size = static_cast<uint32_t>(output_weight.info.dimensions[1]);
    gemm_batch(
        output_weight,
        norm_,
        shape_.batch_size,
        shape_.hidden_size,
        vocab_size,
        logits_
    );
}
