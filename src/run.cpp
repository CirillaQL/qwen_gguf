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
    const gguf_tensor_data tensor = load_gguf_tensor_data(model, tensor_name);
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
