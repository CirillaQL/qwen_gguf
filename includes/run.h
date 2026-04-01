#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf.h"

struct config {
    uint32_t max_new_tokens = 4096;
    float temperature = 1.0f;
    uint32_t top_k = 40;
    float top_p = 0.95f;
    float repeat_penalty = 1.0f;
    uint32_t repeat_last_n = 64;
    int64_t seed = -1;
    uint32_t eos_token_id = 0;

    bool do_sample = true;
};


struct embedding_batch {
    uint32_t token_count = 0;
    uint32_t hidden_size = 0;
    std::vector<float> values;

    float at(uint32_t token_index, uint32_t hidden_index) const;
    std::vector<float> token_embedding(uint32_t token_index) const;
};

embedding_batch lookup_embeddings(
    const gguf_model &model,
    const std::vector<int32_t> &token_ids,
    const std::string &tensor_name = "token_embd.weight"
);

embedding_batch RMSNorm(const embedding_batch &token_embedding, float eps = 1e-6f);
