#pragma once

#include <cstdint>

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