#pragma once

#include <algorithm>
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

struct run_state_shape {
    uint32_t batch_size = 1;
    uint32_t max_seq_len = 0;
    uint32_t hidden_size = 0;
    uint32_t n_layers = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t ffn_hidden_size = 0;

    uint32_t q_dim() const {
        return n_heads * head_dim;
    }

    uint32_t kv_dim() const {
        return n_kv_heads * head_dim;
    }
};

class run_state {
public:
    run_state() = default;

    explicit run_state(run_state_shape shape)
        : shape_(shape) {
        resize_buffers();
    }

    const run_state_shape &shape() const {
        return shape_;
    }

    void set_shape(run_state_shape shape) {
        shape_ = shape;
        resize_buffers();
        current_seq_len_ = 0;
    }

    uint32_t current_seq_len() const {
        return current_seq_len_;
    }

    void set_current_seq_len(uint32_t seq_len) {
        current_seq_len_ = seq_len;
    }

    void resize_buffers() {
        hidden_.resize(hidden_elements());
        norm_.resize(hidden_elements());

        q_.resize(q_elements());
        k_.resize(kv_elements());
        v_.resize(kv_elements());
        attn_out_.resize(hidden_elements());

        attn_scores_.resize(attn_score_elements());
        attn_probs_.resize(attn_score_elements());
        attn_ctx_.resize(attn_ctx_elements());

        key_cache_.resize(kv_cache_elements());
        value_cache_.resize(kv_cache_elements());
    }

    void reset_sequence() {
        current_seq_len_ = 0;
        std::fill(key_cache_.begin(), key_cache_.end(), 0.0f);
        std::fill(value_cache_.begin(), value_cache_.end(), 0.0f);
    }

    void compute_qkv(const gguf_model &model, uint32_t layer_index);
    void compute_attention(const gguf_model &model, uint32_t layer_index, uint32_t position_index);
    void run_block(const gguf_model &model, uint32_t layer_index, uint32_t position_index);
    void apply_final_norm(const gguf_model &model);
    void compute_logits(const gguf_model &model);

    std::vector<float> hidden_;
    std::vector<float> norm_;
    std::vector<float> logits_;

    std::vector<float> q_;
    std::vector<float> k_;
    std::vector<float> v_;

    std::vector<float> attn_out_;

    std::vector<float> key_cache_;
    std::vector<float> value_cache_;

private:
    void apply_rope(uint32_t position_index);
    void update_kv_cache(uint32_t layer_index, uint32_t position_index);

    size_t hidden_elements() const {
        return static_cast<size_t>(shape_.batch_size) * shape_.hidden_size;
    }

    size_t q_elements() const {
        return static_cast<size_t>(shape_.batch_size) * shape_.q_dim();
    }

    size_t kv_elements() const {
        return static_cast<size_t>(shape_.batch_size) * shape_.kv_dim();
    }

    size_t attn_score_elements() const {
        return static_cast<size_t>(shape_.batch_size) * shape_.n_heads * shape_.max_seq_len;
    }

    size_t attn_ctx_elements() const {
        return q_elements();
    }

    size_t kv_cache_elements() const {
        return static_cast<size_t>(shape_.n_layers)
             * static_cast<size_t>(shape_.max_seq_len)
             * shape_.kv_dim();
    }

    size_t kv_cache_offset(uint32_t layer_index, uint32_t position_index) const {
        return (static_cast<size_t>(layer_index) * shape_.max_seq_len + position_index) * shape_.kv_dim();
    }

    run_state_shape shape_{};

    uint32_t current_seq_len_ = 0;

    std::vector<float> attn_scores_;
    std::vector<float> attn_probs_;
    std::vector<float> attn_ctx_;
};

embedding_batch lookup_embeddings(
    const gguf_model &model,
    const std::vector<int32_t> &token_ids,
    const std::string &tensor_name = "token_embd.weight"
);

embedding_batch RMSNorm(const embedding_batch &token_embedding, float eps = 1e-6f);
