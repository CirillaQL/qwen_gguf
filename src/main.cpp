#include "gguf.h"
#include "run.h"
#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

uint32_t metadata_u32(const gguf_model &model, const std::string &key) {
    const auto it = model.metadata.kvs_map.find(key);
    if (it == model.metadata.kvs_map.end()) {
        throw std::runtime_error("missing metadata key: " + key);
    }

    const gguf_metadata_kv &kv = model.metadata.kvs[it->second];
    if (kv.value.type != GGUF_TYPE_UINT32) {
        throw std::runtime_error("metadata key is not UINT32: " + key);
    }

    return std::get<uint32_t>(kv.value.data);
}

void forward_token(
    const gguf_model &model,
    run_state &state,
    int32_t token_id,
    uint32_t position_index
) {
    const embedding_batch embedding = lookup_embeddings(model, {token_id});
    state.hidden_ = embedding.values;

    for (uint32_t layer_index = 0; layer_index < state.shape().n_layers; ++layer_index) {
        state.run_block(model, layer_index, position_index);
    }

    state.apply_final_norm(model);
    state.compute_logits(model);
    state.set_current_seq_len(position_index + 1U);
}

int32_t sample_next_token(
    const std::vector<float> &logits,
    const std::vector<int32_t> &history,
    const config &cfg,
    std::mt19937 &rng
) {
    if (logits.empty()) {
        throw std::runtime_error("cannot sample from empty logits");
    }

    std::vector<float> adjusted = logits;

    if (cfg.repeat_penalty != 1.0f && !history.empty()) {
        const size_t repeat_window = std::min(history.size(), static_cast<size_t>(cfg.repeat_last_n));
        const size_t repeat_begin = history.size() - repeat_window;
        for (size_t i = repeat_begin; i < history.size(); ++i) {
            const int32_t token_id = history[i];
            if (token_id < 0 || static_cast<size_t>(token_id) >= adjusted.size()) {
                continue;
            }

            float &logit = adjusted[static_cast<size_t>(token_id)];
            if (logit < 0.0f) {
                logit *= cfg.repeat_penalty;
            } else {
                logit /= cfg.repeat_penalty;
            }
        }
    }

    if (!cfg.do_sample || cfg.temperature <= 0.0f) {
        return static_cast<int32_t>(
            std::distance(adjusted.begin(), std::max_element(adjusted.begin(), adjusted.end()))
        );
    }

    for (float &logit : adjusted) {
        logit /= cfg.temperature;
    }

    std::vector<int32_t> candidate_ids(adjusted.size());
    std::iota(candidate_ids.begin(), candidate_ids.end(), 0);

    if (cfg.top_k > 0 && static_cast<size_t>(cfg.top_k) < candidate_ids.size()) {
        std::partial_sort(
            candidate_ids.begin(),
            candidate_ids.begin() + cfg.top_k,
            candidate_ids.end(),
            [&](int32_t lhs, int32_t rhs) { return adjusted[lhs] > adjusted[rhs]; }
        );
        candidate_ids.resize(cfg.top_k);
    }

    std::sort(
        candidate_ids.begin(),
        candidate_ids.end(),
        [&](int32_t lhs, int32_t rhs) { return adjusted[lhs] > adjusted[rhs]; }
    );

    float max_logit = adjusted[candidate_ids.front()];
    std::vector<float> probs;
    probs.reserve(candidate_ids.size());
    float prob_sum = 0.0f;
    for (int32_t token_id : candidate_ids) {
        const float prob = std::exp(adjusted[token_id] - max_logit);
        probs.push_back(prob);
        prob_sum += prob;
    }

    for (float &prob : probs) {
        prob /= prob_sum;
    }

    if (cfg.top_p < 1.0f) {
        float cumulative = 0.0f;
        size_t keep_count = 0;
        for (; keep_count < probs.size(); ++keep_count) {
            cumulative += probs[keep_count];
            if (cumulative >= cfg.top_p) {
                ++keep_count;
                break;
            }
        }

        keep_count = std::max<size_t>(1, std::min(keep_count, probs.size()));
        candidate_ids.resize(keep_count);
        probs.resize(keep_count);

        const float kept_sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (float &prob : probs) {
            prob /= kept_sum;
        }
    }

    std::discrete_distribution<size_t> distribution(probs.begin(), probs.end());
    return candidate_ids[distribution(rng)];
}

}  // namespace

int main() {
    const std::string path = "qwen2.5-1.5b-instruct-fp16.gguf";
    const std::string prompt = "你是谁？请用一句话回答。";

    try {
        const gguf_model model = load_gguf_model(path);
        const Tokenizer tokenizer = Tokenizer::from_model(model);
        std::cout << "[cache]" << '\n';
        std::cout << "preloading tensors into memory..." << '\n';
        preload_gguf_tensors(model);
        std::cout << "tensor_count: " << model.tensor_cache.size() << '\n';

        run_state_shape shape{};
        shape.batch_size = 1;
        shape.max_seq_len = metadata_u32(model, "qwen2.context_length");
        shape.hidden_size = metadata_u32(model, "qwen2.embedding_length");
        shape.n_layers = metadata_u32(model, "qwen2.block_count");
        shape.n_heads = metadata_u32(model, "qwen2.attention.head_count");
        shape.n_kv_heads = metadata_u32(model, "qwen2.attention.head_count_kv");
        shape.head_dim = shape.hidden_size / shape.n_heads;
        shape.ffn_hidden_size = metadata_u32(model, "qwen2.feed_forward_length");

        config cfg{};
        cfg.max_new_tokens = 8;
        cfg.temperature = 0.8f;
        cfg.top_k = 40;
        cfg.top_p = 0.95f;
        cfg.repeat_penalty = 1.1f;
        cfg.repeat_last_n = 64;
        cfg.seed = 42;
        cfg.eos_token_id = static_cast<uint32_t>(std::max(0, tokenizer.get_eos_token_id()));
        cfg.do_sample = true;

        const std::vector<int32_t> prompt_token_ids = tokenizer.encode(prompt);
        if (prompt_token_ids.empty()) {
            throw std::runtime_error("prompt produced no tokens");
        }
        if (prompt_token_ids.size() >= shape.max_seq_len) {
            throw std::runtime_error("prompt is longer than model context length");
        }

        run_state state(shape);
        state.reset_sequence();

        std::vector<int32_t> full_token_ids = prompt_token_ids;
        for (uint32_t position_index = 0; position_index < prompt_token_ids.size(); ++position_index) {
            forward_token(model, state, prompt_token_ids[position_index], position_index);
        }

        std::mt19937 rng(static_cast<uint32_t>(cfg.seed < 0 ? std::random_device{}() : cfg.seed));

        std::cout << "[prompt]" << '\n' << prompt << '\n';
        std::cout << "[generated]" << '\n';
        std::cout.flush();

        for (uint32_t step = 0; step < cfg.max_new_tokens; ++step) {
            const int32_t next_token = sample_next_token(state.logits_, full_token_ids, cfg, rng);
            if (tokenizer.get_eos_token_id() >= 0 && next_token == tokenizer.get_eos_token_id()) {
                break;
            }

            full_token_ids.push_back(next_token);
            std::cout << tokenizer.decode({next_token});
            std::cout.flush();

            if (full_token_ids.size() >= shape.max_seq_len) {
                break;
            }

            forward_token(
                model,
                state,
                next_token,
                static_cast<uint32_t>(full_token_ids.size() - 1)
            );
        }
        std::cout << '\n';
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
