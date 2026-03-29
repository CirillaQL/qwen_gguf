#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"

class Tokenizer {
private:
    std::string model_type;
    std::string pre_type;

    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t, gguf_string_hash> token_to_id;
    std::unordered_map<std::string, uint32_t, gguf_string_hash> merge_ranks;

    int32_t bos_token_id = -1;
    int32_t eos_token_id = -1;
    bool add_bos_token = false;

    int32_t require_token_id(const std::string &token) const;

public:
    Tokenizer() = default;

    static Tokenizer from_model(const gguf_model &model);

    const std::string &get_model_type() const;
    const std::string &get_pre_type() const;

    int32_t get_bos_token_id() const;
    int32_t get_eos_token_id() const;

    size_t vocab_size() const;
    const std::string &token_piece(int32_t token_id) const;
    std::vector<std::string> normalization(const std::vector<std::string> &pieces) const;
    std::vector<std::string> pre_tokenization(const std::string &text) const;
    std::vector<std::string> model(const std::vector<std::string> &pieces) const;
    std::vector<int32_t> post_tokenization(const std::vector<std::string> &pieces) const;
    std::vector<int32_t> encode(const std::string &text) const;
    std::string decode(const std::vector<int32_t> &token_ids) const;
};
