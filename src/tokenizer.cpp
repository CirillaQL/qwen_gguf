#include "tokenizer.h"

#include <array>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

const gguf_metadata_value *find_metadata_value(const gguf_metadata &metadata, const std::string &key) {
    const auto it = metadata.kvs_map.find(key);
    if (it == metadata.kvs_map.end()) {
        return nullptr;
    }

    return &metadata.kvs[it->second].value;
}

const gguf_metadata_value &require_metadata_value(const gguf_metadata &metadata, const std::string &key) {
    const gguf_metadata_value *value = find_metadata_value(metadata, key);
    if (value == nullptr) {
        throw std::runtime_error("required metadata key not found: " + key);
    }

    return *value;
}

const gguf_metadata_array &require_metadata_array(
    const gguf_metadata &metadata,
    const std::string &key,
    gguf_type expected_element_type
) {
    const gguf_metadata_value &value = require_metadata_value(metadata, key);
    if (value.type != GGUF_TYPE_ARRAY) {
        throw std::runtime_error("metadata key '" + key + "' is not ARRAY");
    }

    const auto &array_ptr = std::get<std::shared_ptr<gguf_metadata_array>>(value.data);
    if (array_ptr == nullptr) {
        throw std::runtime_error("metadata key '" + key + "' contains a null ARRAY payload");
    }

    if (array_ptr->element_type != expected_element_type) {
        throw std::runtime_error(
            "metadata key '" + key + "' has unexpected ARRAY element type: " +
            std::string(gguf_type_name(array_ptr->element_type))
        );
    }

    return *array_ptr;
}

std::string require_string_metadata(const gguf_metadata &metadata, const std::string &key) {
    const gguf_metadata_value &value = require_metadata_value(metadata, key);
    if (value.type != GGUF_TYPE_STRING) {
        throw std::runtime_error("metadata key '" + key + "' is not STRING");
    }

    return std::get<gguf_string>(value.data).data;
}

int32_t optional_token_id_metadata(const gguf_metadata &metadata, const std::string &key) {
    const gguf_metadata_value *value = find_metadata_value(metadata, key);
    if (value == nullptr) {
        return -1;
    }

    if (value->type != GGUF_TYPE_UINT32) {
        throw std::runtime_error("metadata key '" + key + "' is not UINT32");
    }

    return static_cast<int32_t>(std::get<uint32_t>(value->data));
}

bool optional_bool_metadata(const gguf_metadata &metadata, const std::string &key, bool default_value) {
    const gguf_metadata_value *value = find_metadata_value(metadata, key);
    if (value == nullptr) {
        return default_value;
    }

    if (value->type != GGUF_TYPE_BOOL) {
        throw std::runtime_error("metadata key '" + key + "' is not BOOL");
    }

    return std::get<bool>(value->data);
}

std::vector<std::string> read_string_array(const gguf_metadata &metadata, const std::string &key) {
    const gguf_metadata_array &array = require_metadata_array(metadata, key, GGUF_TYPE_STRING);
    std::vector<std::string> values;
    values.reserve(array.values.size());

    for (const gguf_metadata_value &value : array.values) {
        values.push_back(std::get<gguf_string>(value.data).data);
    }

    return values;
}

void validate_special_token_id(const std::string &name, int32_t token_id, size_t vocab_size) {
    if (token_id < 0) {
        return;
    }

    if (static_cast<size_t>(token_id) >= vocab_size) {
        throw std::runtime_error(name + " is out of vocabulary range");
    }
}

void append_utf8(std::string &output, uint32_t codepoint) {
    if (codepoint <= 0x7F) {
        output.push_back(static_cast<char>(codepoint));
        return;
    }

    if (codepoint <= 0x7FF) {
        output.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        return;
    }

    if (codepoint <= 0xFFFF) {
        output.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        return;
    }

    output.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
}

bool should_preserve_bpe_byte(size_t byte_value) {
    return ('!' <= byte_value && byte_value <= '~') || (0xA1 <= byte_value && byte_value <= 0xAC) ||
           (0xAE <= byte_value && byte_value <= 0xFF);
}

std::array<std::string, 256> build_bpe_byte_encoder() {
    std::array<std::string, 256> encoder{};
    uint32_t extra_codepoint = 256;

    for (size_t byte_value = 0; byte_value < encoder.size(); ++byte_value) {
        const uint32_t codepoint =
            should_preserve_bpe_byte(byte_value) ? static_cast<uint32_t>(byte_value) : extra_codepoint++;
        append_utf8(encoder[byte_value], codepoint);
    }

    return encoder;
}

std::unordered_map<std::string, unsigned char, gguf_string_hash> build_bpe_byte_decoder() {
    const std::array<std::string, 256> encoder = build_bpe_byte_encoder();
    std::unordered_map<std::string, unsigned char, gguf_string_hash> decoder;
    decoder.reserve(encoder.size());

    for (size_t byte_value = 0; byte_value < encoder.size(); ++byte_value) {
        decoder.emplace(encoder[byte_value], static_cast<unsigned char>(byte_value));
    }

    return decoder;
}

std::string normalize_bpe_piece(const std::string &input) {
    static const std::array<std::string, 256> kByteEncoder = build_bpe_byte_encoder();

    std::string normalized;
    normalized.reserve(input.size() * 2);

    for (const char ch : input) {
        normalized += kByteEncoder[static_cast<unsigned char>(ch)];
    }

    return normalized;
}

bool is_ascii_space(unsigned char ch) {
    switch (ch) {
        case ' ':
        case '\t':
        case '\n':
        case '\r':
        case '\v':
        case '\f':
            return true;
        default:
            return false;
    }
}

bool is_ascii_alpha(unsigned char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
}

bool is_ascii_digit(unsigned char ch) {
    return ch >= '0' && ch <= '9';
}

bool is_word_like_byte(unsigned char ch) {
    return is_ascii_alpha(ch) || ch >= 0x80;
}

bool is_number_like_byte(unsigned char ch) {
    return is_ascii_digit(ch);
}

bool starts_with_contraction(const std::string &text, size_t offset, std::string &matched) {
    static const std::array<const char *, 6> kContractions = {"'s", "'t", "'re", "'ve", "'m", "'ll"};

    for (const char *candidate : kContractions) {
        const size_t length = std::strlen(candidate);
        if (offset + length > text.size()) {
            continue;
        }
        if (text.compare(offset, length, candidate) == 0) {
            matched.assign(candidate, length);
            return true;
        }
    }

    if (offset + 2 <= text.size() && text.compare(offset, 2, "'d") == 0) {
        matched = "'d";
        return true;
    }

    return false;
}

std::vector<std::string> split_utf8_codepoints(const std::string &text) {
    std::vector<std::string> pieces;
    pieces.reserve(text.size());

    for (size_t i = 0; i < text.size();) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        size_t codepoint_length = 1;

        if ((ch & 0x80U) == 0) {
            codepoint_length = 1;
        } else if ((ch & 0xE0U) == 0xC0U) {
            codepoint_length = 2;
        } else if ((ch & 0xF0U) == 0xE0U) {
            codepoint_length = 3;
        } else if ((ch & 0xF8U) == 0xF0U) {
            codepoint_length = 4;
        } else {
            throw std::runtime_error("invalid UTF-8 leading byte in BPE piece");
        }

        if (i + codepoint_length > text.size()) {
            throw std::runtime_error("truncated UTF-8 sequence in BPE piece");
        }

        for (size_t j = 1; j < codepoint_length; ++j) {
            const unsigned char continuation = static_cast<unsigned char>(text[i + j]);
            if ((continuation & 0xC0U) != 0x80U) {
                throw std::runtime_error("invalid UTF-8 continuation byte in BPE piece");
            }
        }

        pieces.push_back(text.substr(i, codepoint_length));
        i += codepoint_length;
    }

    return pieces;
}

std::string merge_key(const std::string &left, const std::string &right) {
    return left + " " + right;
}

}  // namespace

Tokenizer Tokenizer::from_model(const gguf_model &model) {
    const gguf_metadata &metadata = model.metadata;
    Tokenizer tokenizer{};
    tokenizer.model_type = require_string_metadata(metadata, "tokenizer.ggml.model");
    tokenizer.pre_type = require_string_metadata(metadata, "tokenizer.ggml.pre");
    tokenizer.id_to_token = read_string_array(metadata, "tokenizer.ggml.tokens");
    const std::vector<std::string> merges = read_string_array(metadata, "tokenizer.ggml.merges");

    tokenizer.token_to_id.reserve(tokenizer.id_to_token.size());
    for (size_t i = 0; i < tokenizer.id_to_token.size(); ++i) {
        const auto [_, inserted] = tokenizer.token_to_id.emplace(tokenizer.id_to_token[i], static_cast<int32_t>(i));
        if (!inserted) {
            throw std::runtime_error("duplicate tokenizer token piece: " + tokenizer.id_to_token[i]);
        }
    }

    tokenizer.merge_ranks.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) {
        const auto [_, inserted] = tokenizer.merge_ranks.emplace(merges[i], static_cast<uint32_t>(i));
        if (!inserted) {
            throw std::runtime_error("duplicate tokenizer merge rule: " + merges[i]);
        }
    }

    tokenizer.bos_token_id = optional_token_id_metadata(metadata, "tokenizer.ggml.bos_token_id");
    tokenizer.eos_token_id = optional_token_id_metadata(metadata, "tokenizer.ggml.eos_token_id");
    tokenizer.add_bos_token = optional_bool_metadata(metadata, "tokenizer.ggml.add_bos_token", false);

    validate_special_token_id("tokenizer.ggml.bos_token_id", tokenizer.bos_token_id, tokenizer.id_to_token.size());
    validate_special_token_id("tokenizer.ggml.eos_token_id", tokenizer.eos_token_id, tokenizer.id_to_token.size());

    return tokenizer;
}

const std::string &Tokenizer::get_model_type() const {
    return model_type;
}

const std::string &Tokenizer::get_pre_type() const {
    return pre_type;
}

int32_t Tokenizer::get_bos_token_id() const {
    return bos_token_id;
}

int32_t Tokenizer::get_eos_token_id() const {
    return eos_token_id;
}

size_t Tokenizer::vocab_size() const {
    return id_to_token.size();
}

int32_t Tokenizer::require_token_id(const std::string &token) const {
    const auto it = token_to_id.find(token);
    if (it == token_to_id.end()) {
        throw std::runtime_error("token piece not found in vocabulary: " + token);
    }

    return it->second;
}

const std::string &Tokenizer::token_piece(int32_t token_id_value) const {
    if (token_id_value < 0 || static_cast<size_t>(token_id_value) >= id_to_token.size()) {
        throw std::runtime_error("token id out of range: " + std::to_string(token_id_value));
    }

    return id_to_token[static_cast<size_t>(token_id_value)];
}

std::vector<std::string> Tokenizer::normalization(const std::vector<std::string> &pieces) const {
    std::vector<std::string> normalized_pieces;
    normalized_pieces.reserve(pieces.size());

    for (const std::string &piece : pieces) {
        normalized_pieces.push_back(normalize_bpe_piece(piece));
    }

    return normalized_pieces;
}

std::vector<std::string> Tokenizer::pre_tokenization(const std::string &text) const {
    std::vector<std::string> pieces;
    size_t i = 0;

    while (i < text.size()) {
        size_t leading_space_begin = i;
        while (i < text.size() && text[i] == ' ') {
            ++i;
        }
        const bool has_leading_space = i > leading_space_begin;
        const size_t leading_space_count = i - leading_space_begin;

        if (i >= text.size()) {
            if (leading_space_count > 0) {
                pieces.push_back(text.substr(leading_space_begin, leading_space_count));
            }
            break;
        }

        const unsigned char ch = static_cast<unsigned char>(text[i]);

        if (is_ascii_space(ch) && ch != ' ') {
            size_t begin = i;
            while (i < text.size() && is_ascii_space(static_cast<unsigned char>(text[i])) && text[i] != ' ') {
                ++i;
            }
            pieces.push_back(text.substr(begin, i - begin));
            continue;
        }

        std::string contraction;
        if (starts_with_contraction(text, i, contraction)) {
            if (has_leading_space) {
                contraction.insert(0, leading_space_count, ' ');
            }
            pieces.push_back(contraction);
            i += contraction[0] == ' ' ? contraction.size() - leading_space_count : contraction.size();
            continue;
        }

        if (is_word_like_byte(ch)) {
            const size_t begin = i;
            while (i < text.size() && is_word_like_byte(static_cast<unsigned char>(text[i]))) {
                ++i;
            }
            std::string piece = text.substr(begin, i - begin);
            if (has_leading_space) {
                piece.insert(0, leading_space_count, ' ');
            }
            pieces.push_back(std::move(piece));
            continue;
        }

        if (is_number_like_byte(ch)) {
            const size_t begin = i;
            while (i < text.size() && is_number_like_byte(static_cast<unsigned char>(text[i]))) {
                ++i;
            }
            std::string piece = text.substr(begin, i - begin);
            if (has_leading_space) {
                piece.insert(0, leading_space_count, ' ');
            }
            pieces.push_back(std::move(piece));
            continue;
        }

        size_t begin = i;
        while (i < text.size()) {
            const unsigned char current = static_cast<unsigned char>(text[i]);
            if (current == ' ' || is_ascii_space(current) || is_word_like_byte(current) || is_number_like_byte(current)) {
                break;
            }
            std::string ignored;
            if (starts_with_contraction(text, i, ignored)) {
                break;
            }
            ++i;
        }

        std::string piece = text.substr(begin, i - begin);
        if (has_leading_space) {
            piece.insert(0, leading_space_count, ' ');
        }
        pieces.push_back(std::move(piece));
    }

    return pieces;
}

std::vector<std::string> Tokenizer::model(const std::vector<std::string> &pieces) const {
    std::vector<std::string> model_pieces;
    for (const std::string &piece : pieces) {
        if (piece.empty()) {
            continue;
        }

        std::vector<std::string> symbols = split_utf8_codepoints(piece);
        if (symbols.size() < 2) {
            model_pieces.insert(model_pieces.end(), symbols.begin(), symbols.end());
            continue;
        }

        while (symbols.size() > 1) {
            size_t best_pair_index = symbols.size();
            uint32_t best_rank = std::numeric_limits<uint32_t>::max();

            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                const auto it = merge_ranks.find(merge_key(symbols[i], symbols[i + 1]));
                if (it != merge_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pair_index = i;
                }
            }

            if (best_pair_index == symbols.size()) {
                break;
            }

            symbols[best_pair_index] += symbols[best_pair_index + 1];
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_pair_index + 1));
        }

        model_pieces.insert(model_pieces.end(), symbols.begin(), symbols.end());
    }

    return model_pieces;
}

std::vector<int32_t> Tokenizer::post_tokenization(const std::vector<std::string> &pieces) const {
    std::vector<int32_t> token_ids;
    if (add_bos_token && bos_token_id >= 0) {
        token_ids.push_back(bos_token_id);
    }

    for (const std::string &piece : pieces) {
        token_ids.push_back(require_token_id(piece));
    }

    return token_ids;
}

std::vector<int32_t> Tokenizer::encode(const std::string &text) const {
    const std::vector<std::string> pre_tokenized = pre_tokenization(text);
    const std::vector<std::string> normalized = normalization(pre_tokenized);
    const std::vector<std::string> model_pieces = model(normalized);
    return post_tokenization(model_pieces);
}

std::string Tokenizer::decode(const std::vector<int32_t> &token_ids) const {
    static const std::unordered_map<std::string, unsigned char, gguf_string_hash> kByteDecoder =
        build_bpe_byte_decoder();

    std::string output;

    for (int32_t token_id_value : token_ids) {
        const std::string &piece = token_piece(token_id_value);
        const std::vector<std::string> symbols = split_utf8_codepoints(piece);
        for (const std::string &symbol : symbols) {
            const auto it = kByteDecoder.find(symbol);
            if (it == kByteDecoder.end()) {
                throw std::runtime_error("failed to decode token piece: " + piece);
            }
            output.push_back(static_cast<char>(it->second));
        }
    }

    return output;
}
