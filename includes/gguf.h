#pragma once

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <iostream>
#include <unordered_map>

// define gguf type
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,
};

struct gguf_header {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

struct gguf_string {
    uint64_t len;
    std::string data;
};

struct gguf_metadata_array;

using gguf_metadata_payload = std::variant<
    std::monostate,
    uint8_t,
    int8_t,
    uint16_t,
    int16_t,
    uint32_t,
    int32_t,
    float,
    bool,
    gguf_string,
    std::shared_ptr<gguf_metadata_array>,
    uint64_t,
    int64_t,
    double
>;

struct gguf_metadata_value {
    gguf_type type = GGUF_TYPE_COUNT;
    gguf_metadata_payload data{};
};

struct gguf_metadata_array {
    gguf_type element_type = GGUF_TYPE_COUNT;
    uint64_t len = 0;
    std::vector<gguf_metadata_value> values;
};

struct gguf_metadata_kv {
    gguf_string key;
    gguf_metadata_value value;
};

struct gguf_string_hash {
    size_t operator()(const std::string &value) const noexcept {
        // FNV Hash
        constexpr size_t kOffset = 1469598103934665603ULL;
        constexpr size_t kPrime = 1099511628211ULL;

        size_t hash = kOffset;
        for (unsigned char ch : value) {
            hash ^= static_cast<size_t>(ch);
            hash *= kPrime;
        }
        return hash;
    }
};

struct gguf_metadata {
    gguf_header header;
    std::vector<gguf_metadata_kv> kvs;
    std::unordered_map<std::string, size_t, gguf_string_hash> kvs_map;
};

gguf_string read_gguf_string(std::ifstream &input);
gguf_metadata_value read_gguf_metadata_value(std::ifstream &input);
gguf_metadata_kv read_gguf_metadata_kv(std::ifstream &input);
void print_gguf_metadata(const gguf_metadata &meta);
const char * gguf_type_name(gguf_type type);

enum ggml_type : uint32_t {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
};

struct gguf_tensor_info {
    gguf_string name;
    uint32_t n_dimensions = 0;
    std::vector<uint64_t> dimensions;
    ggml_type type = GGML_TYPE_F32;
    uint64_t offset = 0;
};

struct gguf_tensor_data;

struct gguf_model {
    gguf_metadata metadata;
    std::vector<gguf_tensor_info> tensor_infos;
    std::unordered_map<std::string, size_t, gguf_string_hash> tensor_infos_map;
    uint64_t tensor_data_offset = 0;
    uint32_t alignment = 32;
    std::string file_path;
    mutable std::unordered_map<std::string, std::shared_ptr<gguf_tensor_data>, gguf_string_hash> tensor_cache;
};

struct gguf_tensor_data {
    gguf_tensor_info info;
    std::vector<uint8_t> raw_data;
};

gguf_tensor_info read_gguf_tensor_info(std::ifstream &input);
gguf_model load_gguf_model(const std::string &path);
void print_gguf_tensor_info(const gguf_tensor_info &info);
void print_gguf_tensor_overview(const gguf_model &model, size_t limit = 8);
const gguf_tensor_data &load_gguf_tensor_data(const gguf_model &model, const std::string &tensor_name);
void preload_gguf_tensors(const gguf_model &model);
size_t ggml_type_size(ggml_type type);
size_t ggml_blck_size(ggml_type type);
const char *ggml_type_name(ggml_type type);
