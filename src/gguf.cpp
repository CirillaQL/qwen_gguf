#include "gguf.h"

#include <fstream>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {

constexpr uint32_t kGgufMagic = 0x46554747U;
constexpr size_t kMetadataArrayPreviewCount = 5;

uint64_t align_offset(uint64_t offset, uint32_t alignment) {
    const uint64_t mask = static_cast<uint64_t>(alignment) - 1;
    return (offset + mask) & ~mask;
}

template <typename T>
T read_pod(std::ifstream &input, const char *type_name) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error(std::string("failed to read ") + type_name + " from GGUF file");
    }
    return value;
}

uint8_t read_u8(std::ifstream &input) {
    return read_pod<uint8_t>(input, "uint8_t");
}

int8_t read_i8(std::ifstream &input) {
    return read_pod<int8_t>(input, "int8_t");
}

uint16_t read_u16(std::ifstream &input) {
    return read_pod<uint16_t>(input, "uint16_t");
}

int16_t read_i16(std::ifstream &input) {
    return read_pod<int16_t>(input, "int16_t");
}

uint32_t read_u32(std::ifstream &input) {
    return read_pod<uint32_t>(input, "uint32_t");
}

int32_t read_i32(std::ifstream &input) {
    return read_pod<int32_t>(input, "int32_t");
}

uint64_t read_u64(std::ifstream &input) {
    return read_pod<uint64_t>(input, "uint64_t");
}

int64_t read_i64(std::ifstream &input) {
    return read_pod<int64_t>(input, "int64_t");
}

float read_f32(std::ifstream &input) {
    return read_pod<float>(input, "float");
}

double read_f64(std::ifstream &input) {
    return read_pod<double>(input, "double");
}

size_t checked_size(uint64_t len, const char *context) {
    if (len > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(std::string(context) + " is too large for this platform");
    }
    return static_cast<size_t>(len);
}

gguf_type read_gguf_type(std::ifstream &input) {
    const uint32_t raw_type = read_u32(input);
    if (raw_type >= static_cast<uint32_t>(GGUF_TYPE_COUNT)) {
        throw std::runtime_error("invalid gguf metadata type: " + std::to_string(raw_type));
    }
    return static_cast<gguf_type>(raw_type);
}

ggml_type read_ggml_type(std::ifstream &input) {
    const uint32_t raw_type = read_u32(input);
    switch (raw_type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return static_cast<ggml_type>(raw_type);
        default:
            throw std::runtime_error("unsupported ggml tensor type: " + std::to_string(raw_type));
    }
}

uint32_t metadata_u32_or_default(const gguf_metadata &meta, const std::string &key, uint32_t default_value) {
    const auto it = meta.kvs_map.find(key);
    if (it == meta.kvs_map.end()) {
        return default_value;
    }

    const gguf_metadata_kv &kv = meta.kvs[it->second];
    if (kv.value.type != GGUF_TYPE_UINT32) {
        throw std::runtime_error("metadata key '" + key + "' is not UINT32");
    }

    return std::get<uint32_t>(kv.value.data);
}

void build_metadata_index(gguf_metadata &meta) {
    meta.kvs_map.clear();
    meta.kvs_map.reserve(meta.kvs.size());

    for (size_t i = 0; i < meta.kvs.size(); ++i) {
        const std::string &key = meta.kvs[i].key.data;
        const auto [_, inserted] = meta.kvs_map.emplace(key, i);
        if (!inserted) {
            throw std::runtime_error("duplicate metadata key: " + key);
        }
    }
}

void build_tensor_info_index(gguf_model &model) {
    model.tensor_infos_map.clear();
    model.tensor_infos_map.reserve(model.tensor_infos.size());

    for (size_t i = 0; i < model.tensor_infos.size(); ++i) {
        const std::string &name = model.tensor_infos[i].name.data;
        const auto [_, inserted] = model.tensor_infos_map.emplace(name, i);
        if (!inserted) {
            throw std::runtime_error("duplicate tensor name: " + name);
        }
    }
}

uint64_t tensor_element_count(const gguf_tensor_info &info) {
    if (info.dimensions.empty()) {
        return 0;
    }

    return std::accumulate(
        info.dimensions.begin(),
        info.dimensions.end(),
        uint64_t{1},
        [](uint64_t lhs, uint64_t rhs) { return lhs * rhs; }
    );
}

gguf_metadata_value read_gguf_metadata_value_of_type(std::ifstream &input, gguf_type type);

void print_gguf_metadata_value(std::ostream &output, const gguf_metadata_value &value);

void print_gguf_metadata_array(std::ostream &output, const std::shared_ptr<gguf_metadata_array> &array) {
    if (array == nullptr) {
        output << "[]";
        return;
    }

    output << '[';
    size_t printed = 0;
    for (auto it = array->values.cbegin();
         it != array->values.cend() && printed < kMetadataArrayPreviewCount;
         ++it, ++printed) {
        if (it != array->values.cbegin()) {
            output << ", ";
        }
        print_gguf_metadata_value(output, *it);
    }
    if (array->values.size() > kMetadataArrayPreviewCount) {
        output << ", ...";
    }
    output << ']';
}

std::shared_ptr<gguf_metadata_array> read_gguf_metadata_array(std::ifstream &input) {
    auto array = std::make_shared<gguf_metadata_array>();
    array->element_type = read_gguf_type(input);
    array->len = read_u64(input);

    const size_t count = checked_size(array->len, "metadata array length");
    array->values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        array->values.push_back(read_gguf_metadata_value_of_type(input, array->element_type));
    }

    return array;
}

gguf_metadata_value read_gguf_metadata_value_of_type(std::ifstream &input, gguf_type type) {
    gguf_metadata_value value{};
    value.type = type;

    switch (type) {
        case GGUF_TYPE_UINT8:
            value.data.emplace<uint8_t>(read_u8(input));
            break;
        case GGUF_TYPE_INT8:
            value.data.emplace<int8_t>(read_i8(input));
            break;
        case GGUF_TYPE_UINT16:
            value.data.emplace<uint16_t>(read_u16(input));
            break;
        case GGUF_TYPE_INT16:
            value.data.emplace<int16_t>(read_i16(input));
            break;
        case GGUF_TYPE_UINT32:
            value.data.emplace<uint32_t>(read_u32(input));
            break;
        case GGUF_TYPE_INT32:
            value.data.emplace<int32_t>(read_i32(input));
            break;
        case GGUF_TYPE_FLOAT32:
            value.data.emplace<float>(read_f32(input));
            break;
        case GGUF_TYPE_BOOL:
            value.data.emplace<bool>(read_u8(input) != 0);
            break;
        case GGUF_TYPE_STRING:
            value.data.emplace<gguf_string>(read_gguf_string(input));
            break;
        case GGUF_TYPE_ARRAY:
            value.data.emplace<std::shared_ptr<gguf_metadata_array>>(read_gguf_metadata_array(input));
            break;
        case GGUF_TYPE_UINT64:
            value.data.emplace<uint64_t>(read_u64(input));
            break;
        case GGUF_TYPE_INT64:
            value.data.emplace<int64_t>(read_i64(input));
            break;
        case GGUF_TYPE_FLOAT64:
            value.data.emplace<double>(read_f64(input));
            break;
        case GGUF_TYPE_COUNT:
        default:
            throw std::runtime_error("unsupported gguf metadata type");
    }

    return value;
}

void print_gguf_metadata_value(std::ostream &output, const gguf_metadata_value &value) {
    switch (value.type) {
        case GGUF_TYPE_UINT8:
            output << static_cast<uint32_t>(std::get<uint8_t>(value.data));
            break;
        case GGUF_TYPE_INT8:
            output << static_cast<int32_t>(std::get<int8_t>(value.data));
            break;
        case GGUF_TYPE_UINT16:
            output << std::get<uint16_t>(value.data);
            break;
        case GGUF_TYPE_INT16:
            output << std::get<int16_t>(value.data);
            break;
        case GGUF_TYPE_UINT32:
            output << std::get<uint32_t>(value.data);
            break;
        case GGUF_TYPE_INT32:
            output << std::get<int32_t>(value.data);
            break;
        case GGUF_TYPE_FLOAT32:
            output << std::get<float>(value.data);
            break;
        case GGUF_TYPE_BOOL:
            output << (std::get<bool>(value.data) ? "true" : "false");
            break;
        case GGUF_TYPE_STRING:
            output << std::get<gguf_string>(value.data).data;
            break;
        case GGUF_TYPE_ARRAY:
            print_gguf_metadata_array(output, std::get<std::shared_ptr<gguf_metadata_array>>(value.data));
            break;
        case GGUF_TYPE_UINT64:
            output << std::get<uint64_t>(value.data);
            break;
        case GGUF_TYPE_INT64:
            output << std::get<int64_t>(value.data);
            break;
        case GGUF_TYPE_FLOAT64:
            output << std::get<double>(value.data);
            break;
        case GGUF_TYPE_COUNT:
        default:
            output << "<unknown>";
            break;
    }
}

}  // namespace

gguf_string read_gguf_string(std::ifstream &input) {
    gguf_string value{};
    value.len = read_u64(input);
    value.data.resize(checked_size(value.len, "gguf string length"));

    if (!value.data.empty()) {
        input.read(value.data.data(), static_cast<std::streamsize>(value.data.size()));
        if (!input) {
            throw std::runtime_error("failed to read gguf string payload");
        }
    }

    return value;
}

gguf_metadata_value read_gguf_metadata_value(std::ifstream &input) {
    return read_gguf_metadata_value_of_type(input, read_gguf_type(input));
}

gguf_metadata_kv read_gguf_metadata_kv(std::ifstream &input) {
    gguf_metadata_kv kv{};
    kv.key = read_gguf_string(input);
    kv.value = read_gguf_metadata_value(input);
    return kv;
}

gguf_tensor_info read_gguf_tensor_info(std::ifstream &input) {
    gguf_tensor_info info{};
    info.name = read_gguf_string(input);
    info.n_dimensions = read_u32(input);
    info.dimensions.reserve(info.n_dimensions);

    for (uint32_t i = 0; i < info.n_dimensions; ++i) {
        info.dimensions.push_back(read_u64(input));
    }

    info.type = read_ggml_type(input);
    info.offset = read_u64(input);
    return info;
}

gguf_model load_gguf_model(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open GGUF file: " + path);
    }

    gguf_model model{};
    model.file_path = path;

    model.metadata.header.magic = read_u32(input);
    model.metadata.header.version = read_u32(input);
    model.metadata.header.tensor_count = read_u64(input);
    model.metadata.header.metadata_kv_count = read_u64(input);

    if (model.metadata.header.magic != kGgufMagic) {
        throw std::runtime_error("invalid GGUF magic");
    }

    model.metadata.kvs.reserve(checked_size(model.metadata.header.metadata_kv_count, "metadata kv count"));
    for (uint64_t i = 0; i < model.metadata.header.metadata_kv_count; ++i) {
        model.metadata.kvs.push_back(read_gguf_metadata_kv(input));
    }

    build_metadata_index(model.metadata);

    model.alignment = metadata_u32_or_default(model.metadata, "general.alignment", 32);

    model.tensor_infos.reserve(checked_size(model.metadata.header.tensor_count, "tensor count"));
    for (uint64_t i = 0; i < model.metadata.header.tensor_count; ++i) {
        model.tensor_infos.push_back(read_gguf_tensor_info(input));
    }
    build_tensor_info_index(model);

    const uint64_t info_end = static_cast<uint64_t>(input.tellg());
    model.tensor_data_offset = align_offset(info_end, model.alignment);
    return model;
}

void print_gguf_metadata(const gguf_metadata &meta) {
    const gguf_header &header = meta.header;
    const char magic_text[5] = {
        static_cast<char>(header.magic & 0xFF),
        static_cast<char>((header.magic >> 8) & 0xFF),
        static_cast<char>((header.magic >> 16) & 0xFF),
        static_cast<char>((header.magic >> 24) & 0xFF),
        '\0'
    };

    std::cout << "[header]" << '\n';
    std::cout << "magic: 0x"
              << std::hex << std::uppercase << header.magic
              << std::dec << " (" << magic_text << ")" << '\n';
    std::cout << "version: " << header.version << '\n';
    std::cout << "tensor_count: " << header.tensor_count << '\n';
    std::cout << "metadata_kv_count: " << header.metadata_kv_count << '\n';

    std::cout << "[metadata]" << '\n';
    for (auto it = meta.kvs.cbegin(); it != meta.kvs.cend(); ++it) {
        std::cout << it->key.data << ": ";
        print_gguf_metadata_value(std::cout, it->value);
        std::cout << '\n';
    }
}

void print_gguf_tensor_info(const gguf_tensor_info &info) {
    std::cout << info.name.data << " | dims=[";
    for (size_t i = 0; i < info.dimensions.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << info.dimensions[i];
    }
    std::cout << "]"
              << " | type=" << ggml_type_name(info.type)
              << " | offset=" << info.offset
              << '\n';
}

void print_gguf_tensor_overview(const gguf_model &model, size_t limit) {
    std::cout << "[tensors]" << '\n';
    std::cout << "alignment: " << model.alignment << '\n';
    std::cout << "tensor_data_offset: " << model.tensor_data_offset << '\n';

    const size_t count = std::min(limit, model.tensor_infos.size());
    for (size_t i = 0; i < count; ++i) {
        print_gguf_tensor_info(model.tensor_infos[i]);
    }

    if (model.tensor_infos.size() > count) {
        std::cout << "... (" << (model.tensor_infos.size() - count) << " more tensors)" << '\n';
    }
}

const gguf_tensor_data &load_gguf_tensor_data(const gguf_model &model, const std::string &tensor_name) {
    const auto cache_it = model.tensor_cache.find(tensor_name);
    if (cache_it != model.tensor_cache.end()) {
        return *cache_it->second;
    }

    const auto info_it = model.tensor_infos_map.find(tensor_name);
    if (info_it == model.tensor_infos_map.end()) {
        throw std::runtime_error("tensor not found: " + tensor_name);
    }
    const gguf_tensor_info &info = model.tensor_infos[info_it->second];

    const uint64_t element_count = tensor_element_count(info);
    const size_t type_size = ggml_type_size(info.type);
    const size_t block_size = ggml_blck_size(info.type);
    if (block_size == 0 || element_count % block_size != 0) {
        throw std::runtime_error("invalid tensor block layout for: " + tensor_name);
    }

    const uint64_t byte_count = (element_count / block_size) * type_size;

    std::ifstream input(model.file_path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to reopen GGUF file: " + model.file_path);
    }

    const uint64_t absolute_offset = model.tensor_data_offset + info.offset;
    input.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
    if (!input) {
        throw std::runtime_error("failed to seek tensor data for: " + tensor_name);
    }

    auto tensor = std::make_shared<gguf_tensor_data>();
    tensor->info = info;
    tensor->raw_data.resize(checked_size(byte_count, "tensor byte count"));
    input.read(reinterpret_cast<char *>(tensor->raw_data.data()), static_cast<std::streamsize>(tensor->raw_data.size()));
    if (!input) {
        throw std::runtime_error("failed to read tensor data for: " + tensor_name);
    }

    const auto [inserted_it, _] = model.tensor_cache.emplace(tensor_name, std::move(tensor));
    return *inserted_it->second;
}

void preload_gguf_tensors(const gguf_model &model) {
    for (const gguf_tensor_info &info : model.tensor_infos) {
        load_gguf_tensor_data(model, info.name.data);
    }
}

const char *gguf_type_name(gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
            return "UINT8";
        case GGUF_TYPE_INT8:
            return "INT8";
        case GGUF_TYPE_UINT16:
            return "UINT16";
        case GGUF_TYPE_INT16:
            return "INT16";
        case GGUF_TYPE_UINT32:
            return "UINT32";
        case GGUF_TYPE_INT32:
            return "INT32";
        case GGUF_TYPE_FLOAT32:
            return "FLOAT32";
        case GGUF_TYPE_BOOL:
            return "BOOL";
        case GGUF_TYPE_STRING:
            return "STRING";
        case GGUF_TYPE_ARRAY:
            return "ARRAY";
        case GGUF_TYPE_UINT64:
            return "UINT64";
        case GGUF_TYPE_INT64:
            return "INT64";
        case GGUF_TYPE_FLOAT64:
            return "FLOAT64";
        case GGUF_TYPE_COUNT:
            return "COUNT";
        default:
            return "UNKNOWN";
    }
}

size_t ggml_type_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return 4;
        case GGML_TYPE_F16:
            return 2;
        default:
            throw std::runtime_error("unsupported ggml type size query");
    }
}

size_t ggml_blck_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return 1;
        default:
            throw std::runtime_error("unsupported ggml block size query");
    }
}

const char *ggml_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return "F32";
        case GGML_TYPE_F16:
            return "F16";
        default:
            return "UNKNOWN";
    }
}
