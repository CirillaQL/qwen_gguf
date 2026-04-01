#include "gguf.h"
#include "run.h"
#include "tokenizer.h"

#include <algorithm>
#include <exception>
#include <iostream>

int main() {
    const std::string path = "qwen2.5-1.5b-instruct-fp16.gguf";
    // const std::string path = "Qwen3.5-4B-BF16.gguf";

    try {
        const gguf_model model = load_gguf_model(path);
        const Tokenizer tokenizer = Tokenizer::from_model(model);
        print_gguf_metadata(model.metadata);
        std::cout << '\n';
        print_gguf_tensor_overview(model, 12);
        std::cout << '\n';
        std::cout << "[tokenizer]" << '\n';
        std::cout << "model_type: " << tokenizer.get_model_type() << '\n';
        std::cout << "pre_type: " << tokenizer.get_pre_type() << '\n';
        std::cout << "vocab_size: " << tokenizer.vocab_size() << '\n';
        std::cout << "bos_token_id: " << tokenizer.get_bos_token_id() << '\n';
        std::cout << "eos_token_id: " << tokenizer.get_eos_token_id() << '\n';

        const std::vector<int32_t> encoded = tokenizer.encode("你好！世界");
        std::cout << '\n';
        std::cout << "[encode]" << '\n';
        std::cout << "input: 你好！世界" << '\n';
        std::cout << "token_ids:";
        for (int32_t token_id : encoded) {
            std::cout << ' ' << token_id;
        }

        const gguf_tensor_data token_embd = load_gguf_tensor_data(model, "token_embd.weight");
        std::cout << '\n';
        std::cout << "[tensor sample]" << '\n';
        std::cout << token_embd.info.name.data
                  << " raw bytes: " << token_embd.raw_data.size()
                  << '\n';

        const embedding_batch embeddings = lookup_embeddings(model, encoded);
        std::cout << '\n';
        std::cout << "[embedding lookup]" << '\n';
        std::cout << "token_count: " << embeddings.token_count << '\n';
        std::cout << "hidden_size: " << embeddings.hidden_size << '\n';
        std::cout << "all: " << embeddings.values.size() << '\n';

        const embedding_batch rmsnorm_result = RMSNorm(embeddings, 1e-6f);
        std::cout << "[rmsnorm]" << '\n';
        std::cout << "token_count: " << rmsnorm_result.token_count << '\n';
        std::cout << "hidden_size: " << rmsnorm_result.hidden_size << '\n';
        std::cout << "first_token:";
        const uint32_t preview = std::min<uint32_t>(8, rmsnorm_result.hidden_size);
        for (uint32_t i = 0; i < preview; ++i) {
            std::cout << ' ' << rmsnorm_result.at(0, i);
        }
        std::cout << '\n';
        std::cout << "all:";
        for (float value : rmsnorm_result.values) {
            std::cout << value << ' ';
        }
        std::cout << '\n';
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
