#include "gguf.h"
#include "tokenizer.h"

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

        const std::vector<int32_t> encoded = tokenizer.encode("No,becauseI need to know why he had to sell it and don’t say for medical debt or imma lose it");
        std::cout << '\n';
        std::cout << "[encode]" << '\n';
        std::cout << "input: Hello world" << '\n';
        std::cout << "token_ids:";
        for (int32_t token_id : encoded) {
            std::cout << ' ' << token_id;
        }
        std::cout << '\n';
        std::cout << "pieces:";
        for (int32_t token_id : encoded) {
            std::cout << ' ' << tokenizer.token_piece(token_id);
        }
        std::cout << '\n';

        const gguf_tensor_data token_embd = load_gguf_tensor_data(model, "token_embd.weight");
        std::cout << '\n';
        std::cout << "[tensor sample]" << '\n';
        std::cout << token_embd.info.name.data
                  << " raw bytes: " << token_embd.raw_data.size()
                  << '\n';
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
