#include "gguf.h"

#include <exception>
#include <iostream>

int main() {
    const std::string path = "qwen2.5-1.5b-instruct-fp16.gguf";
    // const std::string path = "Qwen3.5-4B-BF16.gguf";

    try {
        const gguf_model model = load_gguf_model(path);
        print_gguf_metadata(model.metadata);
        std::cout << '\n';
        print_gguf_tensor_overview(model, 12);

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
