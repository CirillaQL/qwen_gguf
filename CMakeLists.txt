cmake_minimum_required(VERSION 3.20)

project(qwen_gguf
    VERSION 0.1.0
    DESCRIPTION "Basic C++ project scaffold for macOS Apple Silicon"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
    # Apple Silicon native build by default.
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for macOS" FORCE)
endif()

add_executable(qwen_gguf
    src/gguf.cpp
    src/run.cpp
    src/tokenizer.cpp
    src/main.cpp
)

target_compile_features(qwen_gguf PRIVATE cxx_std_17)

target_include_directories(qwen_gguf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(qwen_gguf PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()
