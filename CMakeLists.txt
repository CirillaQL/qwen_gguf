cmake_minimum_required(VERSION 3.20)

project(qwen_gguf
    VERSION 0.1.0
    DESCRIPTION "Basic C++ project scaffold for macOS Apple Silicon"
    LANGUAGES CXX
)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

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

    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        target_compile_options(qwen_gguf PRIVATE
            $<$<CONFIG:Release>:-mcpu=apple-m4>
        )
    endif()
endif()

target_compile_options(qwen_gguf PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
)

if(APPLE)
    target_link_options(qwen_gguf PRIVATE
        $<$<CONFIG:Release>:-Wl,-dead_strip>
    )
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set_target_properties(qwen_gguf PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
endif()
