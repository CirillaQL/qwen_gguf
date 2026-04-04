cmake_minimum_required(VERSION 3.20)

project(qwen_gguf
    VERSION 0.1.0
    DESCRIPTION "Basic C++ project scaffold for Qwen GGUF experiments"
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

set(QWEN_GGUF_CPU_FLAGS "" CACHE STRING "Extra CPU-specific compile flags for Release builds")
option(QWEN_GGUF_TUNE_NATIVE "Enable native CPU tuning for Release builds" ON)
option(QWEN_GGUF_ENABLE_OPENMP "Enable OpenMP support" ON)

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

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(qwen_gguf PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

set(QWEN_GGUF_RELEASE_CPU_FLAGS "")

if(QWEN_GGUF_CPU_FLAGS)
    separate_arguments(QWEN_GGUF_RELEASE_CPU_FLAGS NATIVE_COMMAND "${QWEN_GGUF_CPU_FLAGS}")
elseif(QWEN_GGUF_TUNE_NATIVE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        set(QWEN_GGUF_RELEASE_CPU_FLAGS
            -march=native
            -mtune=native
        )
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            set(QWEN_GGUF_RELEASE_CPU_FLAGS
                -mcpu=native
            )
        endif()
    endif()
endif()

target_compile_options(qwen_gguf PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
    $<$<CONFIG:Release>:${QWEN_GGUF_RELEASE_CPU_FLAGS}>
)

if(APPLE)
    target_link_options(qwen_gguf PRIVATE
        $<$<CONFIG:Release>:-Wl,-dead_strip>
    )
endif()

if(QWEN_GGUF_ENABLE_OPENMP)
    if(APPLE)
        list(APPEND CMAKE_PREFIX_PATH
            /opt/homebrew/opt/libomp
            /usr/local/opt/libomp
        )
    endif()

    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(qwen_gguf PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(qwen_gguf PRIVATE QWEN_GGUF_USE_OPENMP=1)
    else()
        message(WARNING "OpenMP requested but not found; building without OpenMP support")
    endif()
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set_target_properties(qwen_gguf PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
endif()
