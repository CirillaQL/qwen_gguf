CMAKE ?= cmake
BUILD_DIR := cmake-build
CONFIG ?= Release
CPU_FLAGS ?=
TUNE_NATIVE ?= ON

.PHONY: all configure build run clean

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DQWEN_GGUF_TUNE_NATIVE=$(TUNE_NATIVE) -DQWEN_GGUF_CPU_FLAGS="$(CPU_FLAGS)"

build: configure
	$(CMAKE) --build $(BUILD_DIR)

run: build
	./$(BUILD_DIR)/qwen_gguf

clean:
	rm -rf $(BUILD_DIR) build
