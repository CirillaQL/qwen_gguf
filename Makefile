CMAKE ?= cmake
BUILD_DIR := cmake-build
CONFIG ?= Debug

.PHONY: all configure build run clean

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)

build: configure
	$(CMAKE) --build $(BUILD_DIR)

run: build
	./$(BUILD_DIR)/qwen_gguf

clean:
	rm -rf $(BUILD_DIR) build
