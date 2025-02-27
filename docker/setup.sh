#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR"/config.sh

# Build an image with environment set up

BASE_IMAGE=cnly/dotfiles-full:bullseye-20230109-c20df35

if ! docker image inspect "$BUILT_IMAGE":latest &> /dev/null; then

	<<-EOF docker build -t "$BUILT_IMAGE" -
	FROM $BASE_IMAGE

	# TVM deps
	RUN apt update && apt install -y \
		ninja-build zlib1g zlib1g-dev libssl-dev libbz2-dev libsqlite3-dev llvm-13 libopenblas-dev

	# Glow deps
	RUN apt install -y graphviz libpng-dev \
		libprotobuf-dev ninja-build protobuf-compiler wget \
		opencl-headers libgoogle-glog-dev libboost-all-dev \
		libdouble-conversion-dev libevent-dev libssl-dev libgflags-dev \
		libjemalloc-dev libpthread-stubs0-dev liblz4-dev libzstd-dev libbz2-dev \
		libsodium-dev libfmt-dev clang-13
	RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-13 100 && \
		update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-13 100 && \
		update-alternatives --install /usr/bin/python python /usr/bin/python3 100

	# NNFusion deps
	RUN apt install -y build-essential cmake git curl zlib1g zlib1g-dev libtinfo-dev unzip \
		autoconf automake libtool ca-certificates gdb sqlite3 libsqlite3-dev libcurl4-openssl-dev \
		libprotobuf-dev protobuf-compiler libgflags-dev libgtest-dev \
		libhwloc-dev libgmock-dev

	# Ghidra deps
	RUN apt install -y openjdk-17-jdk-headless unzip

	# Python
	RUN apt install -y libreadline-dev
	RUN /home/linuxbrew/.linuxbrew/bin/pyenv install 3.8.12
	EOF

fi

# Initialise venv
if [ ! -d "$PROJECT_DIR"/venv.docker ]; then
	docker run --rm -i \
		-v "$PROJECT_DIR":"$PROJECT_DIR" \
		-w "$PROJECT_DIR" \
		"$BUILT_IMAGE" \
		/bin/zsh -ic 'source env.sh'
fi

if [ ! -d "$PROJECT_DIR"/ghidra/ghidra-app ]; then
	"$SCRIPT_DIR"/run-in-docker.sh ghidra/install-ghidra.sh
fi

# Build tvm for our docker container
if [ ! -d "$TVM_DIR"/build.docker ]; then
	<<-EOF docker run --rm -i \
		-v "$PROJECT_DIR":"$PROJECT_DIR" \
		-w "$TVM_DIR" \
		"$BUILT_IMAGE" \
		/bin/bash
	set -e
	mkdir -p build.docker && cd build.docker
	cp "$SCRIPT_DIR"/resources/tvm-main.config.cmake ./config.cmake
	cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Release .. -G Ninja && ninja
	EOF
fi

# Build glow for our docker container
if [ ! -d "$GLOW_DIR"/build.docker ]; then
	# https://github.com/pytorch/glow/issues/5041
	git -C "$GLOW_DIR"/thirdparty/folly checkout v2020.10.05.00
	<<-EOF docker run --rm -i \
		-v "$PROJECT_DIR":"$PROJECT_DIR" \
		-w "$GLOW_DIR" \
		"$BUILT_IMAGE" \
		/bin/bash
	set -e
	mkdir -p build.docker && cd build.docker
	cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DGLOW_BUILD_TESTS=0 -DCMAKE_BUILD_TYPE=Release .. -G Ninja && ninja
	EOF
fi

# Build NNFusion for our docker container
BUILD_NNFUSION=
if [ ! -d "$NNFUSION_DIR"/build.docker ] && [ -n "$BUILD_NNFUSION" ]; then
	<<-EOF docker run --rm -i \
		-v "$PROJECT_DIR":"$PROJECT_DIR" \
		-w "$PROJECT_DIR" \
		"$BUILT_IMAGE" \
		/bin/bash
	set -e
	source env.sh
	cd "$NNFUSION_DIR"
	mkdir -p build.docker && cd build.docker
	cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Release .. -G Ninja && ninja
	# Init deps by building a model
	src/tools/nnfusion/nnfusion "$RESOURCES_DIR/nnfusion/init.onnx" -fdefault_device=CPU -format=onnx
	mv nnfusion_rt nnfusion_rt.base
	cd nnfusion_rt.base/cpu_codegen
	cp "$RESOURCES_DIR/nnfusion/eigen-init.cmake" ./eigen/eigen.cmake
	cmake -DBUILD_SHARED_LIBS=1 . && make -j
	# Replace the files for later builds
	for x in eigen threadpool; do cp "$RESOURCES_DIR/nnfusion/\$x.cmake" \$x/; done
	EOF
fi

echo "Built $BUILT_IMAGE - Use $SCRIPT_DIR/run-in-docker.sh to invoke commands inside it"
