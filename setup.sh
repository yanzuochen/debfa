#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

set -x

git submodule update --init --recursive
# https://github.com/pytorch/glow/issues/5041
git -C compilers/glow-main/thirdparty/folly checkout v2020.10.05.00

source "$SCRIPT_DIR"/env.sh

"$SCRIPT_DIR"/tools/ensure_datasets.py

# TODO: Complete build dependencies for host
# sudo apt install -y llvm-11 ninja-build libopenblas-dev libgl1

"$SCRIPT_DIR"/docker/setup.sh
