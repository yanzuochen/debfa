# shellcheck shell=bash

PROJECT_DIR=$(dirname "$SCRIPT_DIR")
TVM_DIR="$PROJECT_DIR"/compilers/tvm-main
GLOW_DIR="$PROJECT_DIR"/compilers/glow-main
NNFUSION_DIR="$PROJECT_DIR"/compilers/nnfusion-main
RESOURCES_DIR="$PROJECT_DIR"/resources

BUILT_IMAGE=debfa-runner
