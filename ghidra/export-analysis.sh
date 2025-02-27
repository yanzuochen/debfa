#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd "$SCRIPT_DIR"/..

./ghidra/exec-analyze-headless.sh -readOnly -noanalysis -process -postScript export-analysis.py
