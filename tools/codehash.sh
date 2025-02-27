#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

for x in "$@"; do
	echo "$x" "$(objdump -s -j .text $x | tail -n+4 | sha1sum)"
done
