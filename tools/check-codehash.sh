#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


result=$("$SCRIPT_DIR"/codehash.sh "$@")
if [ "$(echo "$result" | cut -d' ' -f2 | uniq | wc -l)" = "1" ]; then
	exit 0
fi
echo "$result"
exit 1
