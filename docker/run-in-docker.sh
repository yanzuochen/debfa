#! /usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR"/config.sh

execute() {
	echo "> $*"
	"$@"
}

run_in_docker() {
	execute docker run --rm -it \
		-v "$PROJECT_DIR":"$PROJECT_DIR" \
		-w "$PROJECT_DIR" \
		-e TVM_LIBRARY_PATH="$TVM_DIR"/build.docker \
		--ulimit core=0 \
		"$BUILT_IMAGE" \
		/bin/bash -ic "source \"$PROJECT_DIR\"/env.sh && ( $* )"
}

run_in_docker "$@" || ret=$?

echo "Fixing potential permissions issues..."
# shellcheck disable=SC1010
run_in_docker \
	for x in ghidra/{db,analysis} models built results built-aux\; do \
		chown -R "$(id -u):$(id -g)" "$PROJECT_DIR"/\$x \&\> /dev/null \; \
	done || true

exit $ret
