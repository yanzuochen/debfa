#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DB_DIR="$SCRIPT_DIR"/db
PROJECT_NAME=debfa

analyze_exec="/usr/local/Caskroom/ghidra/1*/ghidra_*/support/analyzeHeadless"

[ -e $analyze_exec ] || analyze_exec="$SCRIPT_DIR"/ghidra-app/support/analyzeHeadless

exec() {
	echo "> $*"
	"$@"
}

mkdir -p "$DB_DIR"

exec $analyze_exec \
	"$DB_DIR" $PROJECT_NAME \
	-scriptPath "$SCRIPT_DIR" \
	"$@"
