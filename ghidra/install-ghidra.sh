#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
GHIDRA_DIR="$SCRIPT_DIR"/ghidra-app
GHIDRA_EXT_DIR="$GHIDRA_DIR"/Ghidra/Extensions
GRADLE_DIR="$SCRIPT_DIR"/gradle-7.6

cd "$SCRIPT_DIR"

# Required:
# sudo apt install -y openjdk-17-jdk-headless unzip

if [ ! -e "$GRADLE_DIR" ]; then
	wget -O gradle.zip \
		https://services.gradle.org/distributions/gradle-7.6-bin.zip
	unzip gradle.zip
	rm -f gradle.zip
fi

if [ ! -e "$GHIDRA_DIR" ]; then
	wget -O ghidra.zip \
		https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_10.2.2_build/ghidra_10.2.2_PUBLIC_20221115.zip
	unzip ghidra.zip
	mv ghidra_10.*_PUBLIC "$GHIDRA_DIR"
	rm -f ghidra.zip
fi

if [ ! -e "$GHIDRA_EXT_DIR"/ghidrathon ]; then
	pushd ghidrathon
	"$GRADLE_DIR"/bin/gradle -PGHIDRA_INSTALL_DIR="$GHIDRA_DIR"
	mv dist/*.zip "$GHIDRA_EXT_DIR"
	pushd "$GHIDRA_EXT_DIR"
	unzip ./*.zip
	rm ./*.zip
	popd
	popd
fi
