# Source me!
# shellcheck disable=SC2148

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TVM_DIR="$SCRIPT_DIR"/compilers/tvm-main
NNFUSION_DIR="$SCRIPT_DIR"/compilers/nnfusion-main
TOOLS_DIR="$SCRIPT_DIR"/tools

VENV_DIR="$SCRIPT_DIR"/venv
[ ! -e /.dockerenv ] || VENV_DIR="$SCRIPT_DIR"/venv.docker

cd "$SCRIPT_DIR" || return

function remove_from_path() {
	local str=$1
	PATH=${PATH//:$str:/:}  # Equivalent to s|:$str:||g
	PATH=${PATH#"$str":}  # Equivalent to s|^$str:||
	PATH=${PATH%:"$str"}  # Equivalent to s|:$str\$||
}

function add_to_path() {
	local str=$1
	local prepend=$2
	remove_from_path "$str"
	if [ -n "$prepend" ]; then
		PATH=$str:$PATH
	else
		PATH=$PATH:$str
	fi
}

if [ ! -d "$VENV_DIR" ]; then
	if type -p pyenv &> /dev/null; then
		eval "$(pyenv init -)"
		if ! pyenv shell 3.8.12; then
			echo 'Installing Python 3.8.12 with pyenv...'
			sudo apt install -y \
				zlib1g zlib1g-dev libssl-dev libbz2-dev libsqlite3-dev \
				libreadline-dev libncursesw5 liblzma-dev
			pyenv install 3.8.12
		fi

		pyenv shell 3.8.12
		echo 'Setting up venv...'
		python3.8 -m venv "$VENV_DIR"
		pyenv shell system
	else
		python_found=
		for python_exe in python3.8 python3 python; do
			$python_exe --version | grep -qE '^Python 3\.8\.' && {
				python_found=1
				break
			}
		done
		[ -n "$python_found" ] || {
			echo 'Python 3.8 is required. Please install it first.'
			return 1
		}
		$python_exe -m venv "$VENV_DIR"
	fi

	source "$VENV_DIR"/bin/activate
	echo 'Installing dependencies...'
	python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
	python -m pip install -r requirements.txt
	deactivate
fi

source "$VENV_DIR"/bin/activate

add_to_path "$TOOLS_DIR" prepend
# shellcheck disable=SC2139
alias deactivate="remove_from_path $TOOLS_DIR; unalias deactivate; deactivate"

export PYTHONPATH="$TVM_DIR/python":"$NNFUSION_DIR/src/python"
