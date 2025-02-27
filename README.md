# DEBFA (DNN Executables Bit-Flip Attacks)

## Overview

This is the research artifact for the paper [Compiled Models, Built-In Exploits:
Uncovering Pervasive Bit-Flip Attack Surfaces in DNN
Executables](https://www.ndss-symposium.org/ndss-paper/compiled-models-built-in-exploits-uncovering-pervasive-bit-flip-attack-surfaces-in-dnn-executables/)
in NDSS 2025. It contains the code, data, and results for the paper and also
enables future research on the topic.

## Getting Started

### Setting Up

These external dependencies are needed for this project:

* [Pyenv](https://github.com/pyenv/pyenv#getting-pyenv)
* Docker

For the remaining dependencies, the first-time setup script will install them
automatically.

Clone this repo:

```sh
git clone https://github.com/yanzuochen/debfa
```

Run the first-time setup script:

```sh
cd debfa
./setup.sh
```

This script will

* Initialise all git submodules including the DL compilers
* Use Pyenv to install Python 3.8.12 and install the dependencies in the
  virtual environment `venv/`
* Build a Docker image `debfa-runner` so later tasks can be run in a stable
  and reproducible environment

### Noteworthy Notes

Remember to

```sh
source env.sh
```

before working on the project.

To use the Docker image built earlier, run

```sh
docker/run-in-docker.sh <command>
```

## Usage

### Adding New Datasets

Usually dataset files live in `datasets/`, as configured in `cfg.py`. If your
new dataset is too large, you may consider symlinking it from somewhere else,
or modifying the config.

Once your dataset is ready, you may need to edit `dataman.py` to expose it.

### Training Models

The definitions of models are in `support/models/`. To obtain the weights for a
model, you can use the training script `support/models/train.py`, for example:

```sh
for m in resnet50 densenet121 googlenet; do
	for x in CIFAR10 MNISTC FashionC; do
		support/models/train.py $m $x
	done
	for x in {0..11}; do
		support/models/train.py $m fake_$x
	done
done

support/models/train.py lenet1 MNIST --image-size 28

for x in {0..9}; do
	support/models/train.py lenet1 fakeM_$x --image-size 28
done
```

After training, you can generated the quantized models by running

```sh
docker/run-in-docker.sh support/models/quant.py
```

### Building and Analysing the Binary

After training, you can add the new model(s) to the build combinations in
`cfg.py`, and then run

```sh
dvc repro
```

which should build the binaries, import them into Ghidra, and generate the
sweep range information for later use.

Note that if you want to rebuild and/or redo the analysis done on a binary, you
need to **manually delete the existing files for them**. For the former you
just need to `rm` the files, but for the latter you may need to do it in Ghidra
GUI. Alternatively, you can discard the changes and restore them by running

```sh
git checkout -- dvc.lock
dvc checkout
```

### Sweeping for Vulnerable Bits

Use `flipsweep.py` to brute force through all bits in compute functions and
look for vulnerable ones. For example, run

```sh
docker/run-in-docker.sh ./flipsweep.py -m resnet50 -d CIFAR10
```

to look for vulnerable bits in the ResNet50 binary trained on CIFAR10. You may
run this command on every sweep configuration you want to test.

This process can take a long time, but you can safely interrupt by pressing
`^C` twice. The resulting file is saved in `results/sweep/`. Note that if you
run `dvc checkout` later, the new file may get **deleted**.

### Obtaining DRAM Profiles

A sweep summary JSON produced by
[Blacksmith](https://github.com/comsec-group/blacksmith) is required to be
placed in `results/dram/` for the Rowhammer attack experiment later.
For instructions to run Blacksmith, please follow the README in its repo.
To reduce the noise in the output of Blacksmith's timing function, you can also
replace it with the one from [TRRespass](https://github.com/vusec/trrespass).

### Reproducing Experiments

Once prior steps are done, you can reproduce the experiments in the paper by
running

```sh
dvc repro
```

which will run the basic analysis, find superbits, and perform the practical
Rowhammer experiments.
The results will be available in `results/`. Note that if you run `dvc
checkout` later, the new files may get **deleted**.
