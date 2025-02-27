from utils import BinaryInfo
import os.path
from typing import List, NamedTuple
from itertools import product

class CompilerConfig(NamedTuple):
    name: str
    version: str
    build_bis: List[BinaryInfo]

project_root = os.path.dirname(os.path.realpath(__file__))
datasets_root = os.path.join(project_root, 'datasets')
glow_root = os.path.join(project_root, 'compilers', 'glow-main')
nnfusion_root = os.path.join(project_root, 'compilers', 'nnfusion-main')

debug_dir = os.path.join(project_root, 'debug')
built_dir = os.path.join(project_root, 'built')
resources_dir = os.path.join(project_root, 'resources')
cache_dir = os.path.join(project_root, '.cache')
built_aux_dir = os.path.join(project_root, 'built-aux')
models_dir = os.path.join(project_root, 'models')

results_dir = os.path.join(project_root, 'results')
sweep_dir = os.path.join(results_dir, 'sweep')
bits_dir = os.path.join(results_dir, 'bits')
rewards_dir = os.path.join(results_dir, 'rewards')
dram_dir = os.path.join(results_dir, 'dram')
superbits_dir = os.path.join(results_dir, 'superbits')

ghidra_dir = os.path.join(project_root, 'ghidra')
analysis_dir = os.path.join(ghidra_dir, 'analysis')

batch_size = 20
dram_profile_path = f'{dram_dir}/sweep-A0-1x256MB.json'

tvm: CompilerConfig
def _init_tvm_cfg():
    model_names = ['resnet50', 'googlenet', 'densenet121']
    q_model_names = [f'Q{x}' for x in model_names]
    datasets = (
        ['CIFAR10', 'MNISTC', 'FashionC'] +
        # [f'merged_{i}' for i in range(10)] +
        [f'fake_{i}' for i in range(12)]
    )
    avx_modes = [True, False]
    opt_levels = [3, 0]

    combinations = list(product(model_names, datasets, avx_modes, opt_levels))
    combinations = [x for x in combinations if x[2] or x[3] != 0]  # Remove non-AVX opt_level=0
    combinations += list(product(q_model_names, datasets, [True], [3]))
    combinations += [['dcgan_g', 'MNIST', True, 3]]
    combinations += [['lenet1', 'MNIST', True, x] for x in [3, 0]]
    combinations += [['lenet1', f'fakeM_{i}', True, 3] for i in range(10)]

    bininfo_combs = tuple(BinaryInfo('tvm', 'main', *x) for x in combinations)
    global tvm
    tvm = CompilerConfig('tvm', 'main', bininfo_combs)
_init_tvm_cfg()

glow: CompilerConfig
def _init_glow_cfg():
    model_names = ['resnet50', 'googlenet', 'densenet121']
    datasets = (
        ['CIFAR10', 'MNISTC', 'FashionC'] +
        # [f'merged_{i}' for i in range(10)] +
        [f'fake_{i}' for i in range(12)]
    )
    combinations = list(product(model_names, datasets))
    combinations += [['dcgan_g', 'MNIST']]

    bininfo_combs = tuple(BinaryInfo('glow', 'main', *x, True, 3) for x in combinations)
    global glow
    glow = CompilerConfig('glow', 'main', bininfo_combs)
_init_glow_cfg()

nnfusion: CompilerConfig
def _init_nnf_cfg():
    model_names = ['resnet50', 'googlenet', 'densenet121']
    datasets = ['CIFAR10', 'MNISTC', 'FashionC']
    combinations = list(product(model_names, datasets))
    # combinations += [['dcgan_g', 'MNIST']]  # TODO

    bininfo_combs = tuple(BinaryInfo('nnfusion', 'main', *x, True, 3) for x in combinations)
    global nnfusion
    nnfusion = CompilerConfig('nnfusion', 'main', bininfo_combs)
_init_nnf_cfg()

all_build_bis = tvm.build_bis + glow.build_bis + nnfusion.build_bis
