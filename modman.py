import torch
import numpy as np
import os
import time
import tempfile
import shutil
import support.models as models
import tvm
from tvm.driver import tvmc
from tvm.target import Target
from tvm import relay, tir
from collections import namedtuple
import onnx
from scipy.special import softmax
from tqdm import tqdm
import subprocess
import ctypes
import warnings
import tarfile

from tvm.contrib.debugger import debug_executor
from tvm.contrib import graph_executor

from tvm.contrib import relay_viz
from tvm.contrib.relay_viz.dot import DotPlotter
from tvm.contrib.relay_viz.interface import DefaultVizParser

import utils
import cgutils
import cfg

data_root = cfg.project_root
targets = {
    'llvm': 'llvm',
    'avx2': 'llvm -mcpu=core-avx2',
    'avx2-cblas': 'llvm -mcpu=core-avx2 -libs=cblas',
}

IRModPack = namedtuple('IRModPack', ['irmod', 'params'])
ModRunResult = namedtuple('ModRunResult', ['outputs', 'exec_time', 'perf'])

def maybe_pad_data(data, batch_size):
    '''If data is smaller than batch_size, pad it with zero arrays.'''
    if data.shape[0] < batch_size:
        return np.concatenate([data, np.zeros((batch_size - data.shape[0], *data.shape[1:]), dtype=data.dtype)])
    return data

class WrappedRtMod:
    def __init__(self, rtmod, output_defs=None, batch_size=cfg.batch_size, nclasses=10, input_name='input0') -> None:
        self.rtmod = rtmod
        if output_defs is None:
            output_defs = [{'shape': [batch_size, nclasses], 'dtype': 'float32'}]
        self.output_defs = ensure_output_defs(output_defs)
        self.input_name = input_name
        self.output_bufs = [tvm.nd.empty(**d) for d in self.output_defs]

    def run(self, data, rettype='pred'):
        assert rettype in {'cov', 'pred', 'all'}
        rtmod = self.rtmod
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        padded_data = maybe_pad_data(data, self.output_bufs[0].shape[0])
        rtmod.set_input(self.input_name, padded_data)
        rtmod.run()

        def get_pred():
            return rtmod.get_output(0, self.output_bufs[0]).numpy()[:data.shape[0]]

        def get_cov():
            return [rtmod.get_output(i+1, buf).numpy()
                    for i, buf in enumerate(self.output_bufs[1:])]

        if rettype == 'cov':
            return get_cov()
        elif rettype == 'pred':
            return get_pred()
        else:
            return [get_pred()] + get_cov()

class GlowModel:
    def __init__(self, so_path, output_defs=None, weights_bin_path=None, batch_size=cfg.batch_size, nclasses=10):
        self.so_path = so_path
        if not weights_bin_path:
            weights_bin_path = f'{cfg.built_aux_dir}/{os.path.basename(so_path)}.weights.bin'
        self.weights_bin_path = weights_bin_path

        if output_defs is None:
            self.outarr = np.zeros((batch_size, nclasses), dtype=np.float32)
        else:
            assert len(output_defs) == 1
            assert output_defs[0]['dtype'] == 'float32'
            self.outarr = np.zeros(output_defs[0]['shape'], dtype=output_defs[0]['dtype'])

        so = ctypes.CDLL(so_path)
        so.init.argtypes = [ctypes.c_char_p]
        so.init.restype = None
        so.run_model.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_float, ndim=4, flags="C_CONTIGUOUS"),
            ctypes.POINTER(np.ctypeslib.c_intp),
            ctypes.POINTER(np.ctypeslib.c_intp),
            np.ctypeslib.ndpointer(ctypes.c_float, ndim=self.outarr.ndim, flags="C_CONTIGUOUS"),
        ]
        so.run_model.restype = ctypes.c_int
        self.so = so
        self.unloaded = False

        so.init(weights_bin_path.encode('utf-8'))

    def run(self, data, rettype='pred'):
        assert rettype == 'pred'
        assert not self.unloaded
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        assert data.dtype == np.float32
        padded_data = maybe_pad_data(data, self.outarr.shape[0])
        assert padded_data.shape[0] == self.outarr.shape[0]
        err = self.so.run_model(padded_data, data.ctypes.strides, data.ctypes.shape, self.outarr)
        if err:
            raise RuntimeError(f'Glow model returned error code {err}')
        return self.outarr.copy()[:data.shape[0]]

    def unload(self):
        dlclose = ctypes.cdll.LoadLibrary('').dlclose
        dlclose.argtypes = [ctypes.c_void_p]
        dlclose(self.so._handle)
        del self.so
        self.unloaded = True

def ensure_output_defs(output_defs):
    return [
        {
            'dtype': d['dtype'],
            'shape': [x.value if isinstance(x, tir.IntImm) else x for x in d['shape']]
        }
        for d in output_defs
    ]

def make_output_bufs(output_defs, zero=False):
    output_defs = ensure_output_defs(output_defs)
    if zero:
        return [tvm.nd.array(np.zeros(**d)) for d in output_defs]
    return [tvm.nd.empty(**d) for d in output_defs]

def load_bounds(model_name, dataset, mode, data_path=None) -> list:
    if not data_path:
        data_path = f'{data_root}/Coverage/{dataset}-{model_name}-{mode}.pth'
    data = torch.load(data_path, map_location=torch.device('cpu'))
    if isinstance(data, list):
        return data
    assert 'range' in data
    # We rely on the order of the dict here
    ret = []
    for lows, highs in data['range'].values():
        ret.append(torch.stack([lows, highs]).numpy())
    return ret

def load_covs(model_name, dataset, mode, data_path=None) -> list:
    if not data_path:
        data_path = f'{data_root}/Coverage/{dataset}-{model_name}-{mode}.pth'
    data = torch.load(data_path, map_location=torch.device('cpu'))
    if isinstance(data, list):
        return data
    # We rely on the order of the dict here
    return [v.numpy() for v in data.values()]

def create_rtmod(libmod, debug=False, debug_dump_root=None):
    dev = tvm.device('cpu')
    if not debug:
        return graph_executor.GraphModule(libmod["default"](dev))
    return debug_executor.create(libmod['get_graph_json'](), libmod, dev, dump_root=debug_dump_root)

def get_torch_mod(model_name, dataset):
    model_class = getattr(models, model_name)
    torch_model = model_class(pretrained=False)
    torch_model.eval()
    torch_model.load_state_dict(
        torch.load(
            f'{cfg.models_dir}/{dataset}/{model_name}/{model_name}.pt',
            map_location=torch.device('cpu')
        )
    )
    return torch_model

def export_torch_mod(model, input_shape, fname, optimise=False, onnx_ver=14):
    # Note: For use with glow/nnfusion, set onnx_ver=12
    x = torch.randn(*input_shape, requires_grad=False, dtype=torch.float32)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    opt_args = {}
    if not optimise:
        opt_args['training'] = torch.onnx.TrainingMode.TRAINING
        opt_args['do_constant_folding'] = False
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', module=r'torch\.onnx\.utils', message=r'.+\.num_batches_tracked.+'
        )
        torch.onnx.export(model, x, fname, **opt_args,
                          export_params=True, opset_version=onnx_ver,
                          input_names=['input0'], output_names=['output0'])

def save_irmod_viz(irmod, basename):
    viz = relay_viz.RelayVisualizer(irmod, plotter=DotPlotter(), parser=DefaultVizParser())
    viz.render(basename)

def extra_params_removed(params_dict):
    return {k: v for k, v in params_dict.items() if not k.startswith('__ep_')}

def extra_params_zeroed(params_dict):
    return {k: (v if not k.startswith('__ep_') else tvm.nd.array(np.zeros_like(v)))
            for k, v in params_dict.items()}

def create_zeroed_extra_params_dict(extra_params_vars, ones=False):
    fill_fn = np.ones if ones else np.zeros
    ep_types = [cgutils.get_type(v) for v in extra_params_vars]
    ep_defs = [{'shape': t.concrete_shape, 'dtype': t.dtype} for t in ep_types]
    return {f'__ep_{i}': tvm.nd.array(fill_fn(**d))
            for i, d in enumerate(ep_defs)}

def get_irmod(
    model_name, dataset, mode, batch_size, image_size, ep_path=None, nchannels=3,
    zero_extra_params=False, include_extra_params=True, allow_ep_load_failure=False,
):
    """Loads a stored model and its params (including extra params for the
    specified mode) from the disk.
    zero_extra_params: If true, zero out the extra parameters in the model.
    Used for range building."""

    input_name = "input0"
    input_shape = (batch_size, nchannels, image_size, image_size)
    if model_name.startswith('Q'):
        fname = f'{cfg.models_dir}/{dataset}/{model_name}/{model_name}-{batch_size}.onnx'
        mod, params = relay.frontend.from_onnx(onnx.load(fname), {'input0': input_shape})
    else:
        torch_model = get_torch_mod(model_name, dataset)
        scripted_model = torch.jit.trace(torch_model, torch.randn(input_shape)).eval()
        mod, params = relay.frontend.from_pytorch(scripted_model, [(input_name, input_shape)])

    if mode not in ['none', 'rb', 'cb', 'npcb', 'gn1', 'gn2', 'gninf'] and include_extra_params:
        ep_load_fn = load_covs
        if mode == 'WNBC':
            mode = 'NBC'
        elif mode == 'npcv':
            mode = 'npc'
        if mode in {'NBC', 'KMN'}:
            ep_load_fn = load_bounds
        try:
            extra_params = ep_load_fn(model_name, dataset, mode, data_path=ep_path)
            extra_params_dict = {f'__ep_{idx}': v for idx, v in enumerate(extra_params)}
            params = {**params, **extra_params_dict}
        except Exception as e:
            if not allow_ep_load_failure:
                raise e
            utils.warn(f'Will skip loading extra params because of error: {e}')

    if zero_extra_params:
        params = extra_params_zeroed(params)

    return IRModPack(mod, params)

def load_module(path, debug=False, debug_dump_root=None):
    libmod = tvm.runtime.load_module(path)
    return create_rtmod(libmod, debug=debug, debug_dump_root=debug_dump_root)

def load_built_model(bi: utils.BinaryInfo):
    fpath = f'{cfg.built_dir}/{bi.fname}'
    if bi.compiler == 'tvm':
        return WrappedRtMod(load_module(fpath), output_defs=bi.output_defs)
    elif bi.compiler == 'glow':
        return GlowModel(fpath, output_defs=bi.output_defs)
    else:
        raise ValueError(f'Unknown compiler: {bi.compiler}')

def build_module(irmod, params, export_path=None, opt_level=3, target=targets['avx2'], is_qnn=False):
    start_time = time.time()
    # basename = os.path.basename(export_path)
    # utils.save_str(irmod, f'{cfg.debug_dir}/relay-irs/{basename}-orig.log')
    if is_qnn:
        irmod = cgutils.QNNPreLegalize()(irmod)
        # utils.save_str(irmod, f'{cfg.debug_dir}/relay-irs/{basename}-preleg.log')
        # Run first half of optimisations
        blocker = cgutils.PassBlocker('SimplifyInference')
        with tvm.transform.PassContext(opt_level=opt_level, instruments=[blocker]):
            irmod, _ = relay.optimize(irmod, target=target)
        # utils.save_str(irmod, f'{cfg.debug_dir}/relay-irs/{basename}-phase1.log')
        car = cgutils.ConstArgsReplacer()
        irmod = car.transform_mod(irmod)
        # utils.save_str(irmod, f'{cfg.debug_dir}/relay-irs/{basename}-phase2.log')
        # Run the remaining optimisations
        blocker.negate = True
        with tvm.transform.PassContext(opt_level=opt_level, instruments=[blocker]):
            irmod, _ = relay.optimize(irmod, target=target)
        # utils.save_str(irmod, f'{cfg.debug_dir}/relay-irs/{basename}-phase3.log')
        print(f'Optimised (level {opt_level}) in {time.time() - start_time:2f} seconds.')
        irmod = car.transform_mod(irmod, recover_mode=True)
        # utils.save_str(irmod, f'{cfg.debug_dir}/relay-irs/{basename}-phase4.log')
        start_time = time.time()
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(irmod, target=target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(irmod, target=target, params=params)
    rtmod = create_rtmod(lib, debug=False)
    print(f'Module built in {time.time() - start_time:2f} seconds.')
    if export_path:
        utils.ensure_dir_of(export_path)
        lib.export_library(export_path)
        print(f'Saved to {export_path}.')
    return rtmod, lib

def build_module_tvmc(irmod, params, export_path=None, target=targets['avx2'], **kwargs):
    start_time = time.time()
    with tempfile.NamedTemporaryFile() as tmpf:
        tvmc_mod = tvmc.TVMCModel(irmod, params=params)
        package = tvmc.compile(tvmc_mod, target=target, package_path=tmpf.name, **kwargs)
        print(f'Module built in {time.time() - start_time:2f} seconds.')
        lib_path = package.lib_path
        if export_path:
            utils.ensure_dir_of(export_path)
            shutil.move(lib_path, export_path)
            lib_path = export_path
            print(f'Saved to {export_path}.')
        lib = tvm.runtime.load_module(lib_path)
        rtmod = create_rtmod(lib)
    return rtmod, lib

def get_json_and_ir(irmod, params, opt_level=3, target=targets['avx2']):
    raw_targets = Target.canon_multi_target_and_host(Target.target_or_current(target))
    with tvm.transform.PassContext(opt_level=opt_level):
        graph_json_str, rtmod, _params = relay.build_module.BuildModule().build(irmod, target=raw_targets, params=params)
    return graph_json_str, rtmod.get_source()

def build_glow_model(model_name, dataset, batch_size, image_size, outdir, nchannels=3, opt_level=3, glow_build_dirname=f'build{".docker" if os.path.exists("/.dockerenv") else ""}'):
    assert not model_name.startswith('Q'), 'Quantized ONNX models are not supported'
    utils.ensure_dir_of(f'{outdir}/.')
    with tempfile.NamedTemporaryFile() as onnx_file:
        torch_model = get_torch_mod(model_name, dataset)
        input_shape = (batch_size, nchannels, image_size, image_size)
        export_torch_mod(torch_model, input_shape, onnx_file.name, onnx_ver=12)
        subprocess.run([
            f'{cfg.glow_root}/{glow_build_dirname}/bin/model-compiler',
            '--backend=CPU', '--network-name=model', '--main-entry-name=forward',
            '--bundle-api=dynamic', '--relocation-model=pic', '-g',
            f'--model={onnx_file.name}', f'--emit-bundle={outdir}'
        ], check=True)
        with utils.cwd(outdir):
            shutil.copy(f'{cfg.resources_dir}/glow/model.cpp', '.')
            subprocess.run([
                'g++', '-shared', '-fPIC', f'-O{opt_level}', f'-omodel.so',
                'model.o', 'model.cpp',
            ], check=True)

def build_nnf_model(model_name, dataset, batch_size, image_size, outdir, nchannels=3, nnf_build_dirname=f'build{".docker" if os.path.exists("/.dockerenv") else ""}'):
    assert not model_name.startswith('Q'), 'Quantized models are not supported'
    nnf_build_dir = f'{cfg.nnfusion_root}/{nnf_build_dirname}'
    utils.ensure_dir_of(f'{outdir}/.')
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export model
        torch_model = get_torch_mod(model_name, dataset)
        input_shape = (batch_size, nchannels, image_size, image_size)
        export_torch_mod(torch_model, input_shape, f'{tmpdir}/model.onnx', onnx_ver=12)
        # Codegen
        subprocess.run([
            f'{nnf_build_dir}/src/tools/nnfusion/nnfusion', 'model.onnx',
            '-format=onnx', '-fdefault_device=CPU', '-fextern_result_memory=1', '-min_log_level=2',
        ], check=True, cwd=tmpdir)
        # Patch & compile
        with utils.cwd(f'{tmpdir}/nnfusion_rt/cpu_codegen'):
            # Patch: Ensure weights are loaded
            subprocess.run([
                'sed', '-i', '-E', r's|(std::ifstream bin_file.+)|\1\n assert(bin_file.is_open());|',
                'nnfusion_rt.cpp'
            ], check=True)
            # Patch: Fix output var name
            out_var_name = utils.load_json('para_info.json')['output']['output0']['name']
            wrong_var_name = f'tensor_{int(out_var_name.split("_")[1]) - 1}'
            subprocess.run([
                'sed', '-i', '-E', rf's|(\b){wrong_var_name}(\b)|\1{out_var_name}\2|',
                'nnfusion_rt.cpp'
            ], check=True)
            # Build
            subprocess.run(['cmake', '-DBUILD_SHARED_LIBS=1', '.'], check=True)
            # Skip redownloading/rebuilding deps
            for x in ['eigen', 'hwloc', 'mkldnn']:
                shutil.rmtree(x)
                os.symlink(f'{nnf_build_dir}/nnfusion_rt.base/cpu_codegen/{x}', x)
            subprocess.run(['make', '-j'], check=True)
            # Output
            shutil.copy('libnnfusion_cpu_rt.so', outdir)
            with tarfile.open(f'{outdir}/data.tar', 'x:') as tar:
                [tar.add(x) for x in ['Constant', 'para_info.json']]

def run_module(rtmod, input_data, output_defs=None, output_bufs=None, input_name='input0',
        nclasses=10, output_labels=False,
        benchmark=False, benchmark_std_threshold=1e-2, benchmark_max_trials=10):
    """Note that debug timing may not work when benchmark is enabled."""

    rtmod.set_input(input_name, input_data)
    exec_time, perf = None, None
    if benchmark:
        start_time = time.time()
        dev = tvm.device('cpu')
        for _ in range(benchmark_max_trials):
            perf = rtmod.benchmark(dev, number=100, repeat=3)
            if perf.std < perf.mean * benchmark_std_threshold:
                break
        else:
            utils.warn(f'Benchmark did not achieve desired stddev.')
        exec_time = time.time() - start_time
    else:
        rtmod.run()

    if not output_defs:
        output_defs = [{'shape': [input_data.shape[0], nclasses], 'dtype': 'float32'}]

    if not output_bufs:
        output_bufs = make_output_bufs(output_defs)

    outputs = [rtmod.get_output(i, buf).numpy() for i, buf in enumerate(output_bufs)]

    if output_labels:
        # print(f'Scores: {outputs[0]}')
        scores = softmax(outputs[0], axis=1)
        labels = np.argsort(scores, axis=1)[:, ::-1]
        outputs[0] = labels

    return ModRunResult(outputs, exec_time, perf)

def run_module_over_loader(rtmod, data_loader, outfile=None, **kwargs):
    """Runs the module over the data loader.
    Returns a list of ModRunResult objects."""
    ret = []
    if outfile and '/' in outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    for i, (xs, ys) in enumerate(tqdm(data_loader)):
        ret.append(run_module(rtmod, xs, **kwargs))
        if i % 100 == 0 and outfile:
            torch.save(ret, outfile)
    if outfile:
        print(f'Saved to {outfile}.')
    return ret
