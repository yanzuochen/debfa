import os
import sys
from typing import NamedTuple
import pickle
import json
from contextlib import contextmanager

class BinaryInfo(NamedTuple):
    compiler: str
    compiler_ver: str
    model_name: str
    dataset: str
    avx: bool
    opt_level: int

    @property
    def fname(self):
        return (
            f'{self.compiler}-{self.compiler_ver}-{self.model_name}-{self.dataset}'
            f'{"-noavx" if not self.avx else ""}'
            f'{f"-{self.opt_level}" if self.opt_level != 3 else ""}'
            '.so'
        )

    @property
    def nchans(self):
        return {
            'lenet1': 1,
            'lenet5': 1,
            'dcgan_g': 100,
        }.get(self.model_name, 3)

    @property
    def input_img_size(self):
        return {
            'lenet1': 28,
            'dcgan_g': 1,
        }.get(self.model_name, 32)

    @property
    def is_gan(self):
        return self.model_name in {'dcgan_g'}

    @property
    def output_defs(self):
        import cfg
        if self.is_gan:
            return [{'shape': (cfg.batch_size, 1, 64, 64), 'dtype': 'float32'}]
        return [{'shape': (cfg.batch_size, 10), 'dtype': 'float32'}]

    def just_datasets_differ(self, other):
        return self.compiler == other.compiler and \
            self.compiler_ver == other.compiler_ver and \
            self.model_name == other.model_name and \
            self.avx == other.avx and \
            self.opt_level == other.opt_level

    @staticmethod
    def from_fname(fname):
        fname = fname.rsplit('.', maxsplit=1)[0]  # Remove extension
        fsplit = fname.split('-')
        compiler, compiler_ver, model_name, dataset, *fsplit = fsplit
        avx = 'noavx' not in fsplit
        try:
            opt_level = int(fsplit[-1])
        except (ValueError, IndexError):
            opt_level = 3
        return BinaryInfo(compiler, compiler_ver, model_name, dataset, avx, opt_level)

@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

def reimport(*modules):
    import importlib
    for module in modules:
        if isinstance(module, str):
            module = importlib.import_module(module)
        importlib.reload(module)

def ensure_dir_of(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def save(obj, filepath, merge=True):
    assert not merge or isinstance(obj, dict)
    ensure_dir_of(filepath)
    if merge and os.path.exists(filepath):
        orig_obj = load(filepath)
        obj = {**orig_obj, **obj}
    with open(filepath, 'wb+') as f:
        pickle.dump(obj, f)

def load(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(obj, filepath, sorted=False):
    ensure_dir_of(filepath)
    with open(filepath, 'w+') as f:
        json.dump(obj, f, indent=2, sort_keys=sorted)

def save_str(s, filepath):
    ensure_dir_of(filepath)
    with open(filepath, 'w+') as f:
        f.write(str(s))

def warn(msg):
    print(f'⚠️  Warning: {msg}', file=sys.stderr)

def thread_first(data, *calls):
    """An approximation of the "->" macro in Clojure."""

    def tryexec(f, func_name):
        try:
            return f()
        except Exception as e:
            print(f'thread_first failed to execute {func_name}: {e}')
            raise

    for c in calls:
        c = list(c) if isinstance(c, tuple) else [c]
        f = c[0]
        if len(c) == 2 and isinstance(c[1], dict):
            data = tryexec(lambda: f(data, **c[1]), f)
            continue
        args = [data] + c[1:]
        data = tryexec(lambda: f(*args), f)
    return data

tf = thread_first
