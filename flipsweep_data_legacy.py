# Data classes and utilities before we switched to dataclasses

import argparse
from collections import namedtuple
from typing import Dict, List, NamedTuple
from functional import seq
import numpy as np
import pickle

class ClassifierFlipResultColl(NamedTuple):
    base10_offset: int
    bitidx: int
    correct_pcts: List[float]
    top_labels_list: List[np.ndarray]

LegacySweepResult = namedtuple('LegacySweepResult', [
    'model_name', 'datasets', 'args', 'retcoll_map'
])

class SweepResult(NamedTuple):
    args: argparse.Namespace
    retcoll_map: Dict[int, List[ClassifierFlipResultColl]]  # Byte offset -> result colls

    @property
    def flat_result_colls(self):
        return seq(self.retcoll_map.values()).flatten().to_list()

    @staticmethod
    def from_legacy(lsr):
        if 'compiler' not in lsr.args:
            lsr.args.compiler = 'tvm'
        if 'compiler_version' not in lsr.args:
            lsr.args.compiler_version = 'main'
        return SweepResult(lsr.args, lsr.retcoll_map)

class SweepResultUnpickler(pickle.Unpickler):
    def __init__(self, *args, legacy=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.legacy = legacy

    def find_class(self, module, name):
        if name == 'SweepResult':
            return LegacySweepResult if self.legacy else SweepResult
        return globals()[name] if name in globals() else super().find_class(module, name)

def load_sweep_result(fname):
    with open(fname, 'rb') as f:
        try:
            return SweepResultUnpickler(f).load()
        except TypeError:
            lsr = SweepResultUnpickler(f, legacy=True).load()
            return SweepResult.from_legacy(lsr)
