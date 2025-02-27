#! /usr/bin/env python3

import os
import resource
import pickle
import shutil
import psutil
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union
import multiprocessing as mp
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partialmethod
from functional import seq
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

import dataman
from eval import evalutils
import utils
import fliputils as fu
import cfg
import flipsweep_data_legacy

@dataclass
class FlipResult:
    '''The flip result for a single bit in a single binary.
    Used only during sweeping; not exported.'''
    correct_pct: float
    top_labels: np.ndarray
    top_label_change_pct: float
    lpips_avg: float
    fid: float

    from_completed: bool = False

    def __setattr__(self, __name: str, __value: Any) -> None:
        assert __name in self.__dataclass_fields__, f'Cannot set attribute {__name}'
        super().__setattr__(__name, __value)

    @staticmethod
    def empty():
        return FlipResult(-1., None, -1., -1., -1.)

    @classmethod
    def get_completed_or_new(cls, args, abs_byteoff, bitidx, bi_idx):
        if not args.completing_file:
            return cls.empty()
        retcolls = completing_retcoll_map.get(abs_byteoff)
        if retcolls is None:  # Not swept yet in completed file
            return cls.empty()
        for retcoll in retcolls:
            if retcoll.bitidx == bitidx:
                break
        else:  # Bit not flippable in completed file
            return None
        nbis = len(retcoll.correct_pcts)
        if bi_idx >= nbis:
            return cls.empty()
        return cls(
            retcoll.correct_pcts[bi_idx],
            retcoll.top_labels_list[bi_idx],
            retcoll.top_label_change_pcts[bi_idx],
            getattr(retcoll, 'lpips_avgs', [-1.]*nbis)[bi_idx],
            getattr(retcoll, 'fids', [-1.]*nbis)[bi_idx],
            from_completed=True,
        )

@dataclass
class FlipResultCollV1:
    '''A collection of flip results for a single bit across all binaries.'''

    base10_offset: int
    bitidx: int
    correct_pcts: List[float]
    top_labels_list: List[np.ndarray]
    top_label_change_pcts: List[float]

    def __setattr__(self, __name: str, __value: Any) -> None:
        assert __name in self.__dataclass_fields__, f'Cannot set attribute {__name}'
        super().__setattr__(__name, __value)

    def append_result(self, ret: FlipResult):
        self.correct_pcts.append(ret.correct_pct)
        self.top_labels_list.append(ret.top_labels)
        self.top_label_change_pcts.append(ret.top_label_change_pct)

@dataclass
class FlipResultCollV2(FlipResultCollV1):
    lpips_avgs: List[float]
    fids: List[float]  # FID calculates the distance between two distributions

    @staticmethod
    def empty(base10_offset, bitidx):
        return FlipResultCollV2(base10_offset, bitidx, [], [], [], [], [])

    def append_result(self, ret: FlipResult):
        super().append_result(ret)
        self.lpips_avgs.append(ret.lpips_avg)
        self.fids.append(ret.fid)

@dataclass
class SweepResultV3:
    args: argparse.Namespace
    retcoll_map: Dict[
        int,
        List[Union[FlipResultCollV1, FlipResultCollV2]]
    ]  # Byte offset -> flip result colls
    orig_top_labels: List[np.ndarray]

    def __setattr__(self, __name: str, __value: Any) -> None:
        assert __name in self.__dataclass_fields__, f'Cannot set attribute {__name}'
        super().__setattr__(__name, __value)

    @property
    def flat_result_colls(self):
        return seq(self.retcoll_map.values()).flatten().to_list()

class SweepResultUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'SweepResult':
            return SweepResultV3
        if name == 'FlipResultColl':
            return FlipResultCollV1
        return globals()[name] if name in globals() else super().find_class(module, name)


val_loaders: Dict[utils.BinaryInfo, DataLoader] = {}
lmis: Dict[utils.BinaryInfo, fu.LoadedModInfo] = {}
completing_retcoll_map: Dict[int, List[Union[FlipResultCollV1, FlipResultCollV2]]] = {}
orig_top_labels: Dict[utils.BinaryInfo, Any] = {}
gan_evaluators: Dict[utils.BinaryInfo, evalutils.GANEvaluator] = {}
specific_bytes_bits: Set[Tuple[int, int]] = set()

# Avoids crashes caused by simultaneous access to cuda (?)
gan_eval_lock = threading.Lock()

def load_sweep_result(fname):
    with open(fname, 'rb') as f:
        try:
            ret = SweepResultUnpickler(f).load()
        except (TypeError, AttributeError):
            ret = flipsweep_data_legacy.load_sweep_result(fname)
        if 'compiler' not in ret.args:
            ret.args.compiler = 'tvm'
        if 'compiler_version' not in ret.args:
            ret.args.compiler_version = 'main'
        if 'completing_file' not in ret.args:
            ret.args.completing_file = None
        if 'label_change_thresh' not in ret.args:
            ret.args.label_change_thresh = 0
        if 'specific_bits_file' not in ret.args:
            ret.args.specific_bits_file = None
        if 'gan_images_only' not in ret.args:
            ret.args.gan_images_only = False
        return ret

def get_val_loader(bi, fast):
    if fast:
        return dataman.get_sampling_loader_v2(bi.dataset, bi.input_img_size, 'test', cfg.batch_size, n_per_class=10, num_workers=0)
    return dataman.get_benign_loader(bi.dataset, bi.input_img_size, 'test', cfg.batch_size, num_workers=0)

def flip_proc_work(bi: utils.BinaryInfo, byteidx, bitidx, is_gan, gan_images_only, fret):
    """The work done by the child process that loads the model and does the flip."""
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # Disable tqdm by default
    lmi = lmis[bi]
    fu.flip_bits(lmi, [fu.new_flip(byteidx, bitidx)])
    if is_gan:
        evaluator = gan_evaluators[bi]
        if not gan_images_only:
            outs = evaluator.get_gan_outputs(lmi.mod)
        else:
            outs = evaluator.get_gan_outputs(
                lmi.mod, debug=True, debug_fname=f'{byteidx:x}_{bitidx}'
            )
        fret.append(outs)
    else:
        val_loader = val_loaders[bi]
        acc, top_labels = evalutils.check_accuracyv2(lmi.mod, val_loader)
        fret.extend([acc, top_labels])

def flip_thread_work(next_byte_fn, on_byte_done, bis, args):
    """The worker that starts the flipping child processes."""
    pman = mp.Manager()
    while True:
        abs_byteoff = next_byte_fn()
        if abs_byteoff is None:
            break
        byte_rets = []
        for bitidx in range(8):
            if specific_bytes_bits and (abs_byteoff, bitidx) not in specific_bytes_bits:
                continue
            unflippable_reason = None
            frc = FlipResultCollV2.empty(abs_byteoff, bitidx)
            for i, bi in enumerate(bis):
                byteidx = abs_byteoff - fu.GHIDRA_BASE_ADDR - lmis[bi].region_elf_offset
                bit_bi_desc = f'0x{abs_byteoff:x} (+0x{byteidx:x}) {bitidx} ({bi.dataset})'
                fr = FlipResult.get_completed_or_new(
                    args, abs_byteoff, bitidx, i
                )
                start_time = time.time()
                time_used = 0
                if fr is None:
                    unflippable_reason = '🗄️ not flippable in completed file'
                    break

                if not fr.from_completed:
                    fret = pman.list()
                    proc = mp.Process(target=flip_proc_work, args=(
                        bi, byteidx, bitidx, args.gan, args.gan_images_only, fret
                    ))
                    proc.start()
                    proc.join(args.timeout)
                    time_used = time.time() - start_time

                    if proc.is_alive():
                        unflippable_reason = f'⏳ timeout'
                        proc.kill()
                        break
                    if proc.exitcode != 0:
                        unflippable_reason = f'💀 exitcode {proc.exitcode}'
                        break
                    if len(fret) != {True: 1, False: 2}[args.gan]:
                        unflippable_reason = f'🤯 len(fret) {len(fret)}'
                        break

                    if not args.gan:
                        acc, fr.top_labels = fret
                        fr.correct_pct = 100 * acc
                        fr.top_labels = np.stack(fr.top_labels)
                        fr.top_label_change_pct = 100 * evalutils.calc_labels_change(
                            orig_top_labels[bi], fr.top_labels
                        )
                    elif not args.gan_images_only:
                        outputs = fret[0]
                        evaluator = gan_evaluators[bi]
                        with gan_eval_lock:
                            top_labels, lchange, lpips, fid = evaluator.eval(outputs)
                        fr.top_labels, fr.top_label_change_pct = top_labels, lchange*100
                        fr.lpips_avg, fr.fid = lpips, fid

                if fr.correct_pct > args.acc_thresh:
                    unflippable_reason = f'💯 acc {fr.correct_pct}%'
                    break
                if fr.top_label_change_pct < args.label_change_thresh:
                    unflippable_reason = f'🔄 label change {fr.top_label_change_pct}%'
                    break

                print(f'Flippable: {bit_bi_desc} @ {time_used:.2f}s')
                print(f'Top labels ({fr.top_label_change_pct:.2f}% changed): {fr.top_labels.flatten().tolist()}')
                if not args.gan:
                    print(f'Acc. after flipping: {fr.correct_pct:.2f}%')
                else:
                    print(f'LPIPS: {fr.lpips_avg:.2f}, FID: {fr.fid:.2f}')
                frc.append_result(fr)
            if unflippable_reason:
                print(f'Unflippable: {bit_bi_desc} ({unflippable_reason}) @ {time_used:.2f}s')
                continue
            byte_rets.append(frc)
        on_byte_done(abs_byteoff, byte_rets)

def get_all_bytes(compute_fns, nchunks=None, chunk_idx=None):
    # For compute fns, generates Ghidra offsets
    ret = []
    for _cfname, cf in compute_fns.items():
        fn_start = cf['base10_offset']
        nbytes = cf['size']
        for byteoff in range(fn_start, fn_start + nbytes):
            ret.append(byteoff)
    if nchunks is not None:
        # Divide ret into chunks and return the chunk at chunk_idx
        ret = np.array_split(ret, nchunks)[chunk_idx].tolist()
    return ret

def read_existing_sweepret(path, expected_args):
    ret = load_sweep_result(path)
    # A list of args that we don't really care about
    ret.args.nworkers = expected_args.nworkers
    ret.args.out_dir = expected_args.out_dir
    ret.args.timeout_x = expected_args.timeout_x
    ret.args.timeout = expected_args.timeout
    ret.args.device = expected_args.device
    ret.args.max_mem_gb = expected_args.max_mem_gb
    ret.args.nchunks = expected_args.nchunks
    ret.args.chunk_idx = expected_args.chunk_idx
    ret.args.save_interval = expected_args.save_interval
    assert ret.args == expected_args, \
        f'Args from existing file are unexpected.\n' \
        f'Actual: {ret.args}\nWanted: {expected_args}'
    return ret

def read_specific_bits_file(path):
    # File format: g_offset,bitidx lines
    ret = []
    with open(path, 'r') as f:
        for line in f:
            g_offset, bitidx = line.strip().split(',')
            ret.append((int(g_offset, 16), int(bitidx)))
    return ret

def sweep(args):
    # Load all compute fn ranges jsons, sort them, make sure they're the same
    bis = (
        seq(args.datasets)
        .map(lambda d: utils.BinaryInfo(
            args.compiler, args.compiler_version,
            args.model_name, d, avx=not args.no_avx, opt_level=args.opt_level
        ))
        .list()
    )
    compute_fns = (
        seq(bis)
        .map(lambda bi: bi.fname)
        .map(lambda n: f'{cfg.ghidra_dir}/analysis/{n}-analysis.json')
        .map(utils.load_json)
        .map(lambda j: j['compute_fns'])
        .reduce(lambda a, b: a if a == b else False)
    )
    assert compute_fns is not False, 'All compute function ranges must be the same'

    if not args.specific_bits_file:
        all_bytes = get_all_bytes(compute_fns, nchunks=args.nchunks, chunk_idx=args.chunk_idx)
        total_bytes = len(all_bytes)
        real_total_bytes = len(get_all_bytes(compute_fns))
    else:
        bytes_bits_pairs = read_specific_bits_file(args.specific_bits_file)
        all_bytes = [x[0] for x in bytes_bits_pairs]
        total_bytes = len(all_bytes)
        real_total_bytes = total_bytes
        specific_bytes_bits.update(bytes_bits_pairs)

    outfile = (
        f'{args.out_dir}/'
        f'{args.compiler}-{args.compiler_version}-'
        f'{args.model_name}-{"+".join(args.datasets)}'
        f'{"-noavx" if args.no_avx else ""}'
        f'{f"-{args.opt_level}" if args.opt_level != 3 else ""}'
        f'-sweep.pkl'
    )
    utils.ensure_dir_of(outfile)

    oldret = None
    if os.path.exists(outfile):
        oldret = read_existing_sweepret(outfile, args)
        print(f'Old progress: {len(oldret.retcoll_map)}/{real_total_bytes} bytes ({len(oldret.retcoll_map) / real_total_bytes:.2%})')
        if len(oldret.retcoll_map) == real_total_bytes:
            print('Already finished')
            return

    if args.completing_file:
        com = load_sweep_result(args.completing_file)
        assert len(com.retcoll_map) == real_total_bytes, 'Completing file must have finished sweeping'
        assert com.args.compiler == args.compiler
        assert com.args.compiler_version == args.compiler_version
        assert com.args.model_name == args.model_name
        assert com.args.opt_level == args.opt_level
        assert com.args.no_avx == args.no_avx
        # Allow completing with stricter thresholds. Old completed results will
        # be discarded during the sweep if they don't meet the new requirement.
        assert com.args.acc_thresh >= args.acc_thresh
        assert com.args.label_change_thresh <= args.label_change_thresh
        assert all(a == b for a, b in zip(com.args.datasets, args.datasets))
        completing_retcoll_map.update(com.retcoll_map)

    lmis.update({
        bi: fu.load_mod(bi) for bi in tqdm(bis, desc='Loading model binaries')
    })

    eval_times = []
    if not args.gan:
        val_loaders.update({
            bi: get_val_loader(bi, args.fast) for bi in tqdm(bis, desc='Loading val loaders')
        })
        accs = []
        for bi in bis:
            start_time = time.time()
            lmi = lmis[bi]
            loader = val_loaders[bi]
            acc, top_labels = evalutils.check_accuracyv2(lmi.mod, loader)
            if not bi.dataset.startswith('fake'):
                assert acc > 0.5, f'Accuracy for {bi.dataset} is {acc:.2%} (too low)'
            eval_times.append(time.time() - start_time)
            accs.append(acc)
            orig_top_labels[bi] = np.stack(top_labels)
        avg_acc = sum(accs) / len(accs)
        print(f'Baseline avg acc: {avg_acc:.2%}')
    else:
        gan_evaluators.update({
            bi: evalutils.GANEvaluator(bi.dataset, device=args.device)
            for bi in tqdm(bis, desc='Loading GAN evaluators')
        })
        for bi in bis:
            lmi = lmis[bi]
            gan_evaluators[bi].set_ref(lmi.mod)
            orig_top_labels[bi] = gan_evaluators[bi].ref_labels.cpu().numpy()
            start_time = time.time()
            gan_evaluators[bi].get_gan_outputs(lmi.mod)
            eval_times.append(time.time() - start_time)

    baseline_eval_time = sum(eval_times) / len(eval_times)
    print(f'Baseline eval time: {baseline_eval_time:.2f}s')
    if args.timeout == 0:
        args.timeout = baseline_eval_time * args.timeout_x

    ret = SweepResultV3(
        args, {}, [np.stack(x) for x in orig_top_labels.values()]
    )
    if oldret is not None:
        ret = oldret

    lock = threading.Lock()

    # Here we use byte as unit for larger task chunks
    bytes_iter = iter(all_bytes)
    def next_byte_fn():
        with lock:
            try:
                byteoff = next(bytes_iter)
                while byteoff in ret.retcoll_map:
                    byteoff = next(bytes_iter)
                return byteoff
            except StopIteration:
                return None

    nbytes_out_of_chunk_done = 0
    if args.nchunks is not None:
        # Count the bytes that've been done but don't belong to the current chunk
        nbytes_out_of_chunk_done = len(set(ret.retcoll_map) - set(all_bytes))

    last_speed_time = time.time()
    last_speed_nbytes = len(ret.retcoll_map)
    last_save_time = time.time()
    def on_byte_done(byteoff, byte_rets):
        nonlocal last_speed_time, last_speed_nbytes, last_save_time
        curr_time = time.time()
        with lock:
            ret.retcoll_map[byteoff] = byte_rets
            nbytes_done = len(ret.retcoll_map) - nbytes_out_of_chunk_done
            should_show_speed = curr_time - last_speed_time > 10
            should_save = curr_time - last_save_time > args.save_interval or nbytes_done == total_bytes
            if should_show_speed or should_save:
                elapsed = curr_time - last_speed_time
                speed = (nbytes_done - last_speed_nbytes) / elapsed
                if speed != 0:
                    print(f'⏱️ Speed: {speed:.2f} bytes/s; '
                          f'remaining: {(total_bytes - nbytes_done)/speed/86400:.2f} days -----')
                last_speed_time = curr_time
                last_speed_nbytes = nbytes_done
            if should_save:
                print(f'💾 Saved: {nbytes_done}/{total_bytes} bytes '
                      f'({nbytes_done/total_bytes:.2%}) -----')
                last_save_time = curr_time
                if os.path.exists(outfile):
                    shutil.move(outfile, f'{outfile}.bak')
                utils.save(ret, outfile, merge=False)

    nworkers = args.nworkers
    with ThreadPoolExecutor(max_workers=nworkers) as executor:
        futures = [
            executor.submit(flip_thread_work, next_byte_fn, on_byte_done, bis, args)
            for _ in range(nworkers)
        ]
        [f.result() for f in futures]

    print(f'Finished sweeping.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--compiler', type=str, default='tvm')
    parser.add_argument('-v', '--compiler-version', type=str, default='main')
    parser.add_argument('-m', '--model-name', type=str, default='resnet50')
    parser.add_argument('-d', '--datasets', type=str, nargs='+', default=['CIFAR10'])
    parser.add_argument('-X', '--no-avx', action='store_true', default=False)
    parser.add_argument('-O', '--opt-level', type=int, default=3)
    parser.add_argument('-g', '--gan', action='store_true', default=False)
    parser.add_argument('-F', '--no-fast', action='store_false', dest='fast', help='Use fewer images for faster testing')
    parser.add_argument('-w', '--nworkers', type=int, default=0)
    parser.add_argument('-a', '--acc-thresh', type=float, default=100, help='Only keep flips resulting in <= this accuracy')
    parser.add_argument('-l', '--label-change-thresh', type=float, default=0, help='Only keep flips resulting in >= this percentage of label changes')
    parser.add_argument('-C', '--completing-file', help='Sweep result file to complete (with new datasets)')
    parser.add_argument('-t', '--timeout-x', type=int, default=0, help='Timeout multiplier (0 for auto)')
    parser.add_argument('-T', '--timeout', type=int, default=0, help='Timeout in seconds (0 for auto)')
    parser.add_argument('-D', '--device', type=str, default='cpu')
    parser.add_argument('-M', '--max-mem-gb', type=float, default=24)
    parser.add_argument('-n', '--nchunks', type=int, default=None, help='Number of chunks to split the sweep into')
    parser.add_argument('-k', '--chunk-idx', type=int, default=None, help='Chunk index (0-based)')
    parser.add_argument('--specific-bits-file', type=str, default=None, help='File containing specific bits to flip - overrides the default bytes/bits list')
    parser.add_argument('--gan-images-only', action='store_true', default=False, help="Just output GAN images, don't calculate metrics")
    parser.add_argument('-s', '--save-interval', type=int, default=300)
    parser.add_argument('-o', '--out-dir', type=str, default=f'{cfg.sweep_dir}')
    args = parser.parse_args()

    # RLIMIT_DATA may require Linux >= 4.7
    resource.setrlimit(resource.RLIMIT_DATA, (int(args.max_mem_gb * 1024**3), -1))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    if args.nworkers == 0:
        args.nworkers = {
            'tvm': 5,
            'glow': psutil.cpu_count(logical=False),
        }[args.compiler]

    if args.timeout_x == 0:
        args.timeout_x = {
            'tvm': 20 + args.nworkers,
            'glow': 5 + args.nworkers // 15,
        }[args.compiler]

    sweep(args)
