#! /usr/bin/env python3

import os
import argparse
import random

# TODO: If we import utils.py the program may crash with -6
# See https://discuss.tvm.apache.org/t/free-invalid-pointer-aborted/11357
import utils

import cfg
import modman
import dataman
from eval import evalutils
from typing import Any, NamedTuple
from fliputils import *
import utils

class ClassifierFlipResult(NamedTuple):
    # For legacy code
    base10_offset: int
    bitidx: int
    correct_pct: float
    top_labels: Any

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--compiler', type=str, default='tvm')
    parser.add_argument('-v', '--compiler-version', type=str, default='main')
    parser.add_argument('-m', '--model-name', type=str, default='googlenet')
    parser.add_argument('-d', '--dataset', type=str, default='CIFAR10')
    parser.add_argument('-s', '--seed', help='Random seed', type=int, default=42)
    parser.add_argument('-n', '--nbits', help='Number of bits to flip', type=int)
    parser.add_argument('-p', '--pause', help='Pause before flipping', action='store_true')
    parser.add_argument('-b', '--byteidx', default=None)
    parser.add_argument('-B', '--bitidx', default=None)
    parser.add_argument('-X', '--no-avx', action='store_true', default=False)
    parser.add_argument('-O', '--opt-level', type=int, default=3)
    parser.add_argument('-g', '--gan', action='store_true', default=False)
    parser.add_argument('-f', '--fast', action='store_true', default=False, help='Use fewer images for faster testing')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-D', '--device', type=str, default='cpu')
    args = parser.parse_args()

    random.seed(args.seed)

    bi = utils.BinaryInfo(
        args.compiler, args.compiler_version,
        args.model_name, args.dataset, avx=(not args.no_avx), opt_level=args.opt_level
    )
    lmi = load_mod(bi)

    if args.gan:
        gan_evaluator = evalutils.GANEvaluator(bi.dataset, device=args.device)
        gan_evaluator.set_ref(lmi.mod)
    else:
        if args.fast:
            val_loader = dataman.get_sampling_loader_v2(args.dataset, bi.input_img_size, 'test', cfg.batch_size, n_per_class=10)
        else:
            val_loader = dataman.get_benign_loader(args.dataset, bi.input_img_size, 'test', cfg.batch_size)

    if args.pause:
        print(f'PID: {os.getpid()}')
        input('Press enter to continue...')

    flips = []
    if args.nbits: # Random flips
        flips = random_flip_bits(lmi, args.nbits)
    elif args.byteidx is not None and args.bitidx is not None: # Specific flips
        flips = [new_flip(args.byteidx, args.bitidx)]
        flip_bits(lmi, flips, quiet=args.quiet)
    else:
        assert False, 'Nothing to do'

    if args.gan:
        outputs = gan_evaluator.get_gan_outputs(
            lmi.mod, debug=True, debug_fname=f'{flips[0].byteidx:x}_{flips[0].bitidx}'
        )
        top_labels, lchange, lpips_avg, fid = gan_evaluator.eval(outputs)
        print(f'Top labels: {top_labels.flatten().tolist()}')
        print(f'Label change: {lchange:.2%}, LPIPS: {lpips_avg:.2f}, FID: {fid:.2f}')
    else:
        correct_pct = assess(lmi, val_loader)
        print(f'Acc. after flipping: {correct_pct:.2f}%')
