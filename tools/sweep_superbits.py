#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import subprocess
import utils
import cfg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('base_fake_sweep', type=str)
    parser.add_argument('--nfakesets', type=int, default=12)
    parser.add_argument('--sweep-dir', type=str, default=cfg.sweep_dir)
    parser.add_argument('--label-change-thresh', type=float, default=50)
    args = parser.parse_args()

    bi = utils.BinaryInfo.from_fname(
        os.path.basename(args.base_fake_sweep).replace('-sweep.pkl', '.so')
    )
    fake_prefix, base_fake_id = bi.dataset.split('_')
    base_fake_id = int(base_fake_id)

    for i in range(1, args.nfakesets):
        curr_fakesets = [f'{fake_prefix}_{base_fake_id + j}' for j in range(i + 1)]
        cmd_args = [
            '--compiler', bi.compiler, '--compiler-version', bi.compiler_ver,
            '--model-name', bi.model_name, '--datasets', *curr_fakesets,
            '--opt-level', str(bi.opt_level),
            '--completing-file',
            args.base_fake_sweep.replace(f'{fake_prefix}_{base_fake_id}', '+'.join(curr_fakesets[:-1])),
            '--label-change-thresh', str(args.label_change_thresh),
        ] + (
            ['--no-avx'] if not bi.avx else []
        ) + (
            ['--gan'] if bi.is_gan else []
        )
        print(f'Running flipsweep.py {" ".join(cmd_args)}')
        subprocess.run(
            [f'{cfg.project_root}/flipsweep.py'] + cmd_args, check=True,
        )
