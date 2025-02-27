#! /usr/bin/env python3

import os
import sys
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import shutil

import utils
import flipsweep

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('base_file')
    parser.add_argument('updates', nargs='+')
    args = parser.parse_args()

    base = flipsweep.load_sweep_result(args.base_file)
    for update in args.updates:
        print(f'Applying {update}')
        update = flipsweep.read_existing_sweepret(update, base.args)
        base.retcoll_map.update(update.retcoll_map)
    print('Done')

    print(f'Backing up {args.base_file} to {args.base_file}.bak')
    shutil.copyfile(args.base_file, f'{args.base_file}.bak')
    utils.save(base, args.base_file, merge=False)
    print(f'Wrote updated sweepret to {args.base_file}')
