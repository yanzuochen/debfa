#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import argparse

import utils
import flipsweep

parser = argparse.ArgumentParser()
parser.add_argument('nbytes', type=int)
parser.add_argument('sweepfile')
args = parser.parse_args()

sr = flipsweep.load_sweep_result(args.sweepfile)
if len(sr.retcoll_map) <= args.nbytes:
    print(f'No need to truncate {args.sweepfile} (already {len(sr.retcoll_map)} bytes)')
    sys.exit(0)

print(f'Truncating {args.sweepfile} to the first {args.nbytes} byte entries')
sr.retcoll_map = {k: v for i, (k, v) in enumerate(sr.retcoll_map.items()) if i < args.nbytes}
utils.save(sr, args.sweepfile, merge=False)
