#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

from eval import evalutils

if __name__ == '__main__':
    files = sys.argv[1:]
    for f in files:
        print(f'Checking accuracy of {f}')
        acc = evalutils.check_so_acc(os.path.realpath(f))
        print(f'----------------------------------')
