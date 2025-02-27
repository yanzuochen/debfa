#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import modman
import utils
import cfg

output_root = f'{cfg.debug_dir}/llvm-ir'

if __name__ == '__main__':
    files = sys.argv[1:]
    for f in files:
        bi = utils.BinaryInfo.from_fname(os.path.basename(f))
        mod, params = modman.get_irmod(
            bi.model_name, bi.dataset, 'none', cfg.batch_size, bi.input_img_size, nchannels=bi.nchans
        )
        _, ir = modman.get_json_and_ir(mod, params, opt_level=bi.opt_level)
        outfile = f'{output_root}/{os.path.basename(f)}.llvm'
        utils.ensure_dir_of(outfile)
        with open(outfile, 'w+') as f:
            f.write(ir)
            f.write(f'\n; {"vim"}: set ft=llvm:\n')
        print(f'Wrote IR to {outfile}')
