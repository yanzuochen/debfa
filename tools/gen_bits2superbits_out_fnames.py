#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import utils
import cfg

if __name__ == '__main__':
    generated_out_fnames = []

    for max_fake_bits_path in sys.argv[1:]:
        # A max_fake_bits_path is a path to a bits.json file containing all
        # fake datasets (e.g. fake_0+...+fake_9 if there're 10 fake datasets).
        max_bi = utils.BinaryInfo.from_fname(os.path.basename(max_fake_bits_path).replace('-bits.json', '.x'))

        def gen_args_for_base_bi(base_bi):
            assert base_bi.just_datasets_differ(max_bi)
            fakesets = max_bi.dataset.split('+')
            ret = []
            for i in range(len(fakesets)):
                # CIFAR10@fake_0, CIFAR10@fake_0+fake_1, ...
                bi = base_bi._replace(dataset=f'{base_bi.dataset}@{"+".join(fakesets[:i+1])}')
                ret.append(bi.fname.replace(".so", "-superbits.csv"))
            return ret

        for base_bi in cfg.all_build_bis:
            if base_bi.dataset.startswith('fake'):
                continue
            if not base_bi.just_datasets_differ(max_bi):
                continue
            base_bits = f'{cfg.bits_dir}/{base_bi.fname.replace(".so", "-bits.json")}'
            if not os.path.exists(base_bits):
                continue
            generated_out_fnames.extend(gen_args_for_base_bi(base_bi))

    print(' '.join(generated_out_fnames))
